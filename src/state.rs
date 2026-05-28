// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            Ordering,
        },
    },
    thread,
    time::Duration,
};

use crossbeam_channel::{
    Receiver,
    Sender,
    bounded,
};
use parking_lot::{
    Mutex,
    RwLock,
};

use crate::{
    compact::flush_memtable,
    compaction::{
        CompactionManager,
        SchedulerConfig,
        SegmentRegistry,
    },
    errs::{
        CesiumError,
        FsError,
    },
    levels::KeyRange,
    manifest_reader::ManifestReader,
    manifest_writer::ManifestWriter,
    memtable::{
        DEFAULT_MEMTABLE_SIZE_IN_BYTES,
        Memtable,
    },
    version::{
        VersionEdit,
        VersionManager,
    },
};

pub const DEFAULT_BLOCK_SIZE: u64 = 4096;
pub const DEFAULT_TARGET_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;
pub const DEFAULT_NUM_MEMTABLES: u64 = 4;

/// The default set of database options.
#[derive(Debug, Clone)]
pub struct DbStorageBuilder {
    /// The size of a given disk block. It's recommended to leave the default
    /// for NVMe drives.
    pub block_size: u64,
    /// The target size of the disk files. This is a soft limit.
    pub target_segment_size: u64,
    /// The amount of tables to hold in-memory before flushing to disk.
    pub num_memtable_limit: u64,
    /// The maximum size of a memtable in bytes before it is frozen.
    pub memtable_size: u64,
    /// Compaction scheduler configuration.
    pub scheduler_config: SchedulerConfig,
    /// Base path for database storage
    pub base_path: Option<PathBuf>,
}

impl DbStorageBuilder {
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn new() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            target_segment_size: DEFAULT_TARGET_SEGMENT_SIZE,
            num_memtable_limit: DEFAULT_NUM_MEMTABLES,
            memtable_size: DEFAULT_MEMTABLE_SIZE_IN_BYTES,
            scheduler_config: SchedulerConfig::default(),
            base_path: None,
        }
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn block_size(mut self, block_size: u64) -> Self {
        self.block_size = block_size;
        self
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn target_segment_size(mut self, target_segment_size: u64) -> Self {
        self.target_segment_size = target_segment_size;
        self
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn num_memtable_limit(mut self, num_memtable_limit: u64) -> Self {
        self.num_memtable_limit = num_memtable_limit;
        self
    }

    pub fn memtable_size(mut self, memtable_size: u64) -> Self {
        self.memtable_size = memtable_size;
        self
    }

    pub fn scheduler_config(mut self, config: SchedulerConfig) -> Self {
        self.scheduler_config = config;
        self
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn base_path(mut self, path: PathBuf) -> Self {
        self.base_path = Some(path);
        self
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn build(self) -> Result<Mutex<DbStorageState>, CesiumError> {
        Ok(Mutex::new(DbStorageState::new(self)?))
    }
}

impl Default for DbStorageBuilder {
    fn default() -> Self {
        DbStorageBuilder::new()
    }
}

/// Default number of LSM-tree levels (L1-L7)
pub const DEFAULT_NUM_LEVELS: usize = 7;

// TODO(@siennathesane): all universal ids (memtable, segment, etc.) need to be
// monotonically increasing
pub struct DbStorageState {
    curr_memtable: RwLock<Arc<Memtable>>,
    frozen_memtables: Arc<Mutex<Vec<Arc<Memtable>>>>,
    /// Version manager for LSM-tree level coordination
    pub version_manager: Arc<VersionManager>,
    /// Compaction manager for background compaction
    pub compaction_manager: Option<Arc<Mutex<CompactionManager>>>,
    /// Manifest writer for crash recovery
    manifest: Option<Arc<Mutex<ManifestWriter>>>,
    /// Maximum number of frozen memtables before flushing to disk
    num_memtable_limit: u64,
    /// Maximum size of a memtable in bytes before it is frozen
    memtable_size: u64,
    /// Base path for Segment storage
    base_path: Option<Arc<PathBuf>>,
    /// Shutdown signal for background flusher
    shutdown: Arc<AtomicBool>,
    /// Background flusher thread handle
    flusher_thread: Option<thread::JoinHandle<()>>,
    /// Notification channel to wake the flusher when new frozen memtables
    /// arrive
    flush_notify_tx: Option<Sender<()>>,
}

impl DbStorageState {
    fn new(opts: DbStorageBuilder) -> Result<Self, CesiumError> {
        let frozen_memtables = Arc::new(Mutex::new(vec![]));
        let shutdown = Arc::new(AtomicBool::new(false));
        let (flush_tx, flush_rx) = bounded::<()>(1);
        let base_path = opts.base_path.map(Arc::new);

        // Recover version set from manifest (if exists) or create new
        let version_manager = if let Some(ref path) = base_path {
            match ManifestReader::recover_version_set(path.as_ref(), DEFAULT_NUM_LEVELS) {
                | Ok(Some(version_set)) => {
                    tracing::info!("Recovered version set from manifest");
                    Arc::new(VersionManager::with_version(version_set))
                },
                | Ok(None) => {
                    tracing::info!("No manifest found, starting fresh");
                    Arc::new(VersionManager::new(DEFAULT_NUM_LEVELS))
                },
                | Err(e) => {
                    tracing::warn!("Failed to recover from manifest: {:?}, starting fresh", e);
                    Arc::new(VersionManager::new(DEFAULT_NUM_LEVELS))
                },
            }
        } else {
            Arc::new(VersionManager::new(DEFAULT_NUM_LEVELS))
        };

        // Initialize or open manifest writer
        let manifest = if let Some(ref path) = base_path {
            // Ensure base directory exists
            std::fs::create_dir_all(path.as_ref())
                .map_err(|e| CesiumError::FsError(FsError::IoError(e)))?;

            let manifest_path = path.as_ref();
            let writer = if manifest_path.join("MANIFEST").exists() {
                ManifestWriter::open_existing(path.as_ref().clone())
                    .map_err(CesiumError::ManifestError)?
            } else {
                ManifestWriter::create(path.as_ref().clone(), 0)
                    .map_err(CesiumError::ManifestError)?
            };
            Some(Arc::new(Mutex::new(writer)))
        } else {
            None
        };

        // Initialize segment registry and compaction manager if base_path is provided
        let registry = base_path
            .as_ref()
            .map(|path| Arc::new(SegmentRegistry::new(path.as_ref().clone())));

        let compaction_manager = if let Some(ref path) = base_path {
            let reg = registry.as_ref().unwrap().clone();
            let mut manager = CompactionManager::new_with_scheduler_config(
                path.as_ref().clone(),
                Arc::clone(&version_manager),
                manifest.clone(),
                reg,
                opts.scheduler_config.clone(),
            );
            manager.start(); // Start background compaction thread
            Some(Arc::new(Mutex::new(manager)))
        } else {
            None
        };

        // Spawn background flusher thread if we have a base_path
        let flusher_thread = if let Some(ref path) = base_path {
            let frozen_clone = Arc::clone(&frozen_memtables);
            let version_mgr_clone = Arc::clone(&version_manager);
            let compaction_mgr_clone = compaction_manager.clone();
            let manifest_clone = manifest.clone();
            let shutdown_clone = Arc::clone(&shutdown);
            let path_clone = Arc::clone(path);
            let flush_rx = flush_rx.clone();
            let registry_clone = registry.clone();

            Some(thread::spawn(move || {
                Self::background_flusher(
                    frozen_clone,
                    version_mgr_clone,
                    compaction_mgr_clone,
                    manifest_clone,
                    shutdown_clone,
                    path_clone,
                    flush_rx,
                    registry_clone,
                );
            }))
        } else {
            None
        };

        Ok(Self {
            curr_memtable: RwLock::new(Arc::new(Memtable::new(0, opts.memtable_size))),
            frozen_memtables,
            version_manager,
            compaction_manager,
            manifest,
            num_memtable_limit: opts.num_memtable_limit,
            memtable_size: opts.memtable_size,
            base_path: base_path.clone(),
            shutdown,
            flusher_thread,
            flush_notify_tx: if base_path.is_some() {
                Some(flush_tx)
            } else {
                None
            },
        })
    }

    /// Background thread that flushes frozen memtables to disk
    fn background_flusher(
        frozen_memtables: Arc<Mutex<Vec<Arc<Memtable>>>>,
        version_manager: Arc<VersionManager>,
        compaction_manager: Option<Arc<Mutex<CompactionManager>>>,
        manifest: Option<Arc<Mutex<ManifestWriter>>>,
        shutdown: Arc<AtomicBool>,
        base_path: Arc<PathBuf>,
        _flush_rx: Receiver<()>,
        registry: Option<Arc<SegmentRegistry>>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // Check if we need to flush
            let frozen_count = frozen_memtables.lock().len();
            let should_flush = frozen_count > 0;

            if should_flush {
                // Flush all available frozen memtables in this iteration to
                // drain the queue faster when writers are outpacing the flusher.
                loop {
                    let memtable_to_flush = {
                        let frozen = frozen_memtables.lock();
                        if frozen.is_empty() {
                            break;
                        }
                        frozen[0].clone()
                    };

                    // Generate unique Segment ID
                    let segment_id = version_manager.next_segment_id();

                    // Build path: base_path/segments/<id>/
                    let segment_path = base_path.join("segments").join(segment_id.to_string());

                    // NOTE: We intentionally do NOT stall the flusher when L0 is full.
                    // The flusher must always drain frozen memtables to prevent
                    // unbounded memory growth. Write backpressure is handled by
                    // the frozen memtable limit in the write path (DbInner::batch).

                    // Flush memtable to disk
                    match flush_memtable(
                        memtable_to_flush.clone(),
                        segment_path.clone(),
                        segment_id,
                    ) {
                        | Ok((segment, min_key, max_key)) => {
                            // Create KeyRange before moving min/max into manifest edit
                            let key_range =
                                KeyRange::new(min_key.clone(), max_key.clone(), segment_id);

                            // Log to manifest BEFORE updating version (write-ahead)
                            if let Some(ref manifest_writer) = manifest {
                                let edit = VersionEdit::AddL0Segment {
                                    segment_id,
                                    key_range: (min_key, max_key),
                                    size: segment.size_in_bytes(),
                                };

                                // Acquire lock ONCE and hold it for all operations
                                let mut manifest_guard = manifest_writer.lock();
                                match manifest_guard.append_edit(&edit) {
                                    | Ok(()) => {
                                        // Sync manifest every 10 edits for durability
                                        if manifest_guard.entry_count() % 10 == 0 {
                                            let _ = manifest_guard.sync();
                                        }
                                    },
                                    | Err(e) => {
                                        tracing::error!(error = ?e, "Failed to write to manifest");
                                    },
                                }
                                // Lock released here when manifest_guard goes
                                // out of scope
                            }

                            // Register the new L0 Segment with VersionManager
                            version_manager.update(|version| {
                                version.add_to_l0(segment.clone(), key_range);
                            });

                            // Register the segment with the registry for lifecycle tracking
                            if let Some(ref reg) = registry {
                                reg.register(segment, segment_path.clone());
                                let (deleted, bytes_freed) = reg.cleanup();
                                if deleted > 0 {
                                    tracing::info!(
                                        segments_deleted = deleted,
                                        bytes_freed = bytes_freed,
                                        "Cleaned up obsolete segments after flush"
                                    );
                                }
                            }

                            // Only remove if this memtable is still at the front
                            // (another thread may have already removed it)
                            let mut frozen = frozen_memtables.lock();
                            if !frozen.is_empty() && Arc::ptr_eq(&frozen[0], &memtable_to_flush) {
                                frozen.remove(0);
                            }

                            tracing::info!(
                                segment_id = segment_id,
                                memtable_id = memtable_to_flush.id(),
                                "Flushed memtable to L0 Segment"
                            );

                            // Notify compaction manager that a new L0 file was created
                            if let Some(ref manager) = compaction_manager {
                                manager.lock().notify_flush();
                            }
                        },
                        | Err(e) => {
                            tracing::error!(
                                error = ?e,
                                segment_id = segment_id,
                                memtable_id = memtable_to_flush.id(),
                                "Failed to flush memtable to disk - keeping in memory"
                            );
                            // Stop trying to flush this batch on error
                            break;
                        },
                    }
                }
            } else {
                // No flush needed - sleep briefly
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn current_memtable(&self) -> Arc<Memtable> {
        self.curr_memtable.read().clone()
    }

    pub fn should_stall_writes(&self) -> bool {
        self.compaction_manager
            .as_ref()
            .map(|m| m.lock().should_stall_writes())
            .unwrap_or(false)
    }

    /// Returns a snapshot of frozen memtables for scanning.
    ///
    /// Returns a clone of the frozen memtables vector, allowing callers to
    /// iterate over them without holding the lock.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn frozen_memtables_for_scan(&self) -> Vec<Arc<Memtable>> {
        self.frozen_memtables.lock().clone()
    }

    /// Searches frozen memtables for a key (newest to oldest).
    ///
    /// Returns the value if found, respecting tombstones.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn get_from_frozen(
        &self,
        key: &crate::keypair::KeyBytes,
    ) -> Option<crate::keypair::ValueBytes> {
        let frozen = self.frozen_memtables.lock();

        // Search from newest (end) to oldest (front)
        for memtable in frozen.iter().rev() {
            if let Some(val) = memtable.get(key) {
                return Some(val);
            }
        }
        None
    }

    /// Returns the number of frozen memtables
    pub fn frozen_count(&self) -> usize {
        self.frozen_memtables.lock().len()
    }

    /// Returns a clone of the frozen-memtables Arc so callers can access the
    /// queue without locking the outer `DbStorageState`.
    pub fn frozen_memtables_arc(&self) -> Arc<Mutex<Vec<Arc<Memtable>>>> {
        Arc::clone(&self.frozen_memtables)
    }

    /// Returns the maximum number of frozen memtables before writes stall
    pub fn memtable_limit(&self) -> u64 {
        self.num_memtable_limit
    }

    /// This generates a new memtable and swaps the existing one.
    ///
    /// The old memtable is frozen and added to the queue. The background
    /// flusher thread will write it to disk when the queue exceeds the limit.
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn new_memtable(&mut self) {
        let next_id = self.curr_memtable.read().clone().id() + 1;
        let new_table = RwLock::new(Arc::new(Memtable::new(next_id, self.memtable_size)));

        // Freeze current memtable and add to frozen queue
        let frozen_memtable = self.curr_memtable.read().clone();
        frozen_memtable.freeze();
        self.frozen_memtables.lock().push(frozen_memtable);

        self.curr_memtable = new_table;

        // Wake the background flusher if it's sleeping
        if let Some(ref tx) = self.flush_notify_tx {
            let _ = tx.try_send(());
        }
    }

    /// Triggers a manual compaction of the entire database
    #[cfg_attr(feature = "telemetry", tracing::instrument(skip_all, level = "debug"))]
    pub fn compact(&self) {
        if let Some(ref manager) = self.compaction_manager {
            manager.lock().compact();
        }
    }

    /// Returns compaction statistics
    pub fn compaction_stats(&self) -> Option<crate::compaction::CompactionStats> {
        self.compaction_manager.as_ref().map(|m| m.lock().stats())
    }

    /// Returns version statistics (L0 segments, total size, etc.)
    pub fn version_stats(&self) -> crate::version::VersionStats {
        self.version_manager.stats()
    }

    /// Performs an orderly shutdown of the database storage layer.
    ///
    /// 1. Freezes the current memtable and waits for background flusher to
    ///    drain
    /// 2. Shuts down the compaction manager
    /// 3. Syncs manifest to disk
    /// 4. Runs final segment cleanup
    pub fn shutdown(&mut self) -> Result<(), crate::errs::CesiumError> {
        // 1. Signal shutdown first so flusher stops
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.flusher_thread.take() {
            let _ = handle.join();
        }

        // 2. Freeze current memtable
        self.new_memtable();

        // 3. Drain any remaining frozen memtables ourselves
        self.drain_frozen_memtables();

        // 4. Drop compaction manager (joins bg_thread + executor)
        self.compaction_manager.take();

        // 5. Sync manifest to ensure all edits are persisted
        if let Some(ref manifest) = self.manifest
            && let Err(e) = manifest.lock().sync() {
                tracing::error!(error = ?e, "Failed to sync manifest during shutdown");
            }

        Ok(())
    }

    /// Syncs all data to disk by freezing the current memtable and flushing
    /// all frozen memtables directly.
    pub fn sync(&mut self) -> Result<(), crate::errs::CesiumError> {
        // Freeze current memtable to ensure all in-memory data gets flushed
        self.new_memtable();

        // Flush all frozen memtables directly (don't wait for bg flusher)
        self.drain_frozen_memtables();

        // Sync manifest to ensure all version edits are persisted
        if let Some(ref manifest) = self.manifest
            && let Err(e) = manifest.lock().sync() {
                tracing::error!(error = ?e, "Failed to sync manifest during sync()");
            }

        Ok(())
    }

    /// Drains all frozen memtables by flushing them to disk inline.
    fn drain_frozen_memtables(&mut self) {
        let base_path = match &self.base_path {
            | Some(p) => p.clone(),
            | None => return,
        };

        loop {
            let memtable = {
                let frozen = self.frozen_memtables.lock();
                if frozen.is_empty() {
                    break;
                }
                frozen[0].clone()
            };

            let segment_id = self.version_manager.next_segment_id();
            let segment_path = base_path.join("segments").join(segment_id.to_string());

            match flush_memtable(memtable.clone(), segment_path, segment_id) {
                | Ok((segment, min_key, max_key)) => {
                    // Create KeyRange before moving min/max into manifest edit
                    let key_range = KeyRange::new(min_key.clone(), max_key.clone(), segment_id);

                    // Log to manifest BEFORE updating version (write-ahead)
                    if let Some(ref manifest_writer) = self.manifest {
                        let edit = VersionEdit::AddL0Segment {
                            segment_id,
                            key_range: (min_key, max_key),
                            size: segment.size_in_bytes(),
                        };

                        // Store result to drop lock before checking entry_count
                        let result = manifest_writer.lock().append_edit(&edit);
                        match result {
                            | Ok(()) => {
                                // Sync manifest every 10 edits for durability
                                if manifest_writer.lock().entry_count() % 10 == 0 {
                                    let _ = manifest_writer.lock().sync();
                                }
                            },
                            | Err(e) => {
                                tracing::error!(error = ?e, "Failed to write to manifest during drain");
                            },
                        }
                    }

                    // Register the new L0 Segment with VersionManager
                    self.version_manager.update(|version| {
                        version.add_to_l0(segment.clone(), key_range);
                    });

                    // Only remove if this memtable is still at the front
                    // (another thread may have already removed it)
                    let mut frozen = self.frozen_memtables.lock();
                    if !frozen.is_empty() && Arc::ptr_eq(&frozen[0], &memtable) {
                        frozen.remove(0);
                    }
                },
                | Err(e) => {
                    tracing::error!(error = ?e, "Failed to flush memtable during drain");
                    break;
                },
            }
        }
    }
}

impl Drop for DbStorageState {
    fn drop(&mut self) {
        // Signal shutdown to flusher
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for flusher thread to exit
        if let Some(handle) = self.flusher_thread.take() {
            let _ = handle.join();
        }

        // Drop compaction manager (its Drop impl joins bg_thread + shuts down executor)
        self.compaction_manager.take();
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use crate::{
        keypair::{
            DEFAULT_NS,
            KeyBytes,
            ValueBytes,
        },
        state::DbStorageBuilder,
    };

    #[test]
    fn test_new_memtable() {
        let state = DbStorageBuilder::default().build().unwrap();

        assert!(state.lock().frozen_memtables.lock().is_empty());

        state.lock().new_memtable();
    }

    #[test]
    fn test_memtable_swap() {
        let state = DbStorageBuilder::default().build().unwrap();

        let initial_id = state.lock().current_memtable().id();
        assert_eq!(initial_id, 0, "initial memtable should have id 0");

        // swap to new memtable
        state.lock().new_memtable();

        let new_id = state.lock().current_memtable().id();
        assert_eq!(new_id, 1, "new memtable should have id 1");

        // frozen memtables should contain the old one
        let frozen = state.lock().frozen_memtables.lock().clone();
        assert_eq!(frozen.len(), 1, "should have 1 frozen memtable");
        assert_eq!(frozen[0].id(), 0, "frozen memtable should have id 0");
    }

    #[test]
    fn test_multiple_memtable_swaps() {
        let state = DbStorageBuilder::default().build().unwrap();

        const NUM_SWAPS: u64 = 5;

        for i in 0..NUM_SWAPS {
            let current_id = state.lock().current_memtable().id();
            assert_eq!(current_id, i);

            state.lock().new_memtable();

            let new_id = state.lock().current_memtable().id();
            assert_eq!(new_id, i + 1);
        }

        // verify all old memtables are frozen
        let frozen = state.lock().frozen_memtables.lock().clone();
        assert_eq!(frozen.len(), NUM_SWAPS as usize);

        // verify frozen memtables have correct ids
        for (idx, memtable) in frozen.iter().enumerate() {
            assert_eq!(memtable.id(), idx as u64);
        }
    }

    #[test]
    fn test_current_memtable_returns_same_instance() {
        let state = DbStorageBuilder::default().build().unwrap();

        let mt1 = state.lock().current_memtable();
        let mt2 = state.lock().current_memtable();

        // should return the same Arc instance
        assert_eq!(mt1.id(), mt2.id());
    }

    #[test]
    fn test_frozen_memtables_preserve_data() {
        let state = DbStorageBuilder::default().build().unwrap();

        // write data to first memtable
        let key = KeyBytes::new(DEFAULT_NS, Bytes::from("test-key"), 1000);
        let val = ValueBytes::new(DEFAULT_NS, Bytes::from("test-value"));
        {
            let current = state.lock().current_memtable();
            assert!(current.put(key.clone(), val.clone()).is_ok());
        }

        // swap to new memtable
        state.lock().new_memtable();

        // verify data is still accessible in frozen memtable
        let frozen = state.lock().frozen_memtables.lock().clone();
        assert_eq!(frozen.len(), 1);

        let retrieved = frozen[0].get(&key);
        assert!(
            retrieved.is_some(),
            "data should be preserved in frozen memtable"
        );
        assert_eq!(retrieved.unwrap().as_bytes(), val.as_bytes());
    }

    #[test]
    fn test_storage_builder_custom_config() {
        let custom_block_size = 8192;
        let custom_segment_size = 16384;
        let custom_memtable_limit = 8;

        let state = DbStorageBuilder::new()
            .block_size(custom_block_size)
            .target_segment_size(custom_segment_size)
            .num_memtable_limit(custom_memtable_limit)
            .build().unwrap();

        // verify state is created successfully
        let current = state.lock().current_memtable();
        assert_eq!(current.id(), 0);
    }

    #[test]
    fn test_storage_builder_chain() {
        let state = DbStorageBuilder::new()
            .block_size(4096)
            .target_segment_size(8192)
            .num_memtable_limit(6)
            .build().unwrap();

        assert_eq!(state.lock().current_memtable().id(), 0);
    }

    #[test]
    fn test_memtable_id_monotonic_increase() {
        let state = DbStorageBuilder::default().build().unwrap();

        let mut prev_id = 0;
        for _ in 0..10 {
            state.lock().new_memtable();
            let current_id = state.lock().current_memtable().id();
            assert!(
                current_id > prev_id,
                "memtable ids should monotonically increase"
            );
            prev_id = current_id;
        }
    }
}
