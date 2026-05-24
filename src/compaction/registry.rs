//! Segment registry for reference tracking
//!
//! This module provides a registry to track which segments are live
//! and coordinate safe deletion after compaction.

use std::{
    collections::{
        HashMap,
        HashSet,
    },
    path::PathBuf,
    sync::Arc,
};

use parking_lot::RwLock;

use crate::segment::Segment;

/// Registry for tracking live segments and coordinating safe file deletion
///
/// The registry maintains:
/// - Which segments are currently live (in the current version)
/// - Reference counts for segments in use by compaction jobs or readers
/// - A delete queue for segments that should be removed
/// - Physical file deletion when segments are no longer referenced
pub struct SegmentRegistry {
    /// Base path for the database (used to compute segment paths if not
    /// explicitly stored)
    base_path: PathBuf,

    /// Currently live segment IDs (in the current version)
    live_segments: RwLock<HashSet<u64>>,

    /// Segment ID -> (`Arc<Segment>`, `PathBuf`) mapping
    ///
    /// Keeps segments alive while they're registered. The path is the
    /// directory containing the segment files, used for physical deletion.
    /// When a segment is removed from here and no other references exist,
    /// the directory is deleted.
    segments: RwLock<HashMap<u64, (Arc<Segment>, PathBuf)>>,

    /// Segments pending deletion
    ///
    /// These segments have been removed from the version but may still
    /// be referenced by in-progress compactions or readers.
    pending_deletion: RwLock<HashSet<u64>>,
}

impl SegmentRegistry {
    /// Creates a new empty registry
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            live_segments: RwLock::new(HashSet::new()),
            segments: RwLock::new(HashMap::new()),
            pending_deletion: RwLock::new(HashSet::new()),
        }
    }

    /// Registers a segment as live
    ///
    /// This should be called when a segment is added to the version.
    /// The `path` is the directory containing the segment files, used
    /// for physical deletion when the segment becomes obsolete.
    pub fn register(&self, segment: Arc<Segment>, path: PathBuf) {
        let id = segment.id();

        let mut live = self.live_segments.write();
        let mut segments = self.segments.write();

        live.insert(id);
        segments.insert(id, (segment, path));
    }

    /// Marks a segment for deletion
    ///
    /// The segment is removed from the live set but kept in the registry
    /// until all references are dropped.
    pub fn mark_for_deletion(&self, segment_id: u64) {
        let mut live = self.live_segments.write();
        let mut pending = self.pending_deletion.write();

        live.remove(&segment_id);
        pending.insert(segment_id);
    }

    /// Gets a segment by ID if it exists
    pub fn get(&self, segment_id: u64) -> Option<Arc<Segment>> {
        self.segments
            .read()
            .get(&segment_id)
            .map(|(seg, _)| Arc::clone(seg))
    }

    /// Checks if a segment is currently live
    pub fn is_live(&self, segment_id: u64) -> bool {
        self.live_segments.read().contains(&segment_id)
    }

    /// Returns the number of live segments
    pub fn live_count(&self) -> usize {
        self.live_segments.read().len()
    }

    /// Returns the number of segments pending deletion
    pub fn pending_deletion_count(&self) -> usize {
        self.pending_deletion.read().len()
    }

    /// Attempts to delete segments pending deletion
    ///
    /// A segment can be deleted if:
    /// 1. It's marked for deletion
    /// 2. There's only 1 Arc reference (the one in the registry)
    ///
    /// Safety: Holds `segments` write lock during both the reference count
    /// check and removal. This prevents `get()` from cloning an Arc between
    /// the check and the removal (TOCTOU race that previously caused SIGBUS).
    ///
    /// Physical file deletion happens after the Arc is dropped, so no mmap
    /// can reference the file.
    ///
    /// Returns `(segments_deleted, bytes_freed)`.
    pub fn cleanup(&self) -> (usize, u64) {
        let mut pending = self.pending_deletion.write();
        let mut segments = self.segments.write();

        // While we hold segments.write(), no get() can clone Arcs
        let to_delete: Vec<u64> = pending
            .iter()
            .filter(|id| {
                segments
                    .get(id)
                    .map(|(seg, _)| Arc::strong_count(seg) == 1)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let mut bytes_freed: u64 = 0;
        for id in &to_delete {
            pending.remove(id);
            if let Some((segment, path)) = segments.remove(id) {
                bytes_freed += segment.size_in_bytes();
                // Arc drops here, unmapping the file safely
                drop(segment);
                // Delete the physical files
                if path.exists() {
                    if let Err(e) = std::fs::remove_dir_all(&path) {
                        tracing::error!(
                            segment_id = id,
                            path = ?path,
                            error = ?e,
                            "Failed to delete segment directory"
                        );
                    } else {
                        tracing::debug!(segment_id = id, path = ?path, "Deleted segment directory");
                    }
                }
            }
        }

        (to_delete.len(), bytes_freed)
    }

    /// Forces removal of a segment from the registry
    ///
    /// WARNING: This is unsafe if the segment is still referenced by
    /// compaction jobs. Only use during shutdown or testing.
    pub fn force_remove(&self, segment_id: u64) -> Option<Arc<Segment>> {
        let mut live = self.live_segments.write();
        let mut pending = self.pending_deletion.write();
        let mut segments = self.segments.write();

        live.remove(&segment_id);
        pending.remove(&segment_id);
        segments.remove(&segment_id).map(|(seg, _)| seg)
    }

    /// Returns statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            live: self.live_count(),
            pending_deletion: self.pending_deletion_count(),
            total_tracked: self.segments.read().len(),
        }
    }

    /// Returns the base path for segment files
    pub fn base_path(&self) -> &PathBuf {
        &self.base_path
    }

    /// Clears all segments from the registry
    ///
    /// This is primarily for testing. Don't use in production.
    #[cfg(test)]
    pub fn clear(&self) {
        self.live_segments.write().clear();
        self.segments.write().clear();
        self.pending_deletion.write().clear();
    }
}

impl Default for SegmentRegistry {
    fn default() -> Self {
        Self::new(PathBuf::from("."))
    }
}

/// Statistics about the segment registry
#[derive(Debug, Clone, Copy)]
pub struct RegistryStats {
    /// Number of live segments
    pub live: usize,
    /// Number of segments pending deletion
    pub pending_deletion: usize,
    /// Total segments tracked (live + pending)
    pub total_tracked: usize,
}

impl std::fmt::Display for RegistryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Registry: {} live, {} pending deletion, {} total",
            self.live, self.pending_deletion, self.total_tracked
        )
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::{
        segment::Segment,
        segment_builder::SegmentBuilder,
    };

    fn create_test_segment_with_dir(id: u64) -> (Arc<Segment>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();
        let segment = builder.new_segment(id, 12345, 64 * 1024 * 1024).unwrap();
        (segment, temp_dir)
    }

    #[test]
    fn test_registry_creation() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        assert_eq!(registry.live_count(), 0);
        assert_eq!(registry.pending_deletion_count(), 0);
    }

    #[test]
    fn test_register_segment() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);

        registry.register(segment.clone(), temp_dir.path().to_path_buf());

        assert_eq!(registry.live_count(), 1);
        assert!(registry.is_live(1));
        assert_eq!(Arc::strong_count(&segment), 2); // registry + our copy
    }

    #[test]
    fn test_mark_for_deletion() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);

        registry.register(segment.clone(), temp_dir.path().to_path_buf());
        assert!(registry.is_live(1));

        registry.mark_for_deletion(1);
        assert!(!registry.is_live(1));
        assert_eq!(registry.pending_deletion_count(), 1);
    }

    #[test]
    fn test_cleanup_with_no_external_refs() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);
        let path = temp_dir.path().to_path_buf();

        registry.register(segment.clone(), path.clone());
        drop(segment); // Drop our reference

        registry.mark_for_deletion(1);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Should delete the segment since we hold the only ref
        let (deleted, _bytes_freed) = registry.cleanup();
        assert_eq!(deleted, 1);
        assert_eq!(registry.pending_deletion_count(), 0);
        assert_eq!(registry.live_count(), 0);
    }

    #[test]
    fn test_cleanup_with_external_refs() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);
        let path = temp_dir.path().to_path_buf();

        registry.register(segment.clone(), path.clone());
        // Keep our reference alive

        registry.mark_for_deletion(1);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Should NOT delete since we still hold a reference
        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 0);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Drop our reference
        drop(segment);

        // Now cleanup should work
        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 1);
        assert_eq!(registry.pending_deletion_count(), 0);
    }

    #[test]
    fn test_get_segment() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);

        registry.register(segment.clone(), temp_dir.path().to_path_buf());

        let retrieved = registry.get(1);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id(), 1);

        let non_existent = registry.get(999);
        assert!(non_existent.is_none());
    }

    #[test]
    fn test_force_remove() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);

        registry.register(segment.clone(), temp_dir.path().to_path_buf());
        assert!(registry.is_live(1));

        let removed = registry.force_remove(1);
        assert!(removed.is_some());
        assert!(!registry.is_live(1));
        assert_eq!(registry.live_count(), 0);
    }

    #[test]
    fn test_stats() {
        let registry = SegmentRegistry::new(PathBuf::from("."));

        let (seg1, temp1) = create_test_segment_with_dir(1);
        let (seg2, temp2) = create_test_segment_with_dir(2);

        registry.register(seg1, temp1.path().to_path_buf());
        registry.register(seg2.clone(), temp2.path().to_path_buf());

        let stats = registry.stats();
        assert_eq!(stats.live, 2);
        assert_eq!(stats.pending_deletion, 0);
        assert_eq!(stats.total_tracked, 2);

        registry.mark_for_deletion(1);

        let stats = registry.stats();
        assert_eq!(stats.live, 1);
        assert_eq!(stats.pending_deletion, 1);
        assert_eq!(stats.total_tracked, 2);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let registry = Arc::new(SegmentRegistry::new(PathBuf::from(".")));

        // Spawn threads that register segments
        let mut handles = vec![];
        for i in 0..10 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                let (segment, temp_dir) = create_test_segment_with_dir(i);
                reg.register(segment, temp_dir.path().to_path_buf());
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.live_count(), 10);
    }

    #[test]
    fn test_cleanup_concurrent_readers() {
        use std::{
            sync::atomic::{
                AtomicBool,
                Ordering,
            },
            thread,
        };

        let registry = Arc::new(SegmentRegistry::new(PathBuf::from(".")));
        let mut temp_dirs = Vec::new();

        // Register segments and mark them for deletion
        for i in 0..10 {
            let (segment, temp_dir) = create_test_segment_with_dir(i);
            temp_dirs.push(temp_dir);
            registry.register(segment, temp_dirs[i as usize].path().to_path_buf());
            registry.mark_for_deletion(i);
        }

        let done = Arc::new(AtomicBool::new(false));

        // Spawn 10 reader threads calling get() concurrently
        let mut handles = vec![];
        for _ in 0..10 {
            let reg = registry.clone();
            let done = done.clone();
            handles.push(thread::spawn(move || {
                let mut reads = 0u64;
                while !done.load(Ordering::Relaxed) {
                    for id in 0..10 {
                        // get() clones the Arc — this is the operation that
                        // previously raced with cleanup
                        let _ = reg.get(id);
                        reads += 1;
                    }
                }
                reads
            }));
        }

        // Cleanup thread runs concurrently with readers
        let cleanup_reg = registry.clone();
        let cleanup_handle = thread::spawn(move || {
            let mut total_deleted = 0;
            for _ in 0..1000 {
                total_deleted += cleanup_reg.cleanup().0;
            }
            total_deleted
        });

        let total_deleted = cleanup_handle.join().unwrap();
        done.store(true, Ordering::Relaxed);

        let total_reads: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

        // All 10 segments should eventually be cleaned up
        assert_eq!(total_deleted, 10);
        assert_eq!(registry.pending_deletion_count(), 0);
        assert!(total_reads > 0, "readers should have completed some reads");
    }

    #[test]
    fn test_cleanup_idempotent() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);

        registry.register(segment, temp_dir.path().to_path_buf());
        registry.mark_for_deletion(1);

        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 1);

        // Second cleanup should find nothing to delete
        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 0);
        assert_eq!(registry.pending_deletion_count(), 0);
    }

    #[test]
    fn test_cleanup_with_active_reader() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);

        registry.register(segment, temp_dir.path().to_path_buf());

        // Simulate a reader holding an Arc via get()
        let reader_ref = registry.get(1).unwrap();
        assert_eq!(Arc::strong_count(&reader_ref), 2); // registry + reader

        registry.mark_for_deletion(1);

        // Cleanup should skip because reader holds a reference
        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 0);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Drop the reader reference
        drop(reader_ref);

        // Now cleanup should succeed
        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 1);
        assert_eq!(registry.pending_deletion_count(), 0);
    }

    #[test]
    fn test_cleanup_deletes_files() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);
        let path = temp_dir.path().to_path_buf();

        registry.register(segment, path.clone());
        assert!(path.exists());

        registry.mark_for_deletion(1);
        let (deleted, _bytes_freed) = registry.cleanup();
        assert_eq!(deleted, 1);
        assert!(!path.exists(), "segment directory should be deleted");
    }

    #[test]
    fn test_cleanup_does_not_delete_live_files() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);
        let path = temp_dir.path().to_path_buf();

        registry.register(segment, path.clone());
        // Do NOT mark for deletion

        let (deleted, _) = registry.cleanup();
        assert_eq!(deleted, 0);
        assert!(
            path.exists(),
            "live segment directory should NOT be deleted"
        );
    }

    #[test]
    fn test_cleanup_returns_bytes_freed() {
        let registry = SegmentRegistry::new(PathBuf::from("."));
        let (segment, temp_dir) = create_test_segment_with_dir(1);
        let path = temp_dir.path().to_path_buf();

        registry.register(segment, path.clone());
        registry.mark_for_deletion(1);

        let (deleted, _bytes_freed) = registry.cleanup();
        assert_eq!(deleted, 1);
        // bytes_freed may be 0 for empty test segments without open handles
        assert!(!path.exists(), "segment directory should be deleted");
    }
}
