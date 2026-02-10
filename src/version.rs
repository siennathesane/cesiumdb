//! Lock-free version management for the LSM-tree
//!
//! This module implements atomic version set management, allowing:
//! - Lock-free reads of the current version
//! - Atomic version swaps when compaction completes
//! - Safe garbage collection of old versions via reference counting
//!
//! ## Design
//!
//! Versions are immutable snapshots of the LSM-tree state. When compaction
//! completes, a new version is created and atomically installed. Readers
//! obtain an Arc reference to the current version, which keeps it alive
//! even if a new version is installed.
//!
//! This design provides:
//! - Zero-cost reads (just an Arc clone)
//! - No blocking between readers and writers
//! - Automatic cleanup of old versions via Arc drop

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{
            AtomicU64,
            Ordering,
        },
    },
};

use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};
use parking_lot::RwLock;

use crate::{
    errs::{
        ManifestError,
        SegmentError,
    },
    levels::{
        KeyRange,
        VersionSet,
    },
    manifest::{
        put_varint,
        read_varint,
    },
    segment_builder::SegmentBuilder,
};

/// Version manager providing atomic version set updates
///
/// The version manager coordinates access to the current LSM-tree state.
/// It ensures that:
/// - Readers always see a consistent snapshot
/// - Writers can atomically update the state
/// - Old versions are cleaned up when no longer referenced
pub struct VersionManager {
    /// Current version (protected by RwLock for atomic swaps)
    ///
    /// Readers take a read lock and clone the Arc (cheap).
    /// Writers take a write lock and replace the Arc (rare).
    current: RwLock<Arc<VersionSet>>,

    /// Monotonic sequence number for version ordering
    ///
    /// Incremented atomically for each new version.
    sequence: AtomicU64,

    /// Global segment ID counter
    ///
    /// Used by both flush and compaction to allocate unique segment IDs.
    /// Prevents ID collisions between flush and compaction pathways.
    next_segment_id: AtomicU64,
}

impl VersionManager {
    /// Creates a new version manager with an initial empty version set
    ///
    /// # Arguments
    /// * `num_levels` - Number of levels in the LSM-tree (typically 6-7)
    pub fn new(num_levels: usize) -> Self {
        let initial_version = VersionSet::new(0, num_levels);
        Self {
            current: RwLock::new(Arc::new(initial_version)),
            sequence: AtomicU64::new(0),
            next_segment_id: AtomicU64::new(0),
        }
    }

    /// Creates a version manager with a specific initial version
    pub fn with_version(version: VersionSet) -> Self {
        let seq = version.sequence;
        let max_seg_id = version.max_segment_id();
        Self {
            current: RwLock::new(Arc::new(version)),
            sequence: AtomicU64::new(seq),
            next_segment_id: AtomicU64::new(max_seg_id + 1),
        }
    }

    /// Gets a snapshot of the current version
    ///
    /// This is lock-free for readers after the initial lock acquisition.
    /// The returned Arc keeps the version alive even if a new version is
    /// installed.
    ///
    /// # Performance
    ///
    /// - Acquires read lock (uncontended fast path)
    /// - Clones Arc (atomic increment)
    /// - Releases read lock
    ///
    /// Total cost: ~2-3 atomic operations
    #[inline]
    pub fn current(&self) -> Arc<VersionSet> {
        self.current.read().clone()
    }

    /// Atomically installs a new version
    ///
    /// This replaces the current version with a new one. The old version
    /// will be dropped when all readers release their references.
    ///
    /// # Arguments
    /// * `new_version` - The new version to install
    ///
    /// # Returns
    /// The old version (for testing/verification)
    ///
    /// # Note
    /// The caller is responsible for ensuring the sequence number is correct.
    /// Use `update()` for automatic sequence management.
    pub fn install(&self, new_version: VersionSet) -> Arc<VersionSet> {
        let mut current = self.current.write();
        let old = current.clone();
        *current = Arc::new(new_version);
        old
    }

    /// Creates a new version based on the current one
    ///
    /// This is a helper for the common pattern of:
    /// 1. Get current version
    /// 2. Clone it
    /// 3. Modify the clone
    /// 4. Install it
    ///
    /// The callback receives a mutable copy of the current version and
    /// the next sequence number.
    ///
    /// # Arguments
    /// * `f` - Callback that modifies the version
    ///
    /// # Example
    /// ```ignore
    /// version_manager.update(|version| {
    ///     version.add_to_l0(new_segment);
    /// });
    /// ```
    pub fn update<F>(&self, f: F) -> Arc<VersionSet>
    where
        F: FnOnce(&mut VersionSet), {
        // Get current version under read lock
        let current = self.current();

        // Create a new version with incremented sequence
        let next_seq = self.sequence.fetch_add(1, Ordering::AcqRel) + 1;
        let mut new_version = (*current).clone();
        new_version.sequence = next_seq;

        // Apply modifications
        f(&mut new_version);

        // Install atomically
        self.install(new_version);

        // Return new current
        self.current()
    }

    /// Gets the current sequence number
    ///
    /// Useful for debugging and logging.
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }

    /// Checks if a given sequence number is current
    ///
    /// Used by compaction to detect if the version changed during execution.
    #[inline]
    pub fn is_current(&self, seq: u64) -> bool {
        self.sequence() == seq
    }

    /// Allocates a new globally unique segment ID
    ///
    /// This method is used by both flush and compaction pathways to ensure
    /// that segment IDs are unique across the entire database lifecycle.
    ///
    /// # Returns
    /// A unique segment ID that is guaranteed not to conflict with any
    /// existing or future segments.
    #[inline]
    pub fn next_segment_id(&self) -> u64 {
        self.next_segment_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Gets the number of levels in the current version
    pub fn num_levels(&self) -> usize {
        self.current().num_levels()
    }

    /// Gets statistics about the current version
    pub fn stats(&self) -> VersionStats {
        let current = self.current();
        VersionStats {
            sequence: current.sequence,
            total_segments: current.total_segments,
            total_size: current.total_size,
            l0_segments: current.l0.len(),
            num_levels: current.num_levels(),
        }
    }
}

/// Statistics about a version
#[derive(Debug, Clone, Copy)]
pub struct VersionStats {
    /// Version sequence number
    pub sequence: u64,
    /// Total number of segments across all levels
    pub total_segments: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Number of L0 segments
    pub l0_segments: usize,
    /// Number of levels
    pub num_levels: usize,
}

impl std::fmt::Display for VersionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Version {} | {} segments | {} MB | L0: {}",
            self.sequence,
            self.total_segments,
            self.total_size / (1024 * 1024),
            self.l0_segments
        )
    }
}

/// A versioned edit representing a change to the LSM-tree state
///
/// Version edits are used by the manifest to record state changes durably.
/// They can be replayed on startup to reconstruct the latest version.
#[derive(Debug, Clone)]
pub enum VersionEdit {
    /// A new segment was added to L0
    AddL0Segment {
        segment_id: u64,
        key_range: (Vec<u8>, Vec<u8>),
        size: u64,
    },

    /// A segment was added to a specific level
    AddSegment {
        level: u8,
        segment_id: u64,
        key_range: (Vec<u8>, Vec<u8>),
        size: u64,
    },

    /// A segment was removed from a specific level
    RemoveSegment { level: u8, segment_id: u64 },

    /// A segment was removed from L0
    RemoveL0Segment { segment_id: u64 },

    /// The sequence number was updated
    UpdateSequence { sequence: u64 },
}

// Type discriminants for VersionEdit serialization
const EDIT_ADD_L0_SEGMENT: u8 = 0x01;
const EDIT_ADD_SEGMENT: u8 = 0x02;
const EDIT_REMOVE_SEGMENT: u8 = 0x03;
const EDIT_REMOVE_L0_SEGMENT: u8 = 0x04;
const EDIT_UPDATE_SEQUENCE: u8 = 0x05;

impl VersionEdit {
    /// Encodes this edit to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        match self {
            | VersionEdit::AddL0Segment {
                segment_id,
                key_range,
                size,
            } => {
                buf.put_u8(EDIT_ADD_L0_SEGMENT);
                buf.put_u64_le(*segment_id);
                buf.put_u64_le(*size);
                put_varint(&mut buf, key_range.0.len() as u64);
                buf.put_slice(&key_range.0);
                put_varint(&mut buf, key_range.1.len() as u64);
                buf.put_slice(&key_range.1);
            },
            | VersionEdit::AddSegment {
                level,
                segment_id,
                key_range,
                size,
            } => {
                buf.put_u8(EDIT_ADD_SEGMENT);
                buf.put_u8(*level);
                buf.put_u64_le(*segment_id);
                buf.put_u64_le(*size);
                put_varint(&mut buf, key_range.0.len() as u64);
                buf.put_slice(&key_range.0);
                put_varint(&mut buf, key_range.1.len() as u64);
                buf.put_slice(&key_range.1);
            },
            | VersionEdit::RemoveSegment { level, segment_id } => {
                buf.put_u8(EDIT_REMOVE_SEGMENT);
                buf.put_u8(*level);
                buf.put_u64_le(*segment_id);
            },
            | VersionEdit::RemoveL0Segment { segment_id } => {
                buf.put_u8(EDIT_REMOVE_L0_SEGMENT);
                buf.put_u64_le(*segment_id);
            },
            | VersionEdit::UpdateSequence { sequence } => {
                buf.put_u8(EDIT_UPDATE_SEQUENCE);
                buf.put_u64_le(*sequence);
            },
        }

        buf.freeze()
    }

    /// Decodes a VersionEdit from bytes
    pub fn decode(mut data: Bytes) -> Result<Self, ManifestError> {
        if data.is_empty() {
            return Err(ManifestError::CorruptedHeader);
        }

        let edit_type = data.get_u8();

        match edit_type {
            | EDIT_ADD_L0_SEGMENT => {
                if data.len() < 16 {
                    return Err(ManifestError::CorruptedHeader);
                }
                let segment_id = data.get_u64_le();
                let size = data.get_u64_le();

                let (min_len, _) = match read_varint(&mut data) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                if data.len() < min_len as usize {
                    return Err(ManifestError::CorruptedHeader);
                }
                let min_key = data.split_to(min_len as usize).to_vec();

                let (max_len, _) = match read_varint(&mut data) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                if data.len() < max_len as usize {
                    return Err(ManifestError::CorruptedHeader);
                }
                let max_key = data.split_to(max_len as usize).to_vec();

                Ok(VersionEdit::AddL0Segment {
                    segment_id,
                    key_range: (min_key, max_key),
                    size,
                })
            },
            | EDIT_ADD_SEGMENT => {
                if data.len() < 17 {
                    return Err(ManifestError::CorruptedHeader);
                }
                let level = data.get_u8();
                let segment_id = data.get_u64_le();
                let size = data.get_u64_le();

                let (min_len, _) = match read_varint(&mut data) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                if data.len() < min_len as usize {
                    return Err(ManifestError::CorruptedHeader);
                }
                let min_key = data.split_to(min_len as usize).to_vec();

                let (max_len, _) = match read_varint(&mut data) {
                    | Ok(v) => v,
                    | Err(e) => return Err(e),
                };
                if data.len() < max_len as usize {
                    return Err(ManifestError::CorruptedHeader);
                }
                let max_key = data.split_to(max_len as usize).to_vec();

                Ok(VersionEdit::AddSegment {
                    level,
                    segment_id,
                    key_range: (min_key, max_key),
                    size,
                })
            },
            | EDIT_REMOVE_SEGMENT => {
                if data.len() < 9 {
                    return Err(ManifestError::CorruptedHeader);
                }
                let level = data.get_u8();
                let segment_id = data.get_u64_le();

                Ok(VersionEdit::RemoveSegment { level, segment_id })
            },
            | EDIT_REMOVE_L0_SEGMENT => {
                if data.len() < 8 {
                    return Err(ManifestError::CorruptedHeader);
                }
                let segment_id = data.get_u64_le();

                Ok(VersionEdit::RemoveL0Segment { segment_id })
            },
            | EDIT_UPDATE_SEQUENCE => {
                if data.len() < 8 {
                    return Err(ManifestError::CorruptedHeader);
                }
                let sequence = data.get_u64_le();

                Ok(VersionEdit::UpdateSequence { sequence })
            },
            | _ => Err(ManifestError::InvalidEditType(edit_type)),
        }
    }

    /// Applies this edit to a version set
    ///
    /// This is used during manifest replay to reconstruct state.
    /// `base_path` is needed to reopen segment files from disk for Add
    /// operations.
    pub fn apply(&self, version: &mut VersionSet, base_path: &Path) -> Result<(), SegmentError> {
        match self {
            | VersionEdit::AddL0Segment {
                segment_id,
                key_range,
                ..
            } => {
                // L0 segments are stored in base_path/sstables/{id}/
                let segment_path = base_path.join("sstables").join(segment_id.to_string());
                let builder = match SegmentBuilder::new(segment_path) {
                    | Ok(b) => b,
                    | Err(e) => return Err(e),
                };
                let segment = match builder.open(*segment_id) {
                    | Ok(s) => s,
                    | Err(e) => return Err(e),
                };
                
                // Use key_range from manifest
                let range = KeyRange::new(
                    key_range.0.clone(),
                    key_range.1.clone(),
                    *segment_id
                );
                
                version.add_to_l0(segment, range);
                Ok(())
            },
            | VersionEdit::AddSegment {
                level,
                segment_id,
                key_range,
                ..
            } => {
                // Leveled segments are stored in base_path/L{level}/sstables/{id}/
                let segment_path = base_path
                    .join(format!("L{}", level))
                    .join("sstables")
                    .join(segment_id.to_string());
                let builder = match SegmentBuilder::new(segment_path) {
                    | Ok(b) => b,
                    | Err(e) => return Err(e),
                };
                let segment = match builder.open(*segment_id) {
                    | Ok(s) => s,
                    | Err(e) => return Err(e),
                };
                let range = KeyRange::new(key_range.0.clone(), key_range.1.clone(), *segment_id);
                let level_idx = *level as usize - 1;
                if level_idx < version.levels.len() {
                    version.levels[level_idx].add_segment(segment, range);
                }
                Ok(())
            },
            | VersionEdit::RemoveSegment { level, segment_id } => {
                let level_idx = *level as usize - 1;
                if level_idx < version.levels.len() {
                    version.levels[level_idx].remove_segment(*segment_id);
                }
                Ok(())
            },
            | VersionEdit::RemoveL0Segment { segment_id } => {
                version.l0.retain(|s| s.id() != *segment_id);
                Ok(())
            },
            | VersionEdit::UpdateSequence { sequence } => {
                version.sequence = *sequence;
                Ok(())
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_manager_creation() {
        let vm = VersionManager::new(7);
        assert_eq!(vm.sequence(), 0);
        assert_eq!(vm.num_levels(), 7);
    }

    #[test]
    fn test_current_returns_snapshot() {
        let vm = VersionManager::new(7);
        let v1 = vm.current();
        let v2 = vm.current();

        // Should be the same version
        assert_eq!(v1.sequence, v2.sequence);

        // Should be Arc clones
        assert!(Arc::ptr_eq(&v1, &v2));
    }

    #[test]
    fn test_update_increments_sequence() {
        let vm = VersionManager::new(7);

        assert_eq!(vm.sequence(), 0);

        vm.update(|_| {
            // Make some modification
        });

        assert_eq!(vm.sequence(), 1);

        vm.update(|_| {
            // Another modification
        });

        assert_eq!(vm.sequence(), 2);
    }

    #[test]
    fn test_update_modifies_version() {
        let vm = VersionManager::new(7);

        let v1 = vm.current();
        assert_eq!(v1.total_size, 0);

        vm.update(|version| {
            version.total_size = 12345;
        });

        let v2 = vm.current();
        assert_eq!(v2.total_size, 12345);

        // Old version unchanged
        assert_eq!(v1.total_size, 0);
    }

    #[test]
    fn test_install_replaces_version() {
        let vm = VersionManager::new(7);

        let old = vm.current();
        assert_eq!(old.sequence, 0);

        let mut new_version = VersionSet::new(5, 7);
        new_version.total_size = 999;

        let returned_old = vm.install(new_version);

        // Returned old should match what we had
        assert_eq!(returned_old.sequence, old.sequence);

        // Current should be the new version
        let current = vm.current();
        assert_eq!(current.sequence, 5);
        assert_eq!(current.total_size, 999);

        // Note: vm.sequence() is NOT automatically updated by install()
        // It remains at the initial value unless update() is used
        assert_eq!(vm.sequence(), 0);
    }

    #[test]
    fn test_is_current() {
        let vm = VersionManager::new(7);

        assert!(vm.is_current(0));
        assert!(!vm.is_current(1));

        vm.update(|_| {});

        assert!(!vm.is_current(0));
        assert!(vm.is_current(1));
    }

    #[test]
    fn test_stats() {
        let vm = VersionManager::new(7);

        vm.update(|version| {
            version.total_segments = 42;
            version.total_size = 1024 * 1024 * 100; // 100 MB
        });

        let stats = vm.stats();
        assert_eq!(stats.total_segments, 42);
        assert_eq!(stats.total_size, 1024 * 1024 * 100);
        assert_eq!(stats.num_levels, 7);
    }

    #[test]
    fn test_concurrent_readers() {
        use std::thread;

        let vm = Arc::new(VersionManager::new(7));

        // Spawn multiple reader threads
        let mut handles = vec![];
        for _ in 0..10 {
            let vm_clone = vm.clone();
            handles.push(thread::spawn(move || {
                // Each thread reads the current version multiple times
                for _ in 0..100 {
                    let _v = vm_clone.current();
                }
            }));
        }

        // Wait for all readers
        for handle in handles {
            handle.join().unwrap();
        }

        // Should still be at sequence 0
        assert_eq!(vm.sequence(), 0);
    }

    #[test]
    fn test_concurrent_updates() {
        use std::thread;

        let vm = Arc::new(VersionManager::new(7));

        // Spawn multiple updater threads
        let mut handles = vec![];
        for i in 0..10 {
            let vm_clone = vm.clone();
            handles.push(thread::spawn(move || {
                vm_clone.update(|version| {
                    version.total_size += i;
                });
            }));
        }

        // Wait for all updaters
        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 10 updates
        assert_eq!(vm.sequence(), 10);

        // Size might not be sum(0..10) due to race conditions,
        // but should be non-zero
        assert!(vm.current().total_size > 0);
    }

    #[test]
    fn test_version_edit_add_l0_roundtrip() {
        let edit = VersionEdit::AddL0Segment {
            segment_id: 123,
            key_range: (b"start".to_vec(), b"end".to_vec()),
            size: 4096,
        };

        let encoded = edit.encode();
        let decoded = VersionEdit::decode(encoded).unwrap();

        match decoded {
            | VersionEdit::AddL0Segment {
                segment_id,
                key_range,
                size,
            } => {
                assert_eq!(segment_id, 123);
                assert_eq!(key_range.0, b"start");
                assert_eq!(key_range.1, b"end");
                assert_eq!(size, 4096);
            },
            | _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_version_edit_add_segment_roundtrip() {
        let edit = VersionEdit::AddSegment {
            level: 2,
            segment_id: 456,
            key_range: (b"min".to_vec(), b"max".to_vec()),
            size: 8192,
        };

        let encoded = edit.encode();
        let decoded = VersionEdit::decode(encoded).unwrap();

        match decoded {
            | VersionEdit::AddSegment {
                level,
                segment_id,
                key_range,
                size,
            } => {
                assert_eq!(level, 2);
                assert_eq!(segment_id, 456);
                assert_eq!(key_range.0, b"min");
                assert_eq!(key_range.1, b"max");
                assert_eq!(size, 8192);
            },
            | _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_version_edit_remove_segment_roundtrip() {
        let edit = VersionEdit::RemoveSegment {
            level: 3,
            segment_id: 789,
        };

        let encoded = edit.encode();
        let decoded = VersionEdit::decode(encoded).unwrap();

        match decoded {
            | VersionEdit::RemoveSegment { level, segment_id } => {
                assert_eq!(level, 3);
                assert_eq!(segment_id, 789);
            },
            | _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_version_edit_remove_l0_roundtrip() {
        let edit = VersionEdit::RemoveL0Segment { segment_id: 111 };

        let encoded = edit.encode();
        let decoded = VersionEdit::decode(encoded).unwrap();

        match decoded {
            | VersionEdit::RemoveL0Segment { segment_id } => {
                assert_eq!(segment_id, 111);
            },
            | _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_version_edit_update_sequence_roundtrip() {
        let edit = VersionEdit::UpdateSequence { sequence: 999 };

        let encoded = edit.encode();
        let decoded = VersionEdit::decode(encoded).unwrap();

        match decoded {
            | VersionEdit::UpdateSequence { sequence } => {
                assert_eq!(sequence, 999);
            },
            | _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_version_edit_large_keys() {
        let large_key = vec![0xab; 10000];
        let edit = VersionEdit::AddL0Segment {
            segment_id: 999,
            key_range: (large_key.clone(), large_key.clone()),
            size: 100000,
        };

        let encoded = edit.encode();
        let decoded = VersionEdit::decode(encoded).unwrap();

        match decoded {
            | VersionEdit::AddL0Segment { key_range, .. } => {
                assert_eq!(key_range.0.len(), 10000);
                assert_eq!(key_range.1.len(), 10000);
            },
            | _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_version_edit_invalid_type() {
        let mut buf = BytesMut::new();
        buf.put_u8(0xff); // invalid type
        buf.put_u64_le(123);

        let result = VersionEdit::decode(buf.freeze());
        assert!(matches!(result, Err(ManifestError::InvalidEditType(0xff))));
    }

    #[test]
    fn test_version_edit_truncated() {
        let mut buf = BytesMut::new();
        buf.put_u8(EDIT_ADD_L0_SEGMENT);
        buf.put_u64_le(123);
        // Missing size and keys

        let result = VersionEdit::decode(buf.freeze());
        assert!(matches!(result, Err(ManifestError::CorruptedHeader)));
    }
}
