//! Segment registry for reference tracking
//!
//! This module provides a registry to track which segments are live
//! and coordinate safe deletion after compaction.

use crate::segment::Segment;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Registry for tracking live segments
///
/// The registry maintains:
/// - Which segments are currently live (in the current version)
/// - Reference counts for segments in use by compaction jobs
/// - A delete queue for segments that should be removed
pub struct SegmentRegistry {
    /// Currently live segment IDs (in the current version)
    live_segments: RwLock<HashSet<u64>>,

    /// Segment ID -> Arc<Segment> mapping
    ///
    /// Keeps segments alive while they're registered.
    /// When a segment is removed from here and no compaction jobs
    /// reference it, it can be safely deleted.
    segments: RwLock<HashMap<u64, Arc<Segment>>>,

    /// Segments pending deletion
    ///
    /// These segments have been removed from the version but may still
    /// be referenced by in-progress compactions.
    pending_deletion: RwLock<HashSet<u64>>,
}

impl SegmentRegistry {
    /// Creates a new empty registry
    pub fn new() -> Self {
        Self {
            live_segments: RwLock::new(HashSet::new()),
            segments: RwLock::new(HashMap::new()),
            pending_deletion: RwLock::new(HashSet::new()),
        }
    }

    /// Registers a segment as live
    ///
    /// This should be called when a segment is added to the version.
    pub fn register(&self, segment: Arc<Segment>) {
        let id = segment.id();

        let mut live = self.live_segments.write();
        let mut segments = self.segments.write();

        live.insert(id);
        segments.insert(id, segment);
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
        self.segments.read().get(&segment_id).cloned()
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
    /// Returns the number of segments actually deleted.
    pub fn cleanup(&self) -> usize {
        let mut pending = self.pending_deletion.write();
        let mut segments = self.segments.write();

        let mut deleted = Vec::new();

        for &segment_id in pending.iter() {
            if let Some(segment) = segments.get(&segment_id) {
                // Check if we hold the only reference
                if Arc::strong_count(segment) == 1 {
                    deleted.push(segment_id);
                }
            }
        }

        // Remove deleted segments
        for id in &deleted {
            pending.remove(id);
            segments.remove(id);
        }

        deleted.len()
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
        segments.remove(&segment_id)
    }

    /// Returns statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            live: self.live_count(),
            pending_deletion: self.pending_deletion_count(),
            total_tracked: self.segments.read().len(),
        }
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
        Self::new()
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
    use super::*;
    use crate::segment::Segment;
    use crate::segment_builder::SegmentBuilder;
    use tempfile::TempDir;

    fn create_test_segment(id: u64) -> Arc<Segment> {
        let temp_dir = TempDir::new().unwrap();
        let builder = SegmentBuilder::new(temp_dir.path().to_path_buf()).unwrap();
        builder
            .new_segment(id, 12345, 64 * 1024 * 1024)
            .unwrap()
    }

    #[test]
    fn test_registry_creation() {
        let registry = SegmentRegistry::new();
        assert_eq!(registry.live_count(), 0);
        assert_eq!(registry.pending_deletion_count(), 0);
    }

    #[test]
    fn test_register_segment() {
        let registry = SegmentRegistry::new();
        let segment = create_test_segment(1);

        registry.register(segment.clone());

        assert_eq!(registry.live_count(), 1);
        assert!(registry.is_live(1));
        assert_eq!(Arc::strong_count(&segment), 2); // registry + our copy
    }

    #[test]
    fn test_mark_for_deletion() {
        let registry = SegmentRegistry::new();
        let segment = create_test_segment(1);

        registry.register(segment.clone());
        assert!(registry.is_live(1));

        registry.mark_for_deletion(1);
        assert!(!registry.is_live(1));
        assert_eq!(registry.pending_deletion_count(), 1);
    }

    #[test]
    fn test_cleanup_with_no_external_refs() {
        let registry = SegmentRegistry::new();
        let segment = create_test_segment(1);

        registry.register(segment.clone());
        drop(segment); // Drop our reference

        registry.mark_for_deletion(1);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Should delete the segment since we hold the only ref
        let deleted = registry.cleanup();
        assert_eq!(deleted, 1);
        assert_eq!(registry.pending_deletion_count(), 0);
        assert_eq!(registry.live_count(), 0);
    }

    #[test]
    fn test_cleanup_with_external_refs() {
        let registry = SegmentRegistry::new();
        let segment = create_test_segment(1);

        registry.register(segment.clone());
        // Keep our reference alive

        registry.mark_for_deletion(1);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Should NOT delete since we still hold a reference
        let deleted = registry.cleanup();
        assert_eq!(deleted, 0);
        assert_eq!(registry.pending_deletion_count(), 1);

        // Drop our reference
        drop(segment);

        // Now cleanup should work
        let deleted = registry.cleanup();
        assert_eq!(deleted, 1);
        assert_eq!(registry.pending_deletion_count(), 0);
    }

    #[test]
    fn test_get_segment() {
        let registry = SegmentRegistry::new();
        let segment = create_test_segment(1);

        registry.register(segment.clone());

        let retrieved = registry.get(1);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id(), 1);

        let non_existent = registry.get(999);
        assert!(non_existent.is_none());
    }

    #[test]
    fn test_force_remove() {
        let registry = SegmentRegistry::new();
        let segment = create_test_segment(1);

        registry.register(segment.clone());
        assert!(registry.is_live(1));

        let removed = registry.force_remove(1);
        assert!(removed.is_some());
        assert!(!registry.is_live(1));
        assert_eq!(registry.live_count(), 0);
    }

    #[test]
    fn test_stats() {
        let registry = SegmentRegistry::new();

        let seg1 = create_test_segment(1);
        let seg2 = create_test_segment(2);

        registry.register(seg1);
        registry.register(seg2.clone());

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

        let registry = Arc::new(SegmentRegistry::new());

        // Spawn threads that register segments
        let mut handles = vec![];
        for i in 0..10 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                let segment = create_test_segment(i);
                reg.register(segment);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(registry.live_count(), 10);
    }
}
