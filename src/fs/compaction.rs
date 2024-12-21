use std::sync::{
    atomic::Ordering::SeqCst,
    Arc,
};

use bytes::BytesMut;

use crate::{
    errs::{
        FsError,
        FsError::{
            FRangeNotFound,
            IoError,
        },
    },
    fs::{
        core::Fs,
        handle::FRangeMetadata,
    },
};

/// Configuration for fragmentation detection and compaction
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Threshold ratio of fragmented space to trigger compaction (0.0 to 1.0)
    pub fragmentation_threshold: f64,
    /// Minimum size in bytes for a fragment to be considered for compaction
    pub min_fragment_size: u64,
    /// Maximum number of franges to compact in one operation
    pub max_compact_batch: usize,

    pub block_buffer_size: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            fragmentation_threshold: 0.3, // 30% fragmentation triggers compaction
            min_fragment_size: 4096,      // Ignore fragments smaller than 4KB
            max_compact_batch: 10,        // Compact up to 10 franges at once
            block_buffer_size: 4096,      // 4KiB
        }
    }
}

#[derive(Debug, Default)]
pub struct FragmentationStats {
    pub total_free_space: u64,
    pub free_range_count: u64,
    pub largest_free_block: u64,
    pub average_fragment_size: u64,
    pub small_fragments: u64, // Fragments smaller than page size
}

impl Fs {
    /// Check if compaction is needed and perform it if necessary
    pub fn maybe_compact(self: &Arc<Self>) -> Result<bool, FsError> {
        self.maybe_compact_with_config(&CompactionConfig::default())
    }

    /// Check if compaction is needed and perform it with custom config
    pub fn maybe_compact_with_config(
        self: &Arc<Self>,
        config: &CompactionConfig,
    ) -> Result<bool, FsError> {
        // First check if compaction is needed
        let stats = self.fragmentation_stats();
        let total_space = self.total_free_space() + self.total_used_space();

        let fragmentation_ratio =
            stats.free_range_count as f64 * stats.average_fragment_size as f64 / total_space as f64;

        if fragmentation_ratio < config.fragmentation_threshold {
            return Ok(false);
        }

        // Identify candidate franges for compaction
        let candidates = match self.find_compaction_candidates(config) {
            | Ok(v) => v,
            | Err(e) => return Err(e),
        };
        if candidates.is_empty() {
            return Ok(false);
        }

        // Perform compaction
        match self.compact_franges(config.block_buffer_size, candidates) {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        Ok(true)
    }

    /// Find candidate franges for compaction
    fn find_compaction_candidates(
        self: &Arc<Self>,
        config: &CompactionConfig,
    ) -> Result<Vec<(u64, FRangeMetadata)>, FsError> {
        let mut candidates = Vec::new();

        // Only consider closed franges
        let open_franges = self.open_franges.read();
        let franges = self.franges.read();

        for entry in franges.iter() {
            let id = entry.key();
            let metadata = entry.value();

            // Skip open franges
            if open_franges.contains(id) {
                continue;
            }

            // Consider franges that might benefit from compaction
            let allocated_size = metadata.range().end() - metadata.range().start();
            let used_size = metadata.length().load(SeqCst);

            // Check if frange is significantly fragmented
            if allocated_size - used_size >= config.min_fragment_size {
                candidates.push((*id, metadata.clone()));
            }

            if candidates.len() >= config.max_compact_batch {
                break;
            }
        }

        Ok(candidates)
    }

    /// Perform compaction on selected franges
    fn compact_franges(
        self: &Arc<Self>,
        buffer_size: usize,
        candidates: Vec<(u64, FRangeMetadata)>,
    ) -> Result<(), FsError> {
        for (id, metadata) in candidates {
            // Skip if frange was opened while preparing compaction
            if self.open_franges.read().contains(&id) {
                continue;
            }

            match self.mmap.advise_range(
                memmap2::Advice::WillNeed,
                ((metadata.range().end()) - metadata.range().start()) as usize,
                metadata.range().start() as usize,
            ) {
                | Ok(_) => {},
                | Err(e) => return Err(IoError(e)),
            };

            // Create new frange with exact size needed
            let new_id = match self.create_frange(metadata.length().load(SeqCst)) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            };
            let new_handle = match self.open_frange(new_id) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            };

            // Read data from old frange
            // TODO(@siennathesane): find a more efficient way to copy data. since we are
            // using the `FRangeHandle` API, realistically we can load the
            // ranges, calculate the offsets, then directly copy data with a
            // single memcpy from the source to the dest.
            let old_handle = match self.open_frange(id) {
                | Ok(v) => v,
                | Err(e) => return Err(e),
            };
            let mut buffer = BytesMut::zeroed(buffer_size); // use 4KB buffer for copying

            let mut remaining = metadata.length().load(SeqCst);
            let mut offset = 0;

            while remaining > 0 {
                let chunk_size = remaining.min(buffer.len() as u64) as usize;
                buffer.resize(chunk_size, 0);

                match old_handle.read_at(offset, &mut buffer) {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };

                match new_handle.write_at(offset, &buffer) {
                    | Ok(_) => {},
                    | Err(e) => return Err(e),
                };

                offset += chunk_size as u64;
                remaining -= chunk_size as u64;
            }

            // close both handles
            match self.close_frange(old_handle) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };

            match self.close_frange(new_handle) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };

            // Get the new range information
            let new_range = {
                match self.franges.read().get(&new_id) {
                    | None => return Err(FRangeNotFound),
                    | Some(v) => v,
                }
                .value()
                .range()
                .clone()
            };

            // Update the original frange's metadata to point to the new location
            {
                let mut updated = match self.franges.read().get(&id) {
                    | None => {
                        return Err(FRangeNotFound);
                    },
                    | Some(v) => v.value().clone(),
                };

                updated.set_range(new_range);
                self.franges.write().insert(updated.id(), updated);
            }

            // Remove the temporary frange's metadata
            {
                let franges = self.franges.write();
                franges.remove(&new_id);
            }

            // Add the old range back to free ranges
            {
                let free_ranges = self.free_ranges.write();
                free_ranges.insert(metadata.range().clone());
            }
        }

        // Finalize metadata persist and coalesce
        match self.persist_metadata() {
            | Ok(_) => {},
            | Err(e) => return Err(e),
        };

        match self.coalesce_free_ranges() {
            | Ok(_) => Ok(()),
            | Err(e) => Err(e),
        }
    }
}

/// Implementations for statistics.
impl Fs {
    /// Calculate total free space available in the filesystem
    pub fn total_free_space(self: &Arc<Self>) -> u64 {
        let free_ranges = self.free_ranges.read();
        let sum = free_ranges
            .iter()
            .map(|range| range.end() - range.start())
            .sum();
        sum
    }

    /// Calculate total space used by allocated franges
    pub fn total_used_space(self: &Arc<Self>) -> u64 {
        let franges = self.franges.read();
        let sum = franges.iter().map(|entry| entry.value().size()).sum();
        sum
    }

    /// Check if a given size can be allocated contiguously
    pub fn can_allocate_contiguous(self: &Arc<Self>, size: u64) -> bool {
        let free_ranges = self.free_ranges.read();
        let can_allocate = free_ranges
            .iter()
            .any(|range| range.end() - range.start() >= size);
        can_allocate
    }

    /// Get fragmentation statistics
    pub fn fragmentation_stats(self: &Arc<Self>) -> FragmentationStats {
        let free_ranges = self.free_ranges.read();
        let mut stats = FragmentationStats::default();

        if free_ranges.is_empty() {
            return stats;
        }

        // Collect the ranges to avoid iterator lifetime issues
        let ranges: Vec<_> = free_ranges.iter().collect();

        for range in ranges {
            let size = range.end() - range.start();
            stats.total_free_space += size;
            stats.free_range_count += 1;
            stats.largest_free_block = stats.largest_free_block.max(size);

            if size < self.page_size as u64 {
                stats.small_fragments += 1;
            }
        }

        if stats.free_range_count > 0 {
            stats.average_fragment_size = stats.total_free_space / stats.free_range_count;
        }

        stats
    }
}
