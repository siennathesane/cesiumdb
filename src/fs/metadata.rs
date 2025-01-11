use std::ptr;
use std::sync::atomic::{fence, AtomicBool};
use std::sync::atomic::Ordering::SeqCst;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use crossbeam_skiplist::{SkipMap, SkipSet};
use getset::{CopyGetters, Getters};
use gxhash::{HashSet, HashSetExt};
use crate::errs::FsError;
use crate::errs::FsError::InvalidHeaderFormat;
use crate::fs::Fs;
use crate::fs::handle::{FRangeMetadata, OrderedRange};

#[derive(Debug)]
pub(in crate::fs) struct FsMetadata {
    pub(in crate::fs) franges: SkipMap<u64, FRangeMetadata>,
    pub(in crate::fs) free_ranges: SkipSet<OrderedRange>,
}

impl FsMetadata {
    pub(in crate::fs) fn new() -> Self {
        Self {
            franges: SkipMap::new(),
            free_ranges: SkipSet::new(),
        }
    }

    fn from_fs(fs: &Fs) -> Self {
        let metadata = Self::new();

        {
            let franges = fs.franges.read();
            for entry in franges.iter() {
                metadata.franges.insert(*entry.key(), entry.value().clone());
            }
        }

        {
            let free_ranges = fs.free_ranges.read();
            for range in free_ranges.iter() {
                metadata.free_ranges.insert(range.value().clone());
            }
        }

        metadata
    }

    pub(in crate::fs) unsafe fn serialize_changes(
        &self,
        changes: &MetadataChanges,
        ptr: *const u8,
        mut_ptr: *mut u8,
        len: usize,
        offset: usize,
    ) -> Result<(), FsError> {
        if !changes.has_changes() {
            return Ok(());
        }

        let mut modified_franges: Vec<u64> = changes
            .modified_franges
            .iter()
            .map(|e| *e.value())
            .collect();
        modified_franges.sort();

        let mut current_id_idx = 0;
        let mut position = offset;
        let mut next_update_bytes = match self.franges.get(&modified_franges[0]) {
            | Some(v) => Some(v.value().serialize()),
            | None => panic!("frange {} not found", modified_franges[0]),
        };

        while current_id_idx < modified_franges.len() {
            // Read current record details
            let mut size_bytes = [0u8; 4];
            ptr::copy_nonoverlapping(
                ptr.add(position),
                size_bytes.as_mut_ptr(),
                4
            );
            let current_size = u32::from_le_bytes(size_bytes);

            let mut id_bytes = [0u8; 8];
            ptr::copy_nonoverlapping(
                ptr.add(position + 4),
                id_bytes.as_mut_ptr(),
                8
            );
            let current_id = u64::from_le_bytes(id_bytes);

            if current_id == modified_franges[current_id_idx] {
                let new_bytes = next_update_bytes.take().unwrap();
                let new_size = new_bytes.len();
                let total_new_size = new_size + 4;
                let total_old_size = current_size as usize + 4;

                next_update_bytes = if current_id_idx + 1 < modified_franges.len() {
                    match self.franges.get(&modified_franges[current_id_idx + 1]) {
                        | Some(v) => Some(v.value().serialize()),
                        | None => None,
                    }
                } else {
                    None
                };

                if total_new_size != total_old_size {
                    let shift = total_new_size as isize - total_old_size as isize;
                    let next_pos = position + total_old_size;
                    let metadata_end = offset + total_old_size;

                    if shift > 0 && next_pos + shift as usize > metadata_end {
                        return Err(FsError::InsufficientSpace);
                    }

                    ptr::copy(
                        ptr.add(next_pos),
                        mut_ptr.add(next_pos + shift as usize),
                        len - next_pos
                    );
                }

                // Write size prefix
                let size_bytes = (new_size as u32).to_le_bytes();
                ptr::copy_nonoverlapping(
                    size_bytes.as_ptr(),
                    mut_ptr.add(position),
                    4
                );

                // Write record data
                ptr::copy_nonoverlapping(
                    new_bytes.as_ptr(),
                    mut_ptr.add(position + 4),
                    new_size
                );

                current_id_idx += 1;
                position += total_new_size;
            } else {
                position += current_size as usize + 4;
            }

            if position >= len {
                break;
            }
        }

        fence(SeqCst);
        Ok(())
    }

    pub(in crate::fs) fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // write franges count
        buf.put_u64_le(self.franges.len() as u64);

        // write each frange entry
        for entry in self.franges.iter() {
            // write the size of the frange entry
            let encoded = entry.value().serialize();

            buf.put_u32_le(encoded.len() as u32);
            buf.put_slice(&encoded);
        }

        // write free ranges count
        buf.put_u64_le(self.free_ranges.len() as u64);

        // write each free range
        for range in self.free_ranges.iter() {
            buf.put_u64_le(range.start);
            buf.put_u64_le(range.end);
        }

        buf.freeze()
    }

    pub(in crate::fs) fn deserialize(mut bytes: Bytes) -> Result<Self, FsError> {
        let fs_metadata = Self::new();

        // Read modified franges
        let modified_count = bytes.get_u64_le() as usize;
        let mut seen_ids = HashSet::new();

        for _ in 0..modified_count {
            let key = bytes.get_u64_le();
            seen_ids.insert(key);

            let range = OrderedRange::new(bytes.get_u64_le(), bytes.get_u64_le());
            let metadata = FRangeMetadata::new(
                range,
                bytes.get_u64_le(), // id
                bytes.get_u64_le(), // length
                bytes.get_u64_le(), // size
                bytes.get_u64_le(), // created_at
                bytes.get_u64_le(), // modified_at
            );

            fs_metadata.franges.insert(key, metadata);
        }

        // Read free range status
        let free_ranges_modified = bytes.get_u8() != 0;
        if free_ranges_modified {
            // Read complete free range state
            let free_range_count = bytes.get_u64_le() as usize;
            for _ in 0..free_range_count {
                let range = OrderedRange::new(bytes.get_u64_le(), bytes.get_u64_le());
                fs_metadata.free_ranges.insert(range);
            }
        }

        // Verify we have all franges
        let next_id = bytes.get_u64_le(); // Add this to format 1
        for id in 0..next_id {
            if !seen_ids.contains(&id) && id > 0 {
                // This ID should exist but wasn't in our changes
                return Err(InvalidHeaderFormat("incomplete metadata state".into()));
            }
        }

        Ok(fs_metadata)
    }

    // Calculate total serialized size for pre-allocation
    fn serialized_size(&self) -> usize {
        // 1 byte for format
        // 8 bytes for franges count
        // for each frange: 8 (key) + 7*8 (metadata fields)
        // 8 bytes for free ranges count
        // for each free range: 2*8 (start/end)
        1 + 8 + (self.franges.len() * (8 + 56)) + 8 + (self.free_ranges.len() * 16)
    }

    fn apply_changes(&mut self, other: &FsMetadata, changes: &MetadataChanges) {
        for id in changes.modified_franges.iter() {
            if let Some(metadata) = other.franges.get(&id) {
                self.franges.insert(*id, metadata.value().clone());
            }
        }

        if !changes.modified_ranges.is_empty() {
            for range in other.free_ranges.iter() {
                self.free_ranges.insert(range.value().clone());
            }
        }
    }
}

#[derive(Default)]
pub(in crate::fs) struct MetadataChanges {
    pub(in crate::fs) modified_franges: SkipSet<u64>,         // IDs of modified franges
    pub(in crate::fs) modified_ranges: SkipSet<OrderedRange>, // Modified free ranges
    pub(in crate::fs) header_modified: AtomicBool,
}

impl MetadataChanges {
    fn new() -> Self {
        Self::default()
    }

    pub(in crate::fs) fn mark_frange_modified(&self, id: u64) {
        self.modified_franges.insert(id);
        self.header_modified.store(true, SeqCst);
    }

    pub(in crate::fs) fn mark_range_modified(&self, range: OrderedRange) {
        self.modified_ranges.insert(range);
    }

    pub(in crate::fs) fn clear(&self) {
        self.modified_franges.clear();
        self.modified_ranges.clear();
        self.header_modified.store(false, SeqCst);
    }

    pub(in crate::fs) fn has_changes(&self) -> bool {
        !self.modified_franges.is_empty() ||
            !self.modified_ranges.is_empty() ||
            self.header_modified.load(SeqCst)
    }
}
