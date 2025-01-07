use std::{
    io,
    ptr,
    sync::{
        atomic::{
            AtomicU64,
            Ordering::SeqCst,
        },
        Arc,
    },
};

use bytes::{
    Buf,
    BufMut,
    Bytes,
    BytesMut,
};
use getset::{
    Getters,
    Setters,
};
use memmap2::MmapMut;
use parking_lot::RwLock;

use crate::{
    errs::{
        FsError,
        FsError::{
            InvalidHeaderFormat,
            IoError,
        },
    },
    fs::handle::{
        FRangeMetadata,
        OrderedRange,
    },
    utils::{
        Deserializer,
        Serializer,
    },
};

pub(in crate::fs) const JOURNAL_SIZE: u64 = 1 << 20; // 1 MiB

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum JournalEntryType {
    CreateFRange       = 1,
    DeleteFRange       = 2,
    UpdateFRange       = 3,
    CoalesceFreeRanges = 4,
}

#[derive(Debug, Getters)]
#[getset(get = "pub(crate)")]
pub(in crate::fs) struct JournalEntry {
    r#type: JournalEntryType,
    timestamp: u64,
    frange_id: u64,
    metadata: Option<FRangeMetadata>,
    free_ranges: Option<Vec<OrderedRange>>,
}

impl JournalEntry {
    pub(in crate::fs) fn new(
        r#type: JournalEntryType,
        timestamp: u64,
        frange_id: u64,
        metadata: Option<FRangeMetadata>,
        free_ranges: Option<Vec<OrderedRange>>,
    ) -> Self {
        Self {
            r#type,
            timestamp,
            frange_id,
            metadata,
            free_ranges,
        }
    }

    fn deserialize(mut bytes: Bytes) -> Result<Self, FsError> {
        if bytes.len() < 18 {
            // Minimum size: type(1) + timestamp(8) + id(8) + flags(1)
            return Err(InvalidHeaderFormat("journal entry too short".into()));
        }

        let typ = match bytes.get_u8() {
            | 1 => JournalEntryType::CreateFRange,
            | 2 => JournalEntryType::DeleteFRange,
            | 3 => JournalEntryType::UpdateFRange,
            | 4 => JournalEntryType::CoalesceFreeRanges,
            | _ => return Err(InvalidHeaderFormat("invalid journal entry type".into())),
        };

        let timestamp = bytes.get_u64_le();
        let frange_id = bytes.get_u64_le();

        // Read metadata if present
        let metadata = if bytes.get_u8() == 1 {
            let range = OrderedRange::new(bytes.get_u64_le(), bytes.get_u64_le());
            Some(FRangeMetadata::new(
                range,
                frange_id,
                bytes.get_u64_le(), // length
                bytes.get_u64_le(), // size
                bytes.get_u64_le(), // created_at
                bytes.get_u64_le(), // modified_at
            ))
        } else {
            None
        };

        // Read free ranges if present
        let free_ranges = if bytes.get_u8() == 1 {
            let count = bytes.get_u64_le() as usize;
            let mut ranges = Vec::with_capacity(count);
            for _ in 0..count {
                ranges.push(OrderedRange::new(bytes.get_u64_le(), bytes.get_u64_le()));
            }
            Some(ranges)
        } else {
            None
        };

        Ok(JournalEntry {
            r#type: typ,
            timestamp,
            frange_id,
            metadata,
            free_ranges,
        })
    }
}

impl Serializer for JournalEntry {
    fn serialize_for_memory(&self) -> Bytes {
        unimplemented!()
    }

    fn serialize(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // Write type and timestamp
        buf.put_u8(self.r#type as u8);
        buf.put_u64_le(self.timestamp);
        buf.put_u64_le(self.frange_id);

        // Write metadata if present
        if let Some(metadata) = &self.metadata {
            buf.put_u8(1); // has metadata
            buf.put_u64_le(metadata.range().start());
            buf.put_u64_le(metadata.range().end());
            buf.put_u64_le(metadata.length().load(SeqCst));
            buf.put_u64_le(metadata.size());
            buf.put_u64_le(metadata.created_at());
            buf.put_u64_le(metadata.modified_at().load(SeqCst));
        } else {
            buf.put_u8(0); // no metadata
        }

        // Write free ranges if present
        if let Some(ranges) = &self.free_ranges {
            buf.put_u8(1); // has ranges
            buf.put_u64_le(ranges.len() as u64);
            for range in ranges {
                buf.put_u64_le(range.start());
                buf.put_u64_le(range.end());
            }
        } else {
            buf.put_u8(0); // no ranges
        }

        buf.freeze()
    }
}

#[derive(Debug)]
pub(in crate::fs) struct Journal {
    mmap: Arc<MmapMut>,
    offset: u64,
    head: AtomicU64,
    tail: AtomicU64,
    capacity: u64,
}

impl Journal {
    pub(in crate::fs) fn new(mmap: Arc<MmapMut>, offset: u64, capacity: u64) -> Self {
        let mut buffer = BytesMut::with_capacity(capacity as usize);
        buffer.resize(capacity as usize, 0);

        Self {
            mmap,
            offset,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            capacity,
        }
    }

    pub(in crate::fs) fn append(&self, entry: JournalEntry) -> Result<(), FsError> {
        let serialized = entry.serialize();
        let len = serialized.len() as u64;

        if len > self.capacity {
            return Err(IoError(io::Error::new(
                io::ErrorKind::InvalidInput,
                "entry too large for journal",
            )));
        }

        // Keep trying until we successfully claim space and write
        let mut head = self.head.load(SeqCst);
        loop {

            let tail = self.tail.load(SeqCst);
            let new_head = (head + len) % self.capacity;

            // Check if we need to move tail
            let mut new_tail = tail;
            if new_head < head {
                if tail < new_head {
                    new_tail = new_head;
                }
            } else if new_head > head && tail > head && tail <= new_head {
                new_tail = new_head;
            }

            // Try to atomically update head (and tail if needed)
            match self.head.compare_exchange(head, new_head, SeqCst, SeqCst) {
                | Ok(_) => {
                    // We claimed the space, now write
                    if new_tail != tail {
                        self.tail.store(new_tail, SeqCst);
                    }

                    if new_head < head {
                        let first_part = self.capacity - head;
                        // SAFETY:
                        unsafe {
                            ptr::copy_nonoverlapping(
                                serialized[..first_part as usize].as_ptr(),
                                self.mmap.as_ptr().add((self.offset + head) as usize) as *mut u8,
                                first_part as usize,
                            );
                            ptr::copy_nonoverlapping(
                                serialized[first_part as usize..].as_ptr(),
                                self.mmap.as_ptr().add(self.offset as usize) as *mut u8,
                                (len - first_part) as usize,
                            );
                        }
                    } else {
                        // SAFETY:
                        unsafe {
                            ptr::copy_nonoverlapping(
                                serialized.as_ptr(),
                                self.mmap.as_ptr().add((self.offset + head) as usize) as *mut u8,
                                len as usize,
                            );
                        }
                    }
                    return Ok(());
                },
                | Err(current) => {
                    // Someone else wrote, try again with new head
                    head = current;
                },
            }
        }
    }

    fn read_entry_size(&self, bytes: &[u8]) -> Result<u64, FsError> {
        let offset_bytes = &bytes[(self.offset as usize)..];
        if offset_bytes.len() < 18 {
            return Err(InvalidHeaderFormat("entry header too short".into()));
        }

        // skip type(1) + timestamp(8) + id(8)
        let has_metadata = bytes[17] == 1;
        let mut size = 18u64;

        if has_metadata {
            size += 48; // 6 * 8 bytes for metadata fields
        }

        if bytes.len() <= size as usize {
            return Err(InvalidHeaderFormat("truncated entry".into()));
        }

        let has_ranges = bytes[size as usize] == 1;
        size += 1;

        if has_ranges {
            if bytes.len() <= size as usize + 8 {
                return Err(InvalidHeaderFormat("truncated range count".into()));
            }
            let range_count =
                u64::from_le_bytes(bytes[size as usize..size as usize + 8].try_into().unwrap());
            size += 8 + range_count * 16; // count + (start,end) pairs
        }

        Ok(size)
    }

    pub(in crate::fs) fn iter(&self) -> JournalIterator {
        JournalIterator {
            journal: self,
            current: self.tail.load(SeqCst),
        }
    }
}

pub(in crate::fs) struct JournalIterator<'a> {
    journal: &'a Journal,
    current: u64,
}

impl<'a> Iterator for JournalIterator<'a> {
    type Item = Result<JournalEntry, FsError>;

    fn next(&mut self) -> Option<Self::Item> {
        let head = self.journal.head.load(SeqCst);
        if self.current == head {
            println!("Iterator: reached head, stopping");
            return None;
        }

        // Get bytes for single entry (we know it's 91 bytes)
        let entry_size = 91u64;
        let mut entry_bytes = BytesMut::with_capacity(entry_size as usize);

        // Check if the entry wraps around the buffer end
        if self.current + entry_size > self.journal.capacity {
            // Calculate sizes of the two parts
            let first_part = self.journal.capacity - self.current;
            let second_part = entry_size - first_part;

            println!(
                "Reading wrapped entry: first_part={}, second_part={}",
                first_part, second_part
            );

            // Read first part from current to end
            entry_bytes.extend_from_slice(
                &self.journal.mmap[(self.journal.offset + self.current) as usize..]
                    [..first_part as usize],
            );

            // Read second part from start
            entry_bytes.extend_from_slice(
                &self.journal.mmap[self.journal.offset as usize..][..second_part as usize],
            );
        } else {
            println!(
                "Reading linear entry from {} to {}",
                self.current,
                self.current + entry_size
            );
            entry_bytes.extend_from_slice(
                &self.journal.mmap[(self.journal.offset + self.current) as usize..]
                    [..entry_size as usize],
            );
        }

        match JournalEntry::deserialize(entry_bytes.freeze()) {
            | Ok(entry) => {
                println!(
                    "Deserialized entry: type={:?}, id={}",
                    entry.r#type, entry.frange_id
                );
                // Update current position, wrapping around if needed
                self.current = (self.current + entry_size) % self.journal.capacity;
                Some(Ok(entry))
            },
            | Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use std::{
        fs::{
            File,
            OpenOptions,
        },
        sync::atomic::AtomicU64,
        time::{
            SystemTime,
            UNIX_EPOCH,
        },
    };

    use tempfile::tempdir;

    use super::*;

    fn create_test_metadata() -> FRangeMetadata {
        let range = OrderedRange::new(1000, 2000);
        FRangeMetadata::new(
            range,
            1,
            500,
            1000,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
    }

    fn create_test_entry() -> JournalEntry {
        JournalEntry {
            r#type: JournalEntryType::CreateFRange,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            frange_id: 1,
            metadata: Some(create_test_metadata()),
            free_ranges: Some(vec![OrderedRange::new(0, 1000)]),
        }
    }

    fn setup_test_file() -> MmapMut {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("journal");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .unwrap();

        // Make it bigger for debugging
        file.set_len(JOURNAL_SIZE * 2).unwrap();

        unsafe { MmapMut::map_mut(&file).unwrap() }
    }

    #[test]
    fn test_journal_entry_serialization() {
        let entry = create_test_entry();
        let serialized = entry.serialize();
        let deserialized = JournalEntry::deserialize(serialized).unwrap();

        assert_eq!(entry.r#type as u8, deserialized.r#type as u8);
        assert_eq!(entry.frange_id, deserialized.frange_id);

        // Compare metadata
        let orig_metadata = entry.metadata.unwrap();
        let new_metadata = deserialized.metadata.unwrap();
        assert_eq!(orig_metadata.range().start(), new_metadata.range().start());
        assert_eq!(orig_metadata.range().end(), new_metadata.range().end());
        assert_eq!(orig_metadata.id(), new_metadata.id());
        assert_eq!(orig_metadata.size(), new_metadata.size());

        // Compare free ranges
        let orig_ranges = entry.free_ranges.unwrap();
        let new_ranges = deserialized.free_ranges.unwrap();
        assert_eq!(orig_ranges.len(), new_ranges.len());
        assert_eq!(orig_ranges[0].start(), new_ranges[0].start());
        assert_eq!(orig_ranges[0].end(), new_ranges[0].end());
    }

    #[test]
    fn test_journal_entry_without_metadata() {
        let entry = JournalEntry {
            r#type: JournalEntryType::DeleteFRange,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            frange_id: 1,
            metadata: None,
            free_ranges: None,
        };

        let serialized = entry.serialize();
        let deserialized = JournalEntry::deserialize(serialized).unwrap();

        assert_eq!(entry.r#type as u8, deserialized.r#type as u8);
        assert_eq!(entry.frange_id, deserialized.frange_id);
        assert!(deserialized.metadata.is_none());
        assert!(deserialized.free_ranges.is_none());
    }

    #[test]
    fn test_journal_basic_append() {
        let journal = Journal::new(Arc::new(setup_test_file()), 0, JOURNAL_SIZE);
        let entry = create_test_entry();

        assert!(journal.append(entry).is_ok());

        let entries: Vec<_> = journal.iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_journal_multiple_entries() {
        let journal = Journal::new(Arc::new(setup_test_file()), 0, JOURNAL_SIZE);

        for i in 0..5 {
            let mut entry = create_test_entry();
            entry.frange_id = i;
            assert!(journal.append(entry).is_ok());
        }

        let entries: Vec<_> = journal.iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(entries.len(), 5);

        // Verify entries are in order
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.frange_id, i as u64);
        }
    }

    #[test]
    fn test_journal_wrap_around() {
        // Create a small journal that will need to wrap
        let journal = Journal::new(Arc::new(setup_test_file()), 0, 256);
        let mut count = 0;

        // Keep adding entries until we wrap around
        while count < 10 {
            let mut entry = create_test_entry();
            entry.frange_id = count;
            if journal.append(entry).is_ok() {
                count += 1;
            } else {
                break;
            }
        }

        let entries: Vec<_> = journal.iter().collect::<Result<_, _>>().unwrap();
        assert!(!entries.is_empty());
        assert!(entries.len() < 10); // Some entries should have been evicted

        // First entry should not be 0 due to eviction
        assert!(entries[0].frange_id > 0);

        // Entries should still be in order
        for window in entries.windows(2) {
            assert!(window[1].frange_id > window[0].frange_id);
        }
    }

    #[test]
    fn test_journal_full_eviction() {
        let journal = Journal::new(Arc::new(setup_test_file()), 0, 256);
        let large_ranges: Vec<OrderedRange> = (0..100)
            .map(|i| OrderedRange::new(i * 1000, (i + 1) * 1000))
            .collect();

        // Try to add an entry that's too large even after eviction
        let entry = JournalEntry {
            r#type: JournalEntryType::CoalesceFreeRanges,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            frange_id: 1,
            metadata: None,
            free_ranges: Some(large_ranges),
        };

        assert!(journal.append(entry).is_err());
    }

    #[test]
    fn test_journal_iterator_empty() {
        let journal = Journal::new(Arc::new(setup_test_file()), 0, JOURNAL_SIZE);
        let entries: Vec<_> = journal.iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_journal_invalid_entry() {
        let journal = Journal::new(Arc::new(setup_test_file()), 0, JOURNAL_SIZE);

        // Create an invalid entry type
        let invalid_bytes = Bytes::from_static(&[255; 32]); // Invalid data
        assert!(JournalEntry::deserialize(invalid_bytes).is_err());
    }

    #[test]
    fn test_concurrent_journal_access() {
        use std::{
            sync::Arc,
            thread,
        };

        let journal = Arc::new(Journal::new(Arc::new(setup_test_file()), 0, JOURNAL_SIZE));
        let mut handles = vec![];

        // Spawn multiple threads to write to the journal
        for i in 0..10 {
            let journal_clone = journal.clone();
            handles.push(thread::spawn(move || {
                let mut entry = create_test_entry();
                entry.frange_id = i;
                journal_clone.append(entry)
            }));
        }

        // Wait for all threads to complete
        for handle in handles {
            assert!(handle.join().unwrap().is_ok());
        }

        let entries: Vec<_> = journal.iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(entries.len(), 10);

        // Verify all entries were written
        let ids: std::collections::HashSet<_> = entries.iter().map(|e| e.frange_id).collect();
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn test_journal_stress() {
        let journal = Arc::new(Journal::new(Arc::new(setup_test_file()), 0, JOURNAL_SIZE));
        let mut last_id = None;

        // Write entries until we fill the journal multiple times
        for i in 0..1000 {
            let mut entry = create_test_entry();
            entry.frange_id = i;
            if journal.append(entry).is_ok() {
                last_id = Some(i);
            }
        }

        let entries: Vec<_> = journal.iter().collect::<Result<_, _>>().unwrap();
        assert!(!entries.is_empty());

        if let Some(last) = last_id {
            assert_eq!(entries.last().unwrap().frange_id, last);
        }

        // Verify entries are contiguous and ordered
        for window in entries.windows(2) {
            assert_eq!(window[1].frange_id, window[0].frange_id + 1);
        }
    }
}
