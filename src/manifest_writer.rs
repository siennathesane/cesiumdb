// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Manifest writer for durable LSM-tree state tracking
//!
//! The manifest writer maintains an append-only log of VersionEdit records.
//! When the file grows too large, it performs atomic rotation by writing
//! a full snapshot to MANIFEST.new and then atomically renaming it.

use std::{
    fs::{
        File,
        OpenOptions,
    },
    io::{
        BufWriter,
        Write,
    },
    path::PathBuf,
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
};

use crate::{
    errs::ManifestError,
    manifest::{
        EditEntry,
        ManifestHeader,
    },
    version::VersionEdit,
};

/// Rotation threshold: 10MB or 10,000 edits
const ROTATION_SIZE_BYTES: u64 = 10 * 1024 * 1024;
const ROTATION_EDIT_COUNT: u64 = 10_000;

/// Manifest writer with atomic rotation support
pub struct ManifestWriter {
    file: BufWriter<File>,
    base_path: PathBuf,
    entries_written: AtomicU64,
    bytes_written: AtomicU64,
}

impl ManifestWriter {
    /// Create a new manifest file
    pub fn create(base_path: PathBuf, created_hlc: u64) -> Result<Self, ManifestError> {
        let manifest_path = base_path.join("MANIFEST");

        // Create the file
        let file = match OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&manifest_path)
        {
            | Ok(f) => f,
            | Err(e) => return Err(ManifestError::Io(e)),
        };

        let mut writer = BufWriter::with_capacity(64 * 1024, file);

        // Write header
        let header = ManifestHeader::new(created_hlc);
        let header_bytes = header.encode();
        if let Err(e) = writer.write_all(&header_bytes) {
            return Err(ManifestError::Io(e));
        }
        if let Err(e) = writer.flush() {
            return Err(ManifestError::Io(e));
        }

        Ok(Self {
            file: writer,
            base_path,
            entries_written: AtomicU64::new(0),
            bytes_written: AtomicU64::new(48), // header size
        })
    }

    /// Open an existing manifest file in append mode
    pub fn open_existing(base_path: PathBuf) -> Result<Self, ManifestError> {
        let manifest_path = base_path.join("MANIFEST");

        // Open in append mode
        let file = match OpenOptions::new()
            
            .append(true)
            .open(&manifest_path)
        {
            | Ok(f) => f,
            | Err(e) => return Err(ManifestError::Io(e)),
        };

        // Get current file size
        let metadata = match std::fs::metadata(&manifest_path) {
            | Ok(m) => m,
            | Err(e) => return Err(ManifestError::Io(e)),
        };
        let file_size = metadata.len();

        let writer = BufWriter::with_capacity(64 * 1024, file);

        // Estimate entry count (rough estimate, actual count determined by reader)
        let estimated_entries = if file_size > 48 {
            (file_size - 48) / 100 // assume ~100 bytes per entry
        } else {
            0
        };

        Ok(Self {
            file: writer,
            base_path,
            entries_written: AtomicU64::new(estimated_entries),
            bytes_written: AtomicU64::new(file_size),
        })
    }

    /// Append a VersionEdit to the manifest
    pub fn append_edit(&mut self, edit: &VersionEdit) -> Result<(), ManifestError> {
        // Encode the edit
        let payload = edit.encode();

        // Get the type discriminant from the first byte
        let edit_type = payload[0];

        // Wrap in EditEntry with CRC
        let entry = EditEntry::new(edit_type, payload);
        let entry_bytes = entry.encode();

        // Write to file
        if let Err(e) = self.file.write_all(&entry_bytes) {
            return Err(ManifestError::Io(e));
        }

        // Update counters
        self.entries_written.fetch_add(1, Ordering::Release);
        self.bytes_written
            .fetch_add(entry_bytes.len() as u64, Ordering::Release);

        Ok(())
    }

    /// Flush the buffer and fsync to disk
    pub fn sync(&mut self) -> Result<(), ManifestError> {
        if let Err(e) = self.file.flush() {
            return Err(ManifestError::Io(e));
        }
        if let Err(e) = self.file.get_ref().sync_all() {
            return Err(ManifestError::Io(e));
        }
        Ok(())
    }

    /// Check if rotation is needed
    pub fn should_rotate(&self) -> bool {
        let bytes = self.bytes_written.load(Ordering::Acquire);
        let entries = self.entries_written.load(Ordering::Acquire);

        bytes >= ROTATION_SIZE_BYTES || entries >= ROTATION_EDIT_COUNT
    }

    /// Rotate the manifest by writing a full snapshot
    ///
    /// This creates MANIFEST.new with a complete snapshot of the version set,
    /// then atomically renames it to MANIFEST.
    pub fn rotate(&mut self, snapshot_edits: Vec<VersionEdit>) -> Result<(), ManifestError> {
        let manifest_new = self.base_path.join("MANIFEST.new");

        // Create new manifest file with snapshot
        {
            let file = match OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&manifest_new)
            {
                | Ok(f) => f,
                | Err(e) => return Err(ManifestError::Io(e)),
            };

            let mut writer = BufWriter::with_capacity(64 * 1024, file);

            // Write header
            let header = ManifestHeader::new(0); // Use 0 for snapshot timestamp
            let header_bytes = header.encode();
            if let Err(e) = writer.write_all(&header_bytes) {
                return Err(ManifestError::Io(e));
            }

            // Write all snapshot edits
            for edit in &snapshot_edits {
                let payload = edit.encode();
                let edit_type = payload[0];
                let entry = EditEntry::new(edit_type, payload);
                let entry_bytes = entry.encode();
                if let Err(e) = writer.write_all(&entry_bytes) {
                    return Err(ManifestError::Io(e));
                }
            }

            // Flush and sync
            if let Err(e) = writer.flush() {
                return Err(ManifestError::Io(e));
            }
            if let Err(e) = writer.get_ref().sync_all() {
                return Err(ManifestError::Io(e));
            }
        }

        // Atomic rename
        let manifest_path = self.base_path.join("MANIFEST");
        if let Err(e) = std::fs::rename(&manifest_new, &manifest_path) {
            return Err(ManifestError::Io(e));
        }

        // Reopen the new file
        let file = match OpenOptions::new()
            
            .append(true)
            .open(&manifest_path)
        {
            | Ok(f) => f,
            | Err(e) => return Err(ManifestError::Io(e)),
        };

        let metadata = match std::fs::metadata(&manifest_path) {
            | Ok(m) => m,
            | Err(e) => return Err(ManifestError::Io(e)),
        };
        let file_size = metadata.len();

        self.file = BufWriter::with_capacity(64 * 1024, file);
        self.bytes_written.store(file_size, Ordering::Release);
        self.entries_written
            .store(snapshot_edits.len() as u64, Ordering::Release);

        Ok(())
    }

    /// Get the number of entries written
    pub fn entry_count(&self) -> u64 {
        self.entries_written.load(Ordering::Acquire)
    }

    /// Get the current file size in bytes
    pub fn file_size(&self) -> u64 {
        self.bytes_written.load(Ordering::Acquire)
    }
}

impl Drop for ManifestWriter {
    fn drop(&mut self) {
        // Ensure data is flushed on drop
        let _ = self.file.flush();
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_create_manifest() {
        let temp_dir = TempDir::new().unwrap();
        let writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 12345).unwrap();

        assert_eq!(writer.entry_count(), 0);
        assert_eq!(writer.file_size(), 48); // header size

        let manifest_path = temp_dir.path().join("MANIFEST");
        assert!(manifest_path.exists());
    }

    #[test]
    fn test_append_edit() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        let edit = VersionEdit::AddL0Segment {
            segment_id: 123,
            key_range: (b"start".to_vec(), b"end".to_vec()),
            size: 4096,
        };

        writer.append_edit(&edit).unwrap();
        assert_eq!(writer.entry_count(), 1);
        assert!(writer.file_size() > 48);
    }

    #[test]
    fn test_multiple_edits() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        for i in 0..10 {
            let edit = VersionEdit::AddL0Segment {
                segment_id: i,
                key_range: (vec![i as u8], vec![i as u8 + 1]),
                size: 1000 + i,
            };
            writer.append_edit(&edit).unwrap();
        }

        assert_eq!(writer.entry_count(), 10);
    }

    #[test]
    fn test_should_rotate_size() {
        let temp_dir = TempDir::new().unwrap();
        let writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        // Manually set bytes_written to exceed threshold
        writer
            .bytes_written
            .store(ROTATION_SIZE_BYTES + 1, Ordering::Release);

        assert!(writer.should_rotate());
    }

    #[test]
    fn test_should_rotate_count() {
        let temp_dir = TempDir::new().unwrap();
        let writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        // Manually set entry count to exceed threshold
        writer
            .entries_written
            .store(ROTATION_EDIT_COUNT + 1, Ordering::Release);

        assert!(writer.should_rotate());
    }

    #[test]
    fn test_rotate() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        // Write some edits
        for i in 0..5 {
            let edit = VersionEdit::UpdateSequence { sequence: i };
            writer.append_edit(&edit).unwrap();
        }

        // Create snapshot edits
        let snapshot = vec![
            VersionEdit::AddL0Segment {
                segment_id: 1,
                key_range: (b"a".to_vec(), b"z".to_vec()),
                size: 1000,
            },
            VersionEdit::AddSegment {
                level: 1,
                segment_id: 2,
                key_range: (b"a".to_vec(), b"m".to_vec()),
                size: 2000,
            },
        ];

        writer.rotate(snapshot.clone()).unwrap();

        // Entry count should reset to snapshot size
        assert_eq!(writer.entry_count(), 2);
    }

    #[test]
    fn test_open_existing() {
        let temp_dir = TempDir::new().unwrap();

        // Create and write some edits
        {
            let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();
            let edit = VersionEdit::UpdateSequence { sequence: 42 };
            writer.append_edit(&edit).unwrap();
            writer.sync().unwrap();
        }

        // Reopen and verify
        let writer = ManifestWriter::open_existing(temp_dir.path().to_path_buf()).unwrap();
        assert!(writer.file_size() > 48);
    }

    #[test]
    fn test_sync() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        let edit = VersionEdit::UpdateSequence { sequence: 100 };
        writer.append_edit(&edit).unwrap();
        writer.sync().unwrap();

        // File should exist and have data
        let manifest_path = temp_dir.path().join("MANIFEST");
        let metadata = std::fs::metadata(manifest_path).unwrap();
        assert!(metadata.len() > 48);
    }
}
