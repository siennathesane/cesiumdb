// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

//! Manifest reader for crash recovery
//!
//! The manifest reader parses the MANIFEST file and replays VersionEdit
//! records to reconstruct the LSM-tree state.

use std::{
    fs::File,
    io::Read,
    path::Path,
};

use bytes::Bytes;

use crate::{
    errs::ManifestError,
    levels::VersionSet,
    manifest::{
        EditEntry,
        ManifestHeader,
    },
    version::VersionEdit,
};

/// Manifest reader for recovery
pub struct ManifestReader {
    file: File,
}

impl ManifestReader {
    /// Open a manifest file for reading
    pub fn open(base_path: &Path) -> Result<Self, ManifestError> {
        let manifest_path = base_path.join("MANIFEST");
        let file = match File::open(manifest_path) {
            | Ok(f) => f,
            | Err(e) => return Err(ManifestError::Io(e)),
        };
        Ok(Self { file })
    }

    /// Read all edits from the manifest
    ///
    /// Returns all successfully parsed edits. If corruption is encountered,
    /// returns edits up to the corruption point.
    pub fn read_all_edits(&mut self) -> Result<Vec<VersionEdit>, ManifestError> {
        let mut buf = Vec::new();
        if let Err(e) = self.file.read_to_end(&mut buf) {
            return Err(ManifestError::Io(e));
        }

        if buf.len() < 48 {
            return Err(ManifestError::CorruptedHeader);
        }

        // Parse and validate header
        let header_bytes = Bytes::copy_from_slice(&buf[0..48]);
        let header = match ManifestHeader::decode(header_bytes) {
            | Ok(h) => h,
            | Err(e) => return Err(e),
        };

        if !header.crc_enabled() {
            // For now, we require CRC to be enabled
            return Err(ManifestError::CorruptedHeader);
        }

        let mut edits = Vec::new();
        let mut offset = 48;

        // Parse edit entries
        while offset < buf.len() {
            let remaining = Bytes::copy_from_slice(&buf[offset..]);

            match EditEntry::decode(remaining) {
                | Ok((entry, bytes_read)) => {
                    // Decode the VersionEdit from the payload
                    match VersionEdit::decode(entry.payload) {
                        | Ok(edit) => {
                            edits.push(edit);
                            offset += bytes_read;
                        },
                        | Err(e) => {
                            // Log warning but continue
                            tracing::warn!(
                                "Failed to decode VersionEdit at offset {}: {:?}",
                                offset,
                                e
                            );
                            break;
                        },
                    }
                },
                | Err(e) => {
                    // Log warning but return edits parsed so far
                    tracing::warn!("Corruption detected at offset {}: {:?}", offset, e);
                    break;
                },
            }
        }

        Ok(edits)
    }

    /// Recover a VersionSet from the manifest
    ///
    /// This reads all edits and applies them to reconstruct the LSM-tree state.
    /// If the manifest doesn't exist, returns None.
    pub fn recover_version_set(
        base_path: &Path,
        num_levels: usize,
    ) -> Result<Option<VersionSet>, ManifestError> {
        let manifest_path = base_path.join("MANIFEST");

        if !manifest_path.exists() {
            return Ok(None);
        }

        let mut reader = match Self::open(base_path) {
            | Ok(r) => r,
            | Err(e) => return Err(e),
        };
        let edits = match reader.read_all_edits() {
            | Ok(e) => e,
            | Err(e) => return Err(e),
        };

        if edits.is_empty() {
            return Ok(Some(VersionSet::new(0, num_levels)));
        }

        // Create initial version set
        let mut version = VersionSet::new(0, num_levels);

        // Apply all edits
        for edit in edits {
            // Note: apply() can fail if segments are missing, but we ignore
            // those errors during recovery to be resilient
            match edit.apply(&mut version, base_path) {
                | Ok(()) => {},
                | Err(e) => {
                    tracing::warn!("Failed to apply edit during recovery: {:?}", e);
                    // Continue with other edits
                },
            }
        }

        Ok(Some(version))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::{
        manifest_writer::ManifestWriter,
        version::VersionEdit,
    };

    #[test]
    fn test_read_empty_manifest() {
        let temp_dir = TempDir::new().unwrap();

        // Create empty manifest
        let _writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

        // Read it back
        let mut reader = ManifestReader::open(temp_dir.path()).unwrap();
        let edits = reader.read_all_edits().unwrap();

        assert_eq!(edits.len(), 0);
    }

    #[test]
    fn test_read_single_edit() {
        let temp_dir = TempDir::new().unwrap();

        // Write one edit
        {
            let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();
            let edit = VersionEdit::UpdateSequence { sequence: 42 };
            writer.append_edit(&edit).unwrap();
            writer.sync().unwrap();
        }

        // Read it back
        let mut reader = ManifestReader::open(temp_dir.path()).unwrap();
        let edits = reader.read_all_edits().unwrap();

        assert_eq!(edits.len(), 1);
        match &edits[0] {
            | VersionEdit::UpdateSequence { sequence } => assert_eq!(*sequence, 42),
            | _ => panic!("wrong edit type"),
        }
    }

    #[test]
    fn test_read_multiple_edits() {
        let temp_dir = TempDir::new().unwrap();

        // Write multiple edits
        {
            let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();

            writer
                .append_edit(&VersionEdit::AddL0Segment {
                    segment_id: 1,
                    key_range: (b"a".to_vec(), b"z".to_vec()),
                    size: 1000,
                })
                .unwrap();

            writer
                .append_edit(&VersionEdit::AddSegment {
                    level: 1,
                    segment_id: 2,
                    key_range: (b"a".to_vec(), b"m".to_vec()),
                    size: 2000,
                })
                .unwrap();

            writer
                .append_edit(&VersionEdit::RemoveSegment {
                    level: 1,
                    segment_id: 2,
                })
                .unwrap();

            writer.sync().unwrap();
        }

        // Read them back
        let mut reader = ManifestReader::open(temp_dir.path()).unwrap();
        let edits = reader.read_all_edits().unwrap();

        assert_eq!(edits.len(), 3);
    }

    #[test]
    fn test_recover_version_set_no_manifest() {
        let temp_dir = TempDir::new().unwrap();

        let result = ManifestReader::recover_version_set(temp_dir.path(), 7).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_recover_version_set_empty() {
        let temp_dir = TempDir::new().unwrap();

        // Create empty manifest
        {
            let _writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();
        }

        let result = ManifestReader::recover_version_set(temp_dir.path(), 7).unwrap();
        assert!(result.is_some());

        let version = result.unwrap();
        assert_eq!(version.sequence, 0);
    }

    #[test]
    fn test_recover_version_set_with_sequence() {
        let temp_dir = TempDir::new().unwrap();

        // Write sequence update
        {
            let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();
            writer
                .append_edit(&VersionEdit::UpdateSequence { sequence: 123 })
                .unwrap();
            writer.sync().unwrap();
        }

        let result = ManifestReader::recover_version_set(temp_dir.path(), 7).unwrap();
        let version = result.unwrap();
        assert_eq!(version.sequence, 123);
    }

    #[test]
    fn test_corrupted_entry_stops_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let manifest_path = temp_dir.path().join("MANIFEST");

        // Write valid edits then corrupt the file
        {
            let mut writer = ManifestWriter::create(temp_dir.path().to_path_buf(), 0).unwrap();
            writer
                .append_edit(&VersionEdit::UpdateSequence { sequence: 1 })
                .unwrap();
            writer
                .append_edit(&VersionEdit::UpdateSequence { sequence: 2 })
                .unwrap();
            writer.sync().unwrap();
        }

        // Append garbage to the file
        {
            use std::{
                fs::OpenOptions,
                io::Write,
            };

            let mut file = OpenOptions::new()
                .append(true)
                .open(&manifest_path)
                .unwrap();
            file.write_all(&[0xff; 100]).unwrap();
        }

        // Should read the 2 valid edits and stop at corruption
        let mut reader = ManifestReader::open(temp_dir.path()).unwrap();
        let edits = reader.read_all_edits().unwrap();

        assert_eq!(edits.len(), 2);
    }
}
