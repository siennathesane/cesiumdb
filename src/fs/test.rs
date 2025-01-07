#![allow(non_snake_case)]

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod e2e_tests {
    use std::{
        fs::OpenOptions,
        sync::Arc,
    };
    use memmap2::MmapMut;
    use proptest::{
        collection::vec,
        proptest,
    };
    use rand::{
        thread_rng,
        Rng,
        RngCore,
    };
    use tempfile::tempdir;
    use tokio::task;
    use crate::fs::core::{Fs, FsHeader, INITIAL_METADATA_SIZE};
    use super::*;

    const TEST_FILE_SIZE: u64 = 1024 * 1024 * 100; // 100MB for testing

    async fn setup_test_fs() -> (Arc<Fs>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .unwrap();

        file.set_len(TEST_FILE_SIZE).unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).unwrap();

        (fs, dir)
    }

    proptest! {
        #[test]
        fn test_random_frange_operations(
            operations in vec(0..4u8, 1..100),
            sizes in vec(1024u64..1024*1024, 1..20),
            write_positions in vec(0u64..1024*1024, 1..50),
            write_sizes in vec(128u64..4096, 1..50)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let (fs, _dir) = setup_test_fs().await;
                let mut active_franges = Vec::new();
                let mut rng = thread_rng();

                for op in operations {
                    match op {
                        // Create new frange
                        0 if !sizes.is_empty() => {
                            let size = sizes[rng.gen_range(0..sizes.len())];
                            if let Ok(id) = fs.create_frange(size) {
                                active_franges.push((id, size));
                            }
                        },
                        // Delete existing frange
                        1 if !active_franges.is_empty() => {
                            let idx = rng.gen_range(0..active_franges.len());
                            let (id, _) = active_franges.remove(idx);
                            fs.delete_frange(id).unwrap();
                        },
                        // Write to existing frange
                        2 if !active_franges.is_empty() && !write_positions.is_empty() && !write_sizes.is_empty() => {
                            let (id, max_size) = active_franges[rng.gen_range(0..active_franges.len())];
                            let pos = write_positions[rng.gen_range(0..write_positions.len())] % max_size;
                            let size = write_sizes[rng.gen_range(0..write_sizes.len())].min(max_size - pos);

                            let handle = fs.open_frange(id).unwrap();
                            let data = vec![rng.gen::<u8>(); size as usize];
                            handle.write_at(pos, &data).unwrap();
                            fs.close_frange(handle).unwrap();
                        },
                        // Read and verify
                        3 if !active_franges.is_empty() && !write_positions.is_empty() && !write_sizes.is_empty() => {
                            let (id, max_size) = active_franges[rng.gen_range(0..active_franges.len())];
                            let pos = write_positions[rng.gen_range(0..write_positions.len())] % max_size;
                            let size = write_sizes[rng.gen_range(0..write_sizes.len())].min(max_size - pos);

                            let handle = fs.open_frange(id).unwrap();
                            let mut buf = vec![0u8; size as usize];
                            handle.read_at(pos, &mut buf).unwrap();
                            fs.close_frange(handle).unwrap();
                        },
                        _ => {}
                    }
                }

                // Final verification
                for (id, _) in active_franges {
                    fs.delete_frange(id).unwrap();
                }

                // Verify filesystem is still in a consistent state
                fs.sync().unwrap();
                assert!(fs.coalesce_free_ranges().is_ok());
            });
        }

        #[test]
        fn test_frange_size_boundaries(
            size in 1024u64..1024*1024*10,
            operations in vec(0u8..4, 1..20)
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let (fs, _dir) = setup_test_fs().await;

                // Try to create a frange of the given size
                if let Ok(id) = fs.create_frange(size) {
                    let handle = fs.open_frange(id).unwrap();

                    // Test operations at boundaries
                    let data = vec![1u8; 1024];

                    // Write at start
                    handle.write_at(0, &data).unwrap();

                    // Write at end - size
                    if size >= data.len() as u64 {
                        handle.write_at(size - data.len() as u64, &data).unwrap();
                    }

                    // Verify writes beyond size fail
                    assert!(handle.write_at(size, &data).is_err());

                    fs.close_frange(handle).unwrap();
                    fs.delete_frange(id).unwrap();
                }
            });
        }
    }

    #[tokio::test]
    async fn test_stress_with_mixed_sizes() {
        let (fs, _dir) = setup_test_fs().await;
        let fs = Arc::new(fs);

        // Create mixed workload with different sizes
        let sizes = vec![
            1024,        // 1KB
            1024 * 1024, // 1MB
            1024 * 16,   // 16KB
            1024 * 256,  // 256KB
            1024 * 64,   // 64KB
        ];

        let mut handles = vec![];

        for size in sizes {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                let mut frange_ids = vec![];

                // Create multiple franges of this size
                for _ in 0..5 {
                    if let Ok(id) = fs_clone.create_frange(size) {
                        frange_ids.push(id);

                        // Write some data
                        let handle = fs_clone.open_frange(id).unwrap();
                        let data = vec![thread_rng().gen::<u8>(); (size / 2) as usize];
                        handle.write_at(0, &data).unwrap();
                        fs_clone.close_frange(handle).unwrap();
                    }
                }

                // Delete some randomly
                let mut rng = thread_rng();
                while !frange_ids.is_empty() {
                    let idx = rng.gen_range(0..frange_ids.len());
                    let id = frange_ids.swap_remove(idx);
                    fs_clone.delete_frange(id).unwrap();
                }
            }));
        }

        futures::future::join_all(handles).await;

        // Verify filesystem is still consistent
        fs.sync().unwrap();
        assert!(fs.coalesce_free_ranges().is_ok());
    }

    #[tokio::test]
    async fn test_simulated_crash_recovery() {
        let (fs, dir) = setup_test_fs().await;
        let mut frange_data = Vec::new();

        // Create some initial state
        for size in [1024, 2048, 4096] {
            let id = fs.create_frange(size).unwrap();
            let handle = fs.open_frange(id).unwrap();

            let data = vec![thread_rng().gen::<u8>(); size as usize];
            handle.write_at(0, &data).unwrap();

            fs.close_frange(handle).unwrap();
            frange_data.push((id, data));
        }

        // Force a flush
        fs.sync().unwrap();

        // Simulate crash by dropping the fs and reopening
        drop(fs);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify all data survived
        for (id, original_data) in frange_data {
            let handle = recovered_fs.open_frange(id).unwrap();
            let mut buf = vec![0u8; original_data.len()];
            handle.read_at(0, &mut buf).unwrap();
            assert_eq!(buf, original_data);
            recovered_fs.close_frange(handle).unwrap();
        }
    }

    #[tokio::test]
    async fn test_concurrent_frange_operations() {
        let (fs, _dir) = setup_test_fs().await;
        let fs = Arc::new(fs);

        let mut handles = vec![];

        // Create multiple concurrent writers
        for _ in 0..10 {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                // Create a frange
                let id = fs_clone.create_frange(1024 * 1024).unwrap(); // 1MB each

                // Open it
                let handle = fs_clone.open_frange(id).unwrap();

                // Write random data
                let mut rng = thread_rng();
                let mut data = vec![0u8; 1024 * 512]; // 512KB
                rng.fill_bytes(&mut data);

                // Write multiple times
                for offset in (0..1024 * 1024).step_by(1024 * 512) {
                    handle.write_at(offset as u64, &data).unwrap();
                }

                // Close it
                fs_clone.close_frange(handle).unwrap();

                id
            }));
        }

        // Wait for all operations to complete
        let results = futures::future::join_all(handles).await;
        let frange_ids: Vec<u64> = results.into_iter().map(|r| r.unwrap()).collect();

        // Verify all franges
        for id in frange_ids {
            let handle = fs.open_frange(id).unwrap();
            let mut buf = vec![0u8; 1024 * 512];
            handle.read_at(0, &mut buf).unwrap();
            fs.close_frange(handle).unwrap();
        }
    }

    #[tokio::test]
    async fn test_fragmentation_and_coalescing() {
        let (fs, _dir) = setup_test_fs().await;

        // Create a sequence of franges with gaps
        let mut frange_ids = vec![];
        for _ in 0..5 {
            let id = fs.create_frange(1024 * 1024).unwrap(); // 1MB each
            frange_ids.push(id);
        }

        // Delete alternate franges to create fragmentation
        for i in (0..frange_ids.len()).step_by(2) {
            fs.delete_frange(frange_ids[i]).unwrap();
        }

        // Try to create a large frange that should fit in coalesced space
        let large_id = fs.create_frange(1024 * 1024 * 2).unwrap(); // 2MB

        // Verify the large frange is usable
        let handle = fs.open_frange(large_id).unwrap();
        let data = vec![42u8; 1024 * 1024 * 2];
        handle.write_at(0, &data).unwrap();
        fs.close_frange(handle).unwrap();
    }

    #[tokio::test]
    async fn test_stress_with_many_small_franges() {
        let (fs, _dir) = setup_test_fs().await;
        let fs = Arc::new(fs);

        // Create many small franges concurrently
        let mut handles = vec![];
        for _ in 0..100 {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                let id = fs_clone.create_frange(1024).unwrap(); // 1KB each

                // Write small amounts of data repeatedly
                let handle = fs_clone.open_frange(id).unwrap();
                let data = vec![1u8; 128]; // 128 bytes
                for i in 0..8 {
                    handle.write_at(i * 128, &data).unwrap();
                }
                fs_clone.close_frange(handle).unwrap();

                // Reopen and verify
                let handle = fs_clone.open_frange(id).unwrap();
                let mut buf = vec![0u8; 128];
                handle.read_at(0, &mut buf).unwrap();
                assert_eq!(buf, data);
                fs_clone.close_frange(handle).unwrap();

                id
            }));
        }

        let results = futures::future::join_all(handles).await;
        assert_eq!(results.len(), 100);
    }

    #[tokio::test]
    async fn test_metadata_persistence() {
        let (fs, _dir) = setup_test_fs().await;

        // Create some initial state
        let id1 = fs.create_frange(1024 * 1024).unwrap();
        let id2 = fs.create_frange(1024 * 512).unwrap();

        let handle1 = fs.open_frange(id1).unwrap();
        let data1 = vec![1u8; 1024 * 1024];
        handle1.write_at(0, &data1).unwrap();
        fs.close_frange(handle1).unwrap();

        // Force a flush
        fs.sync().unwrap();

        // "Crash" and recover
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(_dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify state was recovered
        let handle1 = recovered_fs.open_frange(id1).unwrap();
        let mut buf = vec![0u8; 1024 * 1024];
        handle1.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, data1);
        recovered_fs.close_frange(handle1).unwrap();

        // Verify second frange exists
        let handle2 = recovered_fs.open_frange(id2).unwrap();
        recovered_fs.close_frange(handle2).unwrap();
    }

    #[tokio::test]
    async fn test_edge_cases() {
        let (fs, _dir) = setup_test_fs().await;

        // Calculate available space accounting for header and metadata
        let header_size = size_of::<FsHeader>();
        let metadata_size = INITIAL_METADATA_SIZE;
        let available_space = TEST_FILE_SIZE - (header_size + metadata_size) as u64;

        // Test creating a frange at the maximum available size
        let max_id = fs.create_frange(available_space).unwrap();

        // Attempt to create another frange should fail
        assert!(matches!(fs.create_frange(1024), Err(StorageExhausted)));

        // Delete the large frange
        fs.delete_frange(max_id).unwrap();

        // Should now be able to create a small frange
        let small_id = fs.create_frange(1024).unwrap();

        // Test reading/writing at frange boundaries
        let handle = fs.open_frange(small_id).unwrap();
        let data = vec![255u8; 1024];
        handle.write_at(0, &data).unwrap();

        // Reading past end should fail
        let mut buf = vec![0u8; 128];
        assert!(handle.read_at(1024, &mut buf).is_err());

        fs.close_frange(handle).unwrap();
    }
}

#[cfg(test)]
#[allow(clippy::question_mark_used)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod journal_tests {
    use super::*;
    use std::{fs::OpenOptions, sync::Arc};
    use memmap2::MmapMut;
    use tempfile::tempdir;
    use tokio::task;
    use rand::{thread_rng, RngCore};
    use futures::future::join_all;
    use crate::fs::Fs;
    use crate::fs::journal::JOURNAL_SIZE;

    const TEST_FILE_SIZE: u64 = 1024 * 1024 * 100; // 100MB for testing

    async fn setup_test_fs() -> (Arc<Fs>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .unwrap();

        file.set_len(TEST_FILE_SIZE).unwrap();
        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let fs = Fs::init(mmap).unwrap();

        (fs, dir)
    }

    #[tokio::test]
    async fn test_journal_create_frange_recovery() {
        let (fs, dir) = setup_test_fs().await;

        // Create several franges to generate journal entries
        let mut frange_data = Vec::new();
        for size in [1024, 2048, 4096] {
            let id = fs.create_frange(size).unwrap();
            let handle = fs.open_frange(id).unwrap();

            let mut data = vec![0u8; size as usize];
            thread_rng().fill_bytes(&mut data);
            handle.write_at(0, &data).unwrap();

            fs.close_frange(handle).unwrap();
            frange_data.push((id, data));
        }

        // Force sync to ensure journal is written
        fs.sync().unwrap();

        // Simulate crash by dropping the fs and reopening
        drop(fs);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify all franges and their data survived
        for (id, original_data) in frange_data {
            let handle = recovered_fs.open_frange(id).unwrap();
            let mut buf = vec![0u8; original_data.len()];
            handle.read_at(0, &mut buf).unwrap();
            assert_eq!(buf, original_data, "Data mismatch after recovery for frange {}", id);
            recovered_fs.close_frange(handle).unwrap();
        }
    }

    #[tokio::test]
    async fn test_journal_delete_frange_recovery() {
        let (fs, dir) = setup_test_fs().await;

        // Create and then delete some franges
        let mut retained_ids = Vec::new();
        let mut deleted_ids = Vec::new();

        // Create 6 franges, delete 3 of them
        for i in 0..6 {
            let id = fs.create_frange(1024).unwrap();
            if i % 2 == 0 {
                fs.delete_frange(id).unwrap();
                deleted_ids.push(id);
            } else {
                retained_ids.push(id);
            }
        }

        fs.sync().unwrap();

        // Simulate crash and recover
        drop(fs);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify deleted franges are still deleted
        for id in deleted_ids {
            assert!(recovered_fs.open_frange(id).is_err(), "Deleted frange {} still exists", id);
        }

        // Verify retained franges still exist
        for id in retained_ids {
            assert!(recovered_fs.open_frange(id).is_ok(), "Retained frange {} doesn't exist", id);
        }
    }

    #[tokio::test]
    async fn test_journal_update_frange_recovery() {
        let (fs, dir) = setup_test_fs().await;

        // Create a frange and perform multiple updates
        let id = fs.create_frange(4096).unwrap();
        let updates = vec![
            (0, vec![1u8; 1024]),
            (1024, vec![2u8; 1024]),
            (2048, vec![3u8; 1024]),
            (3072, vec![4u8; 1024]),
        ];

        {
            let handle = fs.open_frange(id).unwrap();
            for (offset, data) in &updates {
                handle.write_at(*offset, data).unwrap();
            }
            fs.close_frange(handle).unwrap();
        }

        fs.sync().unwrap();

        // Simulate crash and recover
        drop(fs);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify all updates persisted
        let handle = recovered_fs.open_frange(id).unwrap();
        for (offset, expected_data) in updates {
            let mut buf = vec![0u8; expected_data.len()];
            handle.read_at(offset, &mut buf).unwrap();
            assert_eq!(buf, expected_data, "Data mismatch at offset {}", offset);
        }
        recovered_fs.close_frange(handle).unwrap();
    }

    #[tokio::test]
    async fn test_journal_coalesce_recovery() {
        let (fs, dir) = setup_test_fs().await;

        // Create a fragmented state
        let mut ids = Vec::new();
        for _ in 0..5 {
            ids.push(fs.create_frange(1024).unwrap());
        }

        // Delete alternate franges to create fragmentation
        for i in (0..ids.len()).step_by(2) {
            fs.delete_frange(ids[i]).unwrap();
        }

        // Force coalesce and sync
        fs.coalesce_free_ranges().unwrap();
        fs.sync().unwrap();

        // Try to create a large frange that should fit in coalesced space
        let large_id = fs.create_frange(2048).unwrap();

        // Simulate crash and recover
        drop(fs);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.path().join("test.db"))
            .unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
        let recovered_fs = Fs::new(mmap).unwrap();

        // Verify the large frange is still valid and usable
        let handle = recovered_fs.open_frange(large_id).unwrap();
        let data = vec![42u8; 2048];
        handle.write_at(0, &data).unwrap();
        recovered_fs.close_frange(handle).unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_journal_operations() {
        let (fs, _) = setup_test_fs().await;
        let fs = Arc::new(fs);

        let mut handles = vec![];

        // Spawn multiple tasks performing different journal-generating operations
        for i in 0..10 {
            let fs_clone = fs.clone();
            handles.push(task::spawn(async move {
                let mut rng = thread_rng();
                let id = fs_clone.create_frange(1024).unwrap();

                // Perform multiple operations
                let handle = fs_clone.open_frange(id).unwrap();
                let data = vec![i as u8; 512];
                handle.write_at(0, &data).unwrap();

                // Random additional write
                let offset = rng.next_u32() % 512;
                handle.write_at(offset as u64, &data[..256]).unwrap();

                fs_clone.close_frange(handle).unwrap();

                // 50% chance to delete the frange
                if rng.next_u32() % 2 == 0 {
                    fs_clone.delete_frange(id).unwrap();
                }

                id
            }));
        }

        // Wait for all operations to complete
        let results = join_all(handles).await;

        // Force sync to ensure journal is written
        fs.sync().unwrap();

        // Verify filesystem is in a consistent state
        for result in results {
            let id = result.unwrap();
            match fs.open_frange(id) {
                Ok(handle) => {
                    // If frange exists, it should be readable
                    let mut buf = vec![0u8; 512];
                    assert!(handle.read_at(0, &mut buf).is_ok());
                    fs.close_frange(handle).unwrap();
                },
                Err(_) => {
                    // Frange was deleted, which is also valid
                }
            }
        }
    }

    #[tokio::test]
    async fn test_journal_size_limits() {
        let (fs, _) = setup_test_fs().await;

        // Create enough entries to fill the journal
        let journal_size = JOURNAL_SIZE;
        let mut created_ids = Vec::new();

        // Keep creating and deleting franges until we've generated more journal entries than would fit
        for i in 0..((journal_size / 100) + 1) {
            let id = fs.create_frange(1024).unwrap();
            created_ids.push(id);

            // Every 10 creations, delete some to generate delete entries
            if i % 10 == 0 && !created_ids.is_empty() {
                let delete_id = created_ids.remove(0);
                fs.delete_frange(delete_id).unwrap();
            }
        }

        // The filesystem should still be operational
        let test_id = fs.create_frange(1024).unwrap();
        let handle = fs.open_frange(test_id).unwrap();
        let test_data = vec![1u8; 1024];
        handle.write_at(0, &test_data).unwrap();
        fs.close_frange(handle).unwrap();

        // Verify the test frange is readable
        let handle = fs.open_frange(test_id).unwrap();
        let mut buf = vec![0u8; 1024];
        handle.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, test_data);
        fs.close_frange(handle).unwrap();
    }
}