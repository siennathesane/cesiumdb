use std::{
    cell::SyncUnsafeCell,
    fs::{
        File,
        OpenOptions,
    },
    ops::{
        Deref,
        DerefMut,
        Index,
        IndexMut,
        Range,
    },
    path::PathBuf,
    ptr,
    sync::{
        Arc,
        atomic::{
            AtomicPtr,
            AtomicU64,
            Ordering::{
                AcqRel,
                Acquire,
                Relaxed,
                Release,
            },
        },
    },
};

use memmap2::{
    Advice::WillNeed,
    MmapMut,
};
use parking_lot::Mutex;

use crate::errs::{
    SegmentError,
    SegmentError::IoError,
};

pub struct Map {
    inner: AtomicPtr<SyncUnsafeCell<MmapMut>>,
    file: Mutex<File>,
    current_offset: AtomicU64,
    resize_lock: Mutex<()>,
}

impl Map {
    pub fn new(path: PathBuf, initial_size: u64) -> Result<Self, SegmentError> {
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path.clone())
        {
            | Ok(v) => v,
            | Err(e) => return Err(IoError(e)),
        };

        match file.set_len(initial_size) {
            | Ok(_) => {},
            | Err(e) => return Err(IoError(e)),
        };
        // SAFETY: none, this is an unsafe operation
        let mmap = unsafe {
            match MmapMut::map_mut(&file) {
                | Ok(v) => v,
                | Err(e) => return Err(IoError(e)),
            }
        };

        Ok(Self {
            inner: AtomicPtr::new(Box::into_raw(Box::new(SyncUnsafeCell::new(mmap)))),
            file: Mutex::new(file),
            current_offset: AtomicU64::new(0),
            resize_lock: Mutex::new(()),
        })
    }

    pub fn grow(&self, new_size: u64) -> Result<(), SegmentError> {
        let _guard = self.resize_lock.lock();

        {
            let file = self.file.lock();
            match file.set_len(new_size) {
                | Ok(_) => {},
                | Err(e) => return Err(IoError(e)),
            };
        }

        let new_mmap = {
            let file = self.file.lock();

            // SAFETY: none, this is an unsafe operation
            SyncUnsafeCell::new(unsafe {
                match MmapMut::map_mut(&*file) {
                    | Ok(v) => v,
                    | Err(e) => return Err(IoError(e)),
                }
            })
        };

        // swap the pointers and remove the old one to prevent use after free
        let old_ptr = self.inner.swap(Box::into_raw(Box::new(new_mmap)), AcqRel);
        // SAFETY: this is just a swap
        unsafe {
            drop(Box::from_raw(old_ptr));
        }

        Ok(())
    }

    /// Write to a specific range.
    pub fn write_to_range(
        &self,
        range: Range<usize>,
        writer: impl FnOnce(&mut [u8]),
    ) -> Result<(), SegmentError> {
        let len = self.len();
        let end = range.end;

        // Grow the map if needed
        if end > len {
            match self.grow(end as u64) {
                | Ok(_) => {},
                | Err(e) => return Err(e),
            };
        }

        // now get the reference from the potentially new mapping
        let ptr = self.inner.load(Acquire);

        // SAFETY: We're ensuring synchronized access through the Map's methods
        unsafe {
            let mmap = &*ptr;
            let inner = &mut *mmap.get();
            let slice = &mut inner[range];
            writer(slice);
        }

        Ok(())
    }

    #[inline]
    pub fn warn(&self, range: Range<usize>) {
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, this is an unsafe operation as we are dereferencing a pointer
        unsafe {
            let mmap = &*ptr;
            let inner = &*mmap.get();
            &inner.advise_range(WillNeed, range.start, range.end - range.start);
        }
    }

    pub fn len(&self) -> usize {
        // SAFETY: none, this is an unsafe operation
        unsafe {
            let ptr = self.inner.load(Acquire);
            // please don't ask me to explain this
            (*(*ptr).get()).len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Index<Range<usize>> for Map {
    type Output = [u8];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, this is an unsafe operation as we are dereferencing a pointer
        unsafe {
            let mmap = &*ptr;
            let inner = &*mmap.get();
            &inner[index]
        }
    }
}

impl IndexMut<Range<usize>> for Map {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        // First check if we need to grow
        {
            let ptr = self.inner.load(Acquire);
            let len = self.len();
            if index.end > len {
                match self.grow(index.end as u64) {
                    | Ok(_) => {},
                    // TODO(@siennathesane): handle this error *somehow*
                    | Err(e) => panic!("failed to grow map: {:?}", e),
                };
            }
        }

        // Now get the reference from the potentially new mapping
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, we are mutably dereferencing a pointer
        unsafe {
            let mmap = &*ptr;
            let inner = &mut *mmap.get();
            &mut inner[index]
        }
    }
}

impl Deref for Map {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, we are dereferencing a pointer
        unsafe {
            let mmap = &*ptr;
            let inner = &*mmap.get();
            inner
        }
    }
}

impl DerefMut for Map {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = self.inner.load(Acquire);
        // SAFETY: none, we are mutably dereferencing a pointer
        unsafe {
            let mmap = &mut *ptr;
            let inner = &mut *mmap.get();
            inner
        }
    }
}

/// SAFETY: trait impl
unsafe impl Send for Map {}

/// SAFETY: trail impl
unsafe impl Sync for Map {}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        path::Path,
    };

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_new_map() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_new.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        assert_eq!(map.len(), initial_size as usize);
        assert!(!map.is_empty());
        assert_eq!(map.current_offset.load(Relaxed), 0);
    }

    #[test]
    fn test_grow_map() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_grow.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        assert_eq!(map.len(), initial_size as usize);

        let new_size = 2048;
        map.grow(new_size).unwrap();

        assert_eq!(map.len(), new_size as usize);
    }

    #[test]
    fn test_index_range() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_index.segment");

        let initial_size = 1024;
        let map = Map::new(file_path, initial_size).unwrap();

        // Get a range from the map
        let range = 0..10;
        let slice = &map[range.clone()];

        assert_eq!(slice.len(), 10);
        // New mmap should be zeroed
        assert_eq!(slice, &[0u8; 10]);
    }

    #[test]
    fn test_index_mut_range() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_index_mut.segment");

        let initial_size = 1024;
        let mut map = Map::new(file_path, initial_size).unwrap();

        // Modify a range in the map
        let range = 0..10;
        let slice = &mut map[range.clone()];

        for i in 0..10 {
            slice[i] = i as u8;
        }

        // Check that the changes persisted
        let check_slice = &map[range];
        for i in 0..10 {
            assert_eq!(check_slice[i], i as u8);
        }
    }

    #[test]
    fn test_index_mut_grow() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_index_mut_grow.segment");

        let initial_size = 100;
        let mut map = Map::new(file_path, initial_size).unwrap();

        assert_eq!(map.len(), initial_size as usize);

        // Access beyond the current size to trigger grow
        let range = 100..200;
        let slice = &mut map[range.clone()];

        for i in 0..100 {
            slice[i] = (i + 100) as u8;
        }

        assert!(map.len() >= 200);

        // Check that the changes persisted
        let check_slice = &map[range];
        for i in 0..100 {
            assert_eq!(check_slice[i], (i + 100) as u8);
        }
    }

    #[test]
    fn test_deref() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_deref.segment");

        let initial_size = 10;
        let map = Map::new(file_path, initial_size).unwrap();

        // Use deref to access the entire map as a slice
        let slice: &[u8] = &map;

        assert_eq!(slice.len(), initial_size as usize);
        // New mmap should be zeroed
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_deref_mut() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_deref_mut.segment");

        let initial_size = 10;
        let mut map = Map::new(file_path, initial_size).unwrap();

        // Use deref_mut to access the entire map as a mutable slice
        let slice: &mut [u8] = &mut map;

        // Fill with a pattern
        for i in 0..slice.len() {
            slice[i] = (i * 2) as u8;
        }

        // Check that the changes persisted via a different access method
        for i in 0..initial_size as usize {
            assert_eq!(map[i..i + 1][0], (i * 2) as u8);
        }
    }

    #[test]
    fn test_concurrent_access() {
        use std::{
            sync::Arc,
            thread,
        };

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_concurrent.segment");

        let initial_size = 1024;
        let map = Arc::new(Map::new(file_path, initial_size).unwrap());

        let mut handles = vec![];

        // Spawn multiple threads to read from the map concurrently
        for i in 0..4 {
            let map_clone = Arc::clone(&map);
            let handle = thread::spawn(move || {
                let offset = i * 10;
                let range = offset..offset + 10;
                let slice = &map_clone[range];
                assert_eq!(slice.len(), 10);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_is_empty() {
        let dir = tempdir().unwrap();

        // Test with zero size
        let file_path = dir.path().join("test_empty.segment");
        let map = Map::new(file_path, 0).unwrap();
        assert!(map.is_empty());

        // Test with non-zero size
        let file_path = dir.path().join("test_not_empty.segment");
        let map = Map::new(file_path, 1).unwrap();
        assert!(!map.is_empty());
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_persistence.segment");

        // First, create and modify the map
        {
            let mut map = Map::new(file_path.clone(), 100).unwrap();
            let slice = &mut map[0..10];
            for i in 0..10 {
                slice[i] = (i + 1) as u8;
            }
            // map will be dropped here, which should flush changes
        }

        // Now open the same file again and verify data persisted
        {
            let map = Map::new(file_path, 100).unwrap();
            let slice = &map[0..10];
            for i in 0..10 {
                assert_eq!(slice[i], (i + 1) as u8);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Permission denied")]
    fn test_index_mut_grow_error() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_grow_error.segment");

        // Create a read-only directory to force a permission error on grow
        let readonly_dir = dir.path().join("readonly");
        std::fs::create_dir(&readonly_dir).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(&readonly_dir).unwrap();
            let mut perms = metadata.permissions();
            perms.set_mode(0o555); // read and execute only
            std::fs::set_permissions(&readonly_dir, perms).unwrap();
        }

        let ro_file_path = readonly_dir.join("readonly.segment");

        // Create the file first since we can't create it in read-only dir
        {
            File::create(&ro_file_path).unwrap();

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let metadata = std::fs::metadata(&ro_file_path).unwrap();
                let mut perms = metadata.permissions();
                perms.set_mode(0o444); // read only
                std::fs::set_permissions(&ro_file_path, perms).unwrap();
            }
        }

        let mut map = Map::new(ro_file_path, 10).unwrap();

        // This should panic due to permission error when trying to grow
        let _slice = &mut map[10..20];
    }
}
