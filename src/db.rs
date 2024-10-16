use std::{
    fs::{
        File,
        OpenOptions,
    },
    io,
    marker::PhantomData,
    ptr::copy_nonoverlapping,
};

use memmap2::{
    MmapMut,
    MmapOptions,
};
use thiserror::Error;

use crate::{
    level::{
        Level,
        SEGMENTS_PER_LEVEL,
    },
    segment::SEGMENT_SIZE,
};
use crate::db::CesiumError::NeedsResize;

#[derive(Error, Debug)]
pub enum DbError {
    #[error(transparent)]
    IoError(io::Error),
    #[error("cesium error")]
    CesiumError(CesiumError),
}

#[derive(Error, Debug)]
pub enum CesiumError {
    #[error("segment needs resizing")]
    NeedsResize,
}

pub(crate) struct Db<T: Ord + Copy> {
    file: File,
    mmap: MmapMut,
    levels: Vec<Level<T>>,
    total_len: usize,
    _marker: PhantomData<T>,
}

impl<T: Ord + Copy> Db<T> {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        initial_capacity: usize,
    ) -> Result<Self, DbError> {
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
        {
            | Ok(file) => file,
            | Err(e) => return Err(DbError::IoError(e)),
        };

        let cap = initial_capacity.max(SEGMENT_SIZE);
        match file.set_len((cap * size_of::<T>()) as u64) {
            | Ok(_) => {}
            | Err(e) => return Err(DbError::IoError(e)),
        };

        let mmap = unsafe {
            match MmapOptions::new().map_mut(&file) {
                | Ok(v) => v,
                | Err(e) => return Err(DbError::IoError(e))
            }
        };

        let mut levels = Vec::with_capacity(initial_capacity);
        levels.push(Level::default());

        Ok(Self {
            file,
            mmap,
            levels,
            total_len: 0,
            _marker: PhantomData,
        })
    }


    pub fn insert(&mut self, data: T) -> Result<(), DbError> {
        let mut level = 0;
        let mut insert_val = data;

        loop {
            if level == self.levels.len() {
                self.levels.push(Level::default());
            }

            let current_level = &mut self.levels[level];
            if current_level.len < SEGMENTS_PER_LEVEL * SEGMENT_SIZE {
                let seg_idx = current_level.len / SEGMENT_SIZE;
                let segment = &mut current_level.segments[seg_idx];

                if segment.len < SEGMENT_SIZE {
                    let insert_idx = segment.data[..segment.len]
                        .binary_search(&insert_val)
                        .unwrap_or_else(|e| e);

                    segment
                        .data
                        .copy_within(insert_idx..segment.len, insert_idx + 1);
                    segment.data[insert_idx] = insert_val;
                    segment.len += 1;
                    current_level.len += 1;
                    break;
                } else {
                    // segment is full, push up to the next level
                    let median = segment.data[SEGMENT_SIZE / 2];
                    insert_val = median;
                    level += 1;
                    continue;
                }
            } else {
                level += 1;
                continue;
            }
        }

        self.total_len += 1;
        Ok(())
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.total_len {
            return None;
        };

        let mut current_idx = 0;
        for level in &self.levels {
            for segment in &level.segments {
                if current_idx + segment.len > index {
                    return Some(&segment.data[index - current_idx]);
                }
                current_idx += segment.len
            }
        }

        // this shouldn't happen
        None
    }

    pub fn delete(&mut self, value: &T) -> Result<bool, DbError> {
        let mut deleted = false;
        let mut rebalance_needed = false;
        let mut level_to_update = None;

        // start from the lowest level
        for level in (0..self.levels.len()).rev() {
            let mut segment_len_decrease = 0;
            {
                let level_ref = &mut self.levels[level];
                for (seg_idx, segment) in level_ref.segments.iter_mut().enumerate() {
                    if let Ok(index) = segment.data[..segment.len].binary_search(value) {
                        // Found the value, remove it
                        segment.data.copy_within(index + 1..segment.len, index);
                        segment.len -= 1;
                        segment_len_decrease = 1;
                        self.total_len -= 1;
                        deleted = true;

                        // Check if rebalancing is needed
                        if segment.len < SEGMENT_SIZE / 2 {
                            rebalance_needed = true;
                        }

                        level_to_update = Some((level, seg_idx));
                        break; // Exit the segment loop
                    }
                }
            }

            if deleted {
                // Update the level length outside of the mutable borrow
                if let Some((level, _)) = level_to_update {
                    self.levels[level].len -= segment_len_decrease;
                }
                break; // Exit the level loop
            }
        }

        if rebalance_needed {
            self.rebalance()?;
        }

        self.sync()?;
        Ok(deleted)
    }

    pub fn scan<F>(&self, mut f: F)
    where
        F: FnMut(&T),
    {
        for level in self.levels.iter() {
            for segment in level.segments.iter() {
                for item in segment.data.iter().take(segment.len) {
                    f(item)
                }
            }
        }
    }

    pub fn sync(&mut self) -> Result<(), DbError> {
        let total_size = self.calculate_total_size();

        // Check if we need to grow the mmap
        if total_size > self.mmap.len() {
            self.grow(total_size)?;
        }

        let mut offset = 0;
        for level in &self.levels {
            for segment in &level.segments {
                let size = segment.len * size_of::<T>();
                if offset + size > self.mmap.len() {
                    return Err(DbError::CesiumError(NeedsResize));
                }
                unsafe {
                    let src = segment.data.as_ptr() as *const u8;
                    let dst = self.mmap.as_mut_ptr().add(offset);
                    copy_nonoverlapping(src, dst, size);
                }
                offset += size;
            }
        }

        self.mmap.flush().map_err(DbError::IoError)
    }

    pub fn len(&self) -> usize {
        self.total_len
    }
    
    fn rebalance(&mut self) -> Result<(), DbError> {
        let mut all_items = Vec::new();
        for level in self.levels.iter() {
            for segment in &level.segments {
                all_items.extend_from_slice(&segment.data[..segment.len]);
            }
        }

        all_items.sort_unstable();

        self.levels.clear();

        for item in all_items {
            self.insert(item)?;
        }

        Ok(())
    }

    fn grow(&mut self, required_size: usize) -> Result<(), DbError> {
        let mut new_size = self.mmap.len();
        while new_size < required_size {
            new_size *= 2;
        }

        // Grow the file
        self.file.set_len(new_size as u64).map_err(DbError::IoError)?;
        self.file.sync_all().map_err(DbError::IoError)?;

        // Remap the file
        self.mmap = unsafe {
            MmapOptions::new()
                .map_mut(&self.file)
                .map_err(DbError::IoError)?
        };

        Ok(())
    }

    fn calculate_total_size(&self) -> usize {
        self.levels.iter().fold(0, |acc, level| {
            acc + level.segments.iter().fold(0, |acc, segment| {
                acc + segment.len * std::mem::size_of::<T>()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;
    use super::*;

    #[test]
    fn lifecycle_test() -> Result<(), DbError> {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let filepath = temp_dir.path().join("test.lifecycle.db");
        let mut db: Db<usize> =
            match Db::new(filepath, 10) {
                | Ok(db) => db,
                | Err(e) => return Err(e),
            };

        for i in 0..25 {
            match db.insert(i) {
                | Ok(_) => {}
                | Err(e) => return Err(e),
            };
        }

        assert_eq!(db.len(), 25);
        
        // first open past the regrowth
        let val = db.get(11);
        assert!(val.is_some());
        let val = val.unwrap();
        assert_eq!(11, *val, "the value should be 11");

        for i in 0..25 {
            match db.delete(&(i as usize)) {
                | Ok(_) => {}
                | Err(e) => return Err(e),
            };
        }
        
        let val = db.get(11);
        assert!(val.is_none());

        Ok(())
    }

    #[test]
    fn big_lifecycle_test() -> Result<(), DbError> {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let filepath = temp_dir.path().join("test.lifecycle.db");
        let mut db: Db<usize> =
            match Db::new(filepath, 10) {
                | Ok(db) => db,
                | Err(e) => return Err(e),
            };

        for i in 0..1_000_000 {
            match db.insert(i) {
                | Ok(_) => {}
                | Err(e) => return Err(e),
            };
        }

        assert_eq!(db.len(), 1_000_000);

        for i in 0..1_000 {
            match db.delete(&(i as usize)) {
                | Ok(_) => {}
                | Err(e) => panic!("{}", e),
            };
        }

        Ok(())
    }
}
