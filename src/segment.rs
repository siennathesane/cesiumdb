use std::mem::zeroed;

/// Each segment maps directly to a block size.
pub(crate) const SEGMENT_SIZE: usize = 4096;

#[derive(Debug, Copy, Clone)]
pub(crate) struct Segment<T> {
    pub(crate) data: [T; SEGMENT_SIZE],
    pub(crate) len: usize,
}

impl<T> Default for Segment<T>
where
    T: Copy,
{
    fn default() -> Self {
        Segment {
            data: [unsafe { zeroed() }; SEGMENT_SIZE],
            len: 0,
        }
    }
}
