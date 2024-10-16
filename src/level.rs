use crate::segment::Segment;

pub const SEGMENTS_PER_LEVEL: usize = 16;

pub(crate) struct Level<T> {
    pub(crate) segments: [Segment<T>; SEGMENTS_PER_LEVEL],
    pub(crate) len: usize,
}

impl<T> Default for Level<T>
where
    T: Copy,
{
    fn default() -> Self {
        Level {
            segments: [Segment::default(); SEGMENTS_PER_LEVEL],
            len: 0,
        }
    }
}
