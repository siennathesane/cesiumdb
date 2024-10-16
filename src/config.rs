/// The default size of L0 is 1MiB to ensure that default compilations work as
/// expected.
pub const DEFAULT_L0_SIZE: usize = 2 << 19;

pub const SEGMENTS_PER_LEVEL: usize = 16;

pub struct Config {
    l0_size: usize,
}
