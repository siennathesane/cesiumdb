mod core;
mod compaction;
mod handle;

macro_rules! bench {
    () => {
        #[cfg(feature = "benchmarks")]
        pub
        #[cfg(not(feature = "benchmarks"))]
        pub(in crate::fs)
    }
}
