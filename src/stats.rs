use std::sync::atomic::AtomicUsize;
use std::sync::LazyLock;

pub(crate) static STATS: LazyLock<Stats> = LazyLock::new(|| {
    Stats::default()
});

#[derive(Debug, Default)]
pub(crate) struct Stats {
    pub(crate) current_threads: AtomicUsize,
}