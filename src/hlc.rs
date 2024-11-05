// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::{
    sync::{
        atomic::{
            AtomicBool,
            AtomicU128,
            Ordering::Relaxed,
        },
        Arc,
    },
    thread,
    time::{
        Duration,
        SystemTime,
    },
};

use crate::stats::STATS;

pub trait HLC: Send + Sync {
    fn time(&self) -> u128;
}

/// How often the clock is synchronized with the source.
pub const TICK_FREQUENCY_IN_NS: u64 = 500;

#[repr(C)]
pub struct HybridLogicalClock {
    last_tick: Arc<AtomicU128>,
    done: Arc<AtomicBool>,
}

#[allow(clippy::new_without_default)]
impl HybridLogicalClock {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let last_tick = Arc::new(AtomicU128::new(now));

        let done = Arc::new(AtomicBool::new(false));

        let last_tick_clone = last_tick.clone();
        let done_clone = done.clone();
        thread::spawn(move || {
            while !done_clone.load(Relaxed) {
                thread::sleep(Duration::from_nanos(TICK_FREQUENCY_IN_NS));
                let now = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let diff = now - last_tick_clone.load(Relaxed);
                if diff == 0 {
                    continue;
                }
                last_tick_clone.fetch_add(diff, Relaxed);
            }
            STATS.current_threads.fetch_sub(1, Relaxed);
        });
        STATS.current_threads.fetch_add(1, Relaxed);

        Self { last_tick, done }
    }
}

impl HLC for HybridLogicalClock {
    #[inline]
    fn time(&self) -> u128 {
        self.last_tick
            .store(self.last_tick.load(Relaxed) + 1, Relaxed);
        self.last_tick.load(Relaxed)
    }
}

impl Drop for HybridLogicalClock {
    fn drop(&mut self) {
        self.done.store(true, Relaxed);
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use crate::hlc::{
        HybridLogicalClock,
        HLC,
    };

    #[test]
    fn test_time() {
        let clock = HybridLogicalClock::new();
        let mut last_time = 0;
        for _ in 0..100 {
            let now = clock.time();
            assert_ne!(now, 0, "clock must never be zero");
            assert!(now > last_time, "now must be greater than last");
            last_time = now;
        }
    }
}
