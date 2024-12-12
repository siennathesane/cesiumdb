// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#[cfg(target_arch = "aarch64")]
use std::sync::atomic::AtomicU128 as StdAtomicU128;
use std::{
    sync::{
        atomic::{
            AtomicBool,
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

pub struct HybridLogicalClock {
    #[cfg(target_arch = "aarch64")]
    last_tick: Arc<StdAtomicU128>,
    #[cfg(target_arch = "x86_64")]
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
        #[cfg(target_arch = "aarch64")]
        let last_tick = Arc::new(StdAtomicU128::new(now));
        #[cfg(target_arch = "x86_64")]
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

use std::sync::atomic::{
    AtomicU64,
    Ordering,
};

#[cfg(target_arch = "x86_64")]
#[repr(align(16))]
pub struct AtomicU128 {
    lo: AtomicU64,
    hi: AtomicU64,
}

#[cfg(target_arch = "x86_64")]
impl AtomicU128 {
    pub const fn new(value: u128) -> Self {
        Self {
            lo: AtomicU64::new(value as u64),
            hi: AtomicU64::new((value >> 64) as u64),
        }
    }

    pub fn load(&self, order: Ordering) -> u128 {
        // We need to be careful about the ordering here to prevent torn reads
        let hi = self.hi.load(order);
        let lo = self.lo.load(order);
        ((hi as u128) << 64) | (lo as u128)
    }

    pub fn store(&self, value: u128, order: Ordering) {
        self.hi.store((value >> 64) as u64, order);
        self.lo.store(value as u64, order);
    }

    pub fn compare_exchange(
        &self,
        current: u128,
        new: u128,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u128, u128> {
        let current_hi = (current >> 64) as u64;
        let current_lo = current as u64;
        let new_hi = (new >> 64) as u64;
        let new_lo = new as u64;

        // First try to CAS the high bits
        match self
            .hi
            .compare_exchange(current_hi, new_hi, success, failure)
        {
            | Ok(_) => {
                // High bits matched, now try low bits
                match self
                    .lo
                    .compare_exchange(current_lo, new_lo, success, failure)
                {
                    | Ok(_) => Ok(current),
                    | Err(actual_lo) => {
                        // Low bits failed, restore high bits
                        self.hi.store(current_hi, Ordering::Release);
                        Err(((current_hi as u128) << 64) | (actual_lo as u128))
                    },
                }
            },
            | Err(actual_hi) => {
                // High bits didn't match
                Err(((actual_hi as u128) << 64) | (self.lo.load(failure) as u128))
            },
        }
    }

    pub fn fetch_add(&self, val: u128, order: Ordering) -> u128 {
        loop {
            let current = self.load(Ordering::Relaxed);
            if let Ok(old) =
                self.compare_exchange(current, current.wrapping_add(val), order, Ordering::Relaxed)
            {
                return old;
            }
        }
    }

    pub fn fetch_sub(&self, val: u128, order: Ordering) -> u128 {
        self.fetch_add(val.wrapping_neg(), order)
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod x86_atomic_tests {
    use std::{
        sync::Arc,
        thread,
        time::Duration,
    };

    use super::*;

    #[test]
    fn test_basic_operations() {
        let atomic = AtomicU128::new(0);

        // Test store and load
        atomic.store(u128::MAX, Ordering::SeqCst);
        assert_eq!(atomic.load(Ordering::SeqCst), u128::MAX);

        // Test compare_exchange
        assert_eq!(
            atomic.compare_exchange(u128::MAX, 42, Ordering::SeqCst, Ordering::SeqCst),
            Ok(u128::MAX)
        );
        assert_eq!(atomic.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_edge_values() {
        let atomic = AtomicU128::new(0);
        let test_values = [
            0u128,
            1u128,
            u128::MAX,
            u128::MAX - 1,
            1u128 << 63,
            (1u128 << 64) - 1,
            1u128 << 64,
            (1u128 << 64) + 1,
            1u128 << 127,
        ];

        for &value in &test_values {
            atomic.store(value, Ordering::SeqCst);
            assert_eq!(
                atomic.load(Ordering::SeqCst),
                value,
                "Failed on value: {}",
                value
            );
        }
    }

    #[test]
    fn test_wrapping_behavior() {
        let atomic = AtomicU128::new(u128::MAX);

        // Test wrapping add
        assert_eq!(atomic.fetch_add(1, Ordering::SeqCst), u128::MAX);
        assert_eq!(atomic.load(Ordering::SeqCst), 0);

        // Test wrapping sub
        assert_eq!(atomic.fetch_sub(1, Ordering::SeqCst), 0);
        assert_eq!(atomic.load(Ordering::SeqCst), u128::MAX);
    }

    #[test]
    fn test_compare_exchange_failure() {
        let atomic = AtomicU128::new(0);

        // Expected failure
        let res = atomic.compare_exchange(42, 100, Ordering::SeqCst, Ordering::SeqCst);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), 0);

        // Multiple attempts with different values
        let mut success = false;
        for i in 0..10 {
            match atomic.compare_exchange(0, i, Ordering::SeqCst, Ordering::SeqCst) {
                | Ok(_) => {
                    success = true;
                    break;
                },
                | Err(_) => continue,
            }
        }
        assert!(success, "Compare exchange should succeed at least once");
    }

    #[test]
    fn test_concurrent_increments() {
        let atomic = Arc::new(AtomicU128::new(0));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let atomic = Arc::clone(&atomic);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        atomic.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(atomic.load(Ordering::SeqCst), 4000);
    }

    #[test]
    fn test_concurrent_mixed_operations() {
        let atomic = Arc::new(AtomicU128::new(1000));
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let atomic = Arc::clone(&atomic);
                thread::spawn(move || {
                    for _ in 0..100 {
                        match i % 4 {
                            | 0 => {
                                atomic.fetch_add(2, Ordering::SeqCst);
                            },
                            | 1 => {
                                atomic.fetch_sub(1, Ordering::SeqCst);
                            },
                            | 2 => {
                                let current = atomic.load(Ordering::SeqCst);
                                let _ = atomic.compare_exchange(
                                    current,
                                    current + 1,
                                    Ordering::SeqCst,
                                    Ordering::SeqCst,
                                );
                            },
                            | _ => {
                                atomic.store(atomic.load(Ordering::SeqCst) + 1, Ordering::SeqCst);
                            },
                        }
                        thread::sleep(Duration::from_nanos(1));
                    }
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        let final_value = atomic.load(Ordering::SeqCst);
        assert!(
            final_value > 1000,
            "Value should have increased from concurrent operations"
        );
    }

    #[test]
    fn test_ordering_combinations() {
        let atomic = AtomicU128::new(0);

        // Valid store orderings
        let store_orderings = [
            Ordering::SeqCst,
            Ordering::Release,
            Ordering::Relaxed,
        ];

        // Valid load orderings
        let load_orderings = [
            Ordering::SeqCst,
            Ordering::Acquire,
            Ordering::Relaxed,
        ];

        for &store_order in &store_orderings {
            for &load_order in &load_orderings {
                atomic.store(42, store_order);
                assert_eq!(atomic.load(load_order), 42);
            }
        }

        // Test compare_exchange with valid ordering combinations
        let success_orderings = [
            Ordering::SeqCst,
            Ordering::AcqRel,
            Ordering::Acquire,
            Ordering::Release,
            Ordering::Relaxed,
        ];

        // Failure ordering must be no stronger than success and cannot be Release or AcqRel
        let failure_orderings = [
            Ordering::SeqCst,
            Ordering::Acquire,
            Ordering::Relaxed,
        ];

        for &success_order in &success_orderings {
            for &failure_order in &failure_orderings {
                // Skip invalid combinations where failure is stronger than success
                if (failure_order == Ordering::SeqCst && success_order != Ordering::SeqCst) {
                    continue;
                }

                let _ = atomic.compare_exchange(
                    42,
                    100,
                    success_order,
                    failure_order,
                );
            }
        }
    }

    #[test]
    fn test_concurrent_stress() {
        let atomic = Arc::new(AtomicU128::new(0));
        let thread_count = 16;
        let iterations = 10_000;

        let threads: Vec<_> = (0..thread_count)
            .map(|id| {
                let atomic = Arc::clone(&atomic);
                thread::spawn(move || {
                    let mut local_sum = 0u128;
                    for i in 0..iterations {
                        let value = i as u128 + id as u128;
                        let old = atomic.fetch_add(value, Ordering::SeqCst);
                        local_sum = local_sum.wrapping_add(old);
                    }
                    local_sum
                })
            })
            .collect();

        let mut total_sum = 0u128;
        for thread in threads {
            total_sum = total_sum.wrapping_add(thread.join().unwrap());
        }

        let final_value = atomic.load(Ordering::SeqCst);
        assert!(
            final_value > 0,
            "Final value should be non-zero after stress test"
        );
    }
}
