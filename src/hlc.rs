// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#[cfg(target_arch = "aarch64")]
use std::sync::atomic::AtomicU128 as StdAtomicU128;
use std::{
    sync::{
        Arc,
        atomic::{
            AtomicBool,
            Ordering::Relaxed,
        },
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
    #[cfg(not(loom))]
    use std::{sync::Arc, thread};

    #[cfg(loom)]
    use loom::{sync::Arc, thread};

    use crate::hlc::{
        HLC,
        HybridLogicalClock,
    };

    #[test]
    #[cfg(not(loom))]
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

    #[test]
    #[cfg(not(loom))]
    fn test_concurrent_time_calls() {
        let clock = Arc::new(HybridLogicalClock::new());
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let clock = clock.clone();
                thread::spawn(move || {
                    let mut times = Vec::new();
                    for _ in 0..100 {
                        times.push(clock.time());
                    }
                    times
                })
            })
            .collect();

        let mut all_times = Vec::new();
        for thread in threads {
            all_times.extend(thread.join().unwrap());
        }

        // Verify no duplicates (all timestamps are unique)
        all_times.sort();
        for window in all_times.windows(2) {
            assert_ne!(window[0], window[1], "Timestamps must be unique");
        }
    }

    // Loom tests for HLC concurrent access

    #[test]
    #[cfg(loom)]
    fn loom_hlc_concurrent_time_monotonic() {
        loom::model(|| {
            let clock = Arc::new(HybridLogicalClock::new());

            let c1 = clock.clone();
            let c2 = clock.clone();

            let t1 = thread::spawn(move || {
                let time1 = c1.time();
                let time2 = c1.time();
                assert!(time2 > time1, "Times must be monotonically increasing");
                time1
            });

            let t2 = thread::spawn(move || {
                let time1 = c2.time();
                let time2 = c2.time();
                assert!(time2 > time1, "Times must be monotonically increasing");
                time1
            });

            let _t1_first = t1.join().unwrap();
            let _t2_first = t2.join().unwrap();

            // All timestamps must be unique (no duplicates)
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_hlc_no_duplicate_timestamps() {
        loom::model(|| {
            let clock = Arc::new(HybridLogicalClock::new());

            let c1 = clock.clone();
            let c2 = clock.clone();

            let t1 = thread::spawn(move || c1.time());
            let t2 = thread::spawn(move || c2.time());

            let time1 = t1.join().unwrap();
            let time2 = t2.join().unwrap();

            // The two timestamps from different threads must be different
            assert_ne!(time1, time2, "Concurrent time() calls must return unique timestamps");
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_hlc_multiple_calls() {
        loom::model(|| {
            let clock = Arc::new(HybridLogicalClock::new());

            let c1 = clock.clone();
            let c2 = clock.clone();

            let t1 = thread::spawn(move || {
                let a = c1.time();
                let b = c1.time();
                (a, b)
            });

            let t2 = thread::spawn(move || {
                let a = c2.time();
                let b = c2.time();
                (a, b)
            });

            let (t1_a, t1_b) = t1.join().unwrap();
            let (t2_a, t2_b) = t2.join().unwrap();

            // Each thread's times must be monotonic
            assert!(t1_b > t1_a);
            assert!(t2_b > t2_a);

            // All four timestamps must be unique
            let mut times = vec![t1_a, t1_b, t2_a, t2_b];
            times.sort();
            for window in times.windows(2) {
                assert_ne!(window[0], window[1], "All timestamps must be unique");
            }
        });
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
    #[cfg(not(loom))]
    use std::{
        sync::Arc,
        thread,
        time::Duration,
    };

    #[cfg(loom)]
    use loom::{
        sync::Arc,
        thread,
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
    #[cfg(not(loom))]
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
    #[cfg(not(loom))]
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
        let store_orderings = [Ordering::SeqCst, Ordering::Release, Ordering::Relaxed];

        // Valid load orderings
        let load_orderings = [Ordering::SeqCst, Ordering::Acquire, Ordering::Relaxed];

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

        // Failure ordering must be no stronger than success and cannot be Release or
        // AcqRel
        let failure_orderings = [Ordering::SeqCst, Ordering::Acquire, Ordering::Relaxed];

        for &success_order in &success_orderings {
            for &failure_order in &failure_orderings {
                // Skip invalid combinations where failure is stronger than success
                if (failure_order == Ordering::SeqCst && success_order != Ordering::SeqCst) {
                    continue;
                }

                let _ = atomic.compare_exchange(42, 100, success_order, failure_order);
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

    // Loom tests for x86_64 AtomicU128
    // These test the custom split hi/lo implementation

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_concurrent_stores() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(0));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            let t1 = thread::spawn(move || {
                a1.store(100, Ordering::SeqCst);
            });

            let t2 = thread::spawn(move || {
                a2.store(200, Ordering::SeqCst);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let final_val = atomic.load(Ordering::SeqCst);
            // Final value must be either 100 or 200
            assert!(final_val == 100 || final_val == 200);
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_compare_exchange() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(0));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            let t1 = thread::spawn(move || {
                a1.compare_exchange(0, 100, Ordering::SeqCst, Ordering::SeqCst)
            });

            let t2 = thread::spawn(move || {
                a2.compare_exchange(0, 200, Ordering::SeqCst, Ordering::SeqCst)
            });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();

            // Exactly one should succeed
            assert!(r1.is_ok() ^ r2.is_ok(), "Exactly one CAS should succeed");

            let final_val = atomic.load(Ordering::SeqCst);
            if r1.is_ok() {
                assert_eq!(final_val, 100);
            } else {
                assert_eq!(final_val, 200);
            }
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_fetch_add() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(0));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            let t1 = thread::spawn(move || {
                a1.fetch_add(10, Ordering::SeqCst)
            });

            let t2 = thread::spawn(move || {
                a2.fetch_add(20, Ordering::SeqCst)
            });

            let old1 = t1.join().unwrap();
            let old2 = t2.join().unwrap();

            // Both old values should be valid
            assert!(old1 == 0 || old1 == 10 || old1 == 20);
            assert!(old2 == 0 || old2 == 10 || old2 == 20);

            // Final value must be 30
            assert_eq!(atomic.load(Ordering::SeqCst), 30);
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_load_while_storing() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(100));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            let t1 = thread::spawn(move || {
                a1.store(200, Ordering::Release);
            });

            let t2 = thread::spawn(move || {
                a2.load(Ordering::Acquire)
            });

            t1.join().unwrap();
            let loaded = t2.join().unwrap();

            // t2 saw either 100 or 200
            assert!(loaded == 100 || loaded == 200);

            // Final must be 200
            assert_eq!(atomic.load(Ordering::SeqCst), 200);
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_compare_exchange_rollback() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(0));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            // Thread 1: changes value from 0 to 50
            let t1 = thread::spawn(move || {
                a1.store(50, Ordering::SeqCst);
            });

            // Thread 2: tries CAS with wrong expected value
            let t2 = thread::spawn(move || {
                // This should fail because value is no longer 0
                a2.compare_exchange(0, 100, Ordering::SeqCst, Ordering::SeqCst)
            });

            t1.join().unwrap();
            let cas_result = t2.join().unwrap();

            let final_val = atomic.load(Ordering::SeqCst);

            // If CAS succeeded, final = 100, else final = 50
            if cas_result.is_ok() {
                assert_eq!(final_val, 100);
            } else {
                assert_eq!(final_val, 50);
            }
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_fetch_sub() {
        loom::model(|| {
            let atomic = Arc::new(AtomicU128::new(100));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            let t1 = thread::spawn(move || {
                a1.fetch_sub(10, Ordering::SeqCst)
            });

            let t2 = thread::spawn(move || {
                a2.fetch_sub(20, Ordering::SeqCst)
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // 100 - 10 - 20 = 70
            assert_eq!(atomic.load(Ordering::SeqCst), 70);
        });
    }

    #[test]
    #[cfg(loom)]
    fn loom_atomic128_high_bits_boundary() {
        loom::model(|| {
            // Test at the boundary between lo and hi u64s
            let boundary = (1u128 << 64) - 1;
            let atomic = Arc::new(AtomicU128::new(boundary));

            let a1 = atomic.clone();
            let a2 = atomic.clone();

            let t1 = thread::spawn(move || {
                a1.fetch_add(1, Ordering::SeqCst)
            });

            let t2 = thread::spawn(move || {
                a2.load(Ordering::SeqCst)
            });

            let old1 = t1.join().unwrap();
            let loaded = t2.join().unwrap();

            let final_val = atomic.load(Ordering::SeqCst);

            // After adding 1 to (2^64 - 1), we get 2^64
            assert_eq!(final_val, 1u128 << 64);
        });
    }
}
