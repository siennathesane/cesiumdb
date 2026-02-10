//! SIMD-accelerated operations for high-performance data processing
//!
//! This module provides SIMD implementations for critical operations:
//! - Fast key comparison for merge operations
//! - Vectorized data scanning
//! - Platform-specific optimizations (x86_64, aarch64)

pub mod key_compare;

pub use key_compare::{
    SimdCapabilities,
    simd_compare_keys,
    simd_memcmp,
};
