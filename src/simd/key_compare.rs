//! SIMD-optimized key comparison
#![allow(unused)]
//! Provides vectorized key comparison operations that are significantly
//! faster than byte-by-byte comparison for keys longer than 16 bytes.

use std::cmp::Ordering;

/// SIMD capabilities available on this platform
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    /// SSE2 available (x86_64)
    pub sse2: bool,
    /// AVX2 available (x86_64)
    pub avx2: bool,
    /// NEON available (aarch64)
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detects available SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                avx2: is_x86_feature_detected!("avx2"),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                avx2: false,
                neon: true, // NEON is always available on aarch64
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse2: false,
                avx2: false,
                neon: false,
            }
        }
    }

    /// Returns true if any SIMD acceleration is available
    pub fn has_simd(&self) -> bool {
        self.sse2 || self.avx2 || self.neon
    }
}

/// Fast SIMD-based memory comparison
///
/// Returns Ordering::Equal if slices are equal, otherwise returns the
/// ordering based on the first differing byte.
#[inline]
pub fn simd_memcmp(a: &[u8], b: &[u8]) -> Ordering {
    // For small sizes, fallback to standard comparison
    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }

    if a.len() < 16 {
        return a.cmp(b);
    }

    // Use SIMD for larger comparisons
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 32 {
            // SAFETY: We've checked the feature is available
            unsafe { return simd_memcmp_avx2(a, b) }
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: We've checked the feature is available
            unsafe { return simd_memcmp_sse2(a, b) }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        // SAFETY: NEON is guaranteed on aarch64
        unsafe { return simd_memcmp_neon(a, b) }
    }

    #[allow(unreachable_code)]
    a.cmp(b)
}

/// SIMD-optimized key comparison
///
/// Compares two byte slices using SIMD instructions when beneficial.
#[inline]
pub fn simd_compare_keys(a: &[u8], b: &[u8]) -> Ordering {
    simd_memcmp(a, b)
}

// x86_64 AVX2 implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_memcmp_avx2(a: &[u8], b: &[u8]) -> Ordering {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = a.len();
    let mut offset = 0;

    // Process 32-byte chunks with AVX2
    while offset + 32 <= len {
        // SAFETY: We've verified the offset is in bounds
        let chunk_a = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
        let chunk_b = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);

        let cmp = _mm256_cmpeq_epi8(chunk_a, chunk_b);
        let mask = _mm256_movemask_epi8(cmp);

        // If not all bytes are equal
        if mask != -1 {
            // Find first differing byte
            let diff_byte = (!mask).trailing_zeros() as usize;
            let idx = offset + diff_byte;
            return a[idx].cmp(&b[idx]);
        }

        offset += 32;
    }

    // Handle remaining bytes with scalar comparison
    a[offset..].cmp(&b[offset..])
}

// x86_64 SSE2 implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn simd_memcmp_sse2(a: &[u8], b: &[u8]) -> Ordering {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = a.len();
    let mut offset = 0;

    // Process 16-byte chunks with SSE2
    while offset + 16 <= len {
        // SAFETY: We've verified the offset is in bounds
        let chunk_a = _mm_loadu_si128(a.as_ptr().add(offset) as *const __m128i);
        let chunk_b = _mm_loadu_si128(b.as_ptr().add(offset) as *const __m128i);

        let cmp = _mm_cmpeq_epi8(chunk_a, chunk_b);
        let mask = _mm_movemask_epi8(cmp);

        // If not all bytes are equal
        if mask != 0xffff {
            // Find first differing byte
            let diff_byte = (!mask as u16).trailing_zeros() as usize;
            let idx = offset + diff_byte;
            return a[idx].cmp(&b[idx]);
        }

        offset += 16;
    }

    // Handle remaining bytes with scalar comparison
    a[offset..].cmp(&b[offset..])
}

// aarch64 NEON implementation
#[cfg(target_arch = "aarch64")]
unsafe fn simd_memcmp_neon(a: &[u8], b: &[u8]) -> Ordering {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut offset = 0;

    // Process 16-byte chunks with NEON
    while offset + 16 <= len {
        // SAFETY: We've verified the offset is in bounds
        let chunk_a = unsafe { vld1q_u8(a.as_ptr().add(offset)) };
        // SAFETY: We've verified the offset is in bounds
        let chunk_b = unsafe { vld1q_u8(b.as_ptr().add(offset)) };

        // SAFETY: chunk_a and chunk_b are valid SIMD vectors
        let cmp = unsafe { vceqq_u8(chunk_a, chunk_b) };

        // Extract comparison results to lanes
        // SAFETY: cmp is a valid SIMD vector
        let result_low = unsafe { vget_low_u8(cmp) };
        // SAFETY: cmp is a valid SIMD vector
        let result_high = unsafe { vget_high_u8(cmp) };

        // Check if all bytes are equal
        // SAFETY: result_low and result_high are valid SIMD vectors
        let all_equal = unsafe { vmin_u8(result_low, result_high) };
        // SAFETY: all_equal is a valid SIMD vector, lane 0 is valid
        let all_equal_scalar = unsafe { vget_lane_u64(vreinterpret_u64_u8(all_equal), 0) };

        if all_equal_scalar != !0u64 {
            // Find first differing byte (scalar fallback for simplicity)
            for i in 0..16 {
                let idx = offset + i;
                if a[idx] != b[idx] {
                    return a[idx].cmp(&b[idx]);
                }
            }
        }

        offset += 16;
    }

    // Handle remaining bytes with scalar comparison
    a[offset..].cmp(&b[offset..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_detection() {
        let caps = SimdCapabilities::detect();

        #[cfg(target_arch = "x86_64")]
        {
            // SSE2 should always be available on x86_64
            assert!(caps.sse2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON should always be available on aarch64
            assert!(caps.neon);
        }
    }

    #[test]
    fn test_simd_memcmp_equal() {
        let a = b"hello world this is a test string";
        let b = b"hello world this is a test string";
        assert_eq!(simd_memcmp(a, b), Ordering::Equal);
    }

    #[test]
    fn test_simd_memcmp_less() {
        let a = b"hello world this is a test string";
        let b = b"hello world this is a zest string";
        assert_eq!(simd_memcmp(a, b), Ordering::Less);
    }

    #[test]
    fn test_simd_memcmp_greater() {
        let a = b"hello world this is a zest string";
        let b = b"hello world this is a test string";
        assert_eq!(simd_memcmp(a, b), Ordering::Greater);
    }

    #[test]
    fn test_simd_memcmp_different_lengths() {
        let a = b"short";
        let b = b"longer string";
        assert_eq!(simd_memcmp(a, b), Ordering::Less);
    }

    #[test]
    fn test_simd_memcmp_short() {
        let a = b"abc";
        let b = b"abd";
        assert_eq!(simd_memcmp(a, b), Ordering::Less);
    }

    #[test]
    fn test_simd_memcmp_empty() {
        let a = b"";
        let b = b"";
        assert_eq!(simd_memcmp(a, b), Ordering::Equal);
    }

    #[test]
    fn test_simd_memcmp_long() {
        // Test with data longer than SIMD registers
        let a = vec![0xaau8; 128];
        let b = vec![0xaau8; 128];
        assert_eq!(simd_memcmp(&a, &b), Ordering::Equal);
    }

    #[test]
    fn test_simd_memcmp_long_diff_at_end() {
        // Test difference at the end
        let mut a = vec![0xaau8; 128];
        let mut b = vec![0xaau8; 128];
        a[127] = 0xab;
        b[127] = 0xaa;
        assert_eq!(simd_memcmp(&a, &b), Ordering::Greater);
    }

    #[test]
    fn test_simd_memcmp_long_diff_at_start() {
        // Test difference at the start
        let mut a = vec![0xaau8; 128];
        let mut b = vec![0xaau8; 128];
        a[0] = 0xa9;
        b[0] = 0xaa;
        assert_eq!(simd_memcmp(&a, &b), Ordering::Less);
    }

    #[test]
    fn test_simd_memcmp_long_diff_in_middle() {
        // Test difference in the middle (within SIMD chunk)
        let mut a = vec![0xaau8; 128];
        let mut b = vec![0xaau8; 128];
        a[64] = 0xab;
        b[64] = 0xaa;
        assert_eq!(simd_memcmp(&a, &b), Ordering::Greater);
    }

    #[test]
    fn test_compare_keys() {
        let key1 = b"user:12345:profile";
        let key2 = b"user:12345:settings";
        assert_eq!(simd_compare_keys(key1, key2), Ordering::Less);
    }

    #[test]
    fn test_compare_keys_equal() {
        let key1 = b"user:12345:profile";
        let key2 = b"user:12345:profile";
        assert_eq!(simd_compare_keys(key1, key2), Ordering::Equal);
    }

    // Benchmark-style test (not actual benchmark, just correctness check)
    #[test]
    fn test_simd_vs_scalar_correctness() {
        use rand::Rng;

        let mut rng = rand::rng();

        for _ in 0..100 {
            let len = rng.random_range(1..256);
            let mut a = vec![0u8; len];
            let mut b = vec![0u8; len];

            rng.fill(&mut a[..]);
            rng.fill(&mut b[..]);

            // SIMD and scalar should give same result
            let simd_result = simd_memcmp(&a, &b);
            let scalar_result = a.cmp(&b);

            assert_eq!(
                simd_result, scalar_result,
                "SIMD and scalar gave different results for slices of length {}",
                len
            );
        }
    }
}
