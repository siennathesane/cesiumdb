use std::hash::{BuildHasher, Hasher};
use gxhash::{gxhash64, GxHasher};

pub(crate) struct SeedableHasher {
    seed: i64,
    hasher: GxHasher
}

impl SeedableHasher {
    pub(crate) fn new(seed: i64) -> Self {
        Self {seed, hasher: GxHasher::with_seed(seed)}
    }
}

impl Hasher for SeedableHasher {
    fn finish(&self) -> u64 {
        self.hasher.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.hasher.write(bytes)
    }

    fn write_u8(&mut self, i: u8) {
        self.hasher.write_u8(i)
    }

    fn write_u16(&mut self, i: u16) {
        self.hasher.write_u16(i)
    }

    fn write_u32(&mut self, i: u32) {
        self.hasher.write_u32(i)
    }

    fn write_u64(&mut self, i: u64) {
        self.hasher.write_u64(i)
    }

    fn write_u128(&mut self, i: u128) {
        self.hasher.write_u128(i)
    }

    fn write_usize(&mut self, i: usize) {
        self.hasher.write_usize(i)
    }

    fn write_i8(&mut self, i: i8) {
        self.hasher.write_i8(i)
    }

    fn write_i16(&mut self, i: i16) {
        self.hasher.write_i16(i)
    }

    fn write_i32(&mut self, i: i32) {
        self.hasher.write_i32(i)
    }

    fn write_i64(&mut self, i: i64) {
        self.hasher.write_i64(i)
    }

    fn write_i128(&mut self, i: i128) {
        self.hasher.write_i128(i)
    }

    fn write_isize(&mut self, i: isize) {
        self.hasher.write_isize(i)
    }
}

impl BuildHasher for SeedableHasher {
    type Hasher = SeedableHasher;

    fn build_hasher(&self) -> Self::Hasher {
        SeedableHasher::new(self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_same_seed_produces_same_hash() {
        // create two hashers with the same seed
        let mut hasher1 = SeedableHasher::new(42);
        let mut hasher2 = SeedableHasher::new(42);

        // write the same data to both
        hasher1.write(b"test data");
        hasher2.write(b"test data");

        // they should produce identical hashes
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_different_seeds_produce_different_hashes() {
        // create two hashers with different seeds
        let mut hasher1 = SeedableHasher::new(42);
        let mut hasher2 = SeedableHasher::new(43);

        // write the same data to both
        hasher1.write(b"test data");
        hasher2.write(b"test data");

        // they should produce different hashes
        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_consistency_across_restarts() {
        // first "run"
        let mut hasher1 = SeedableHasher::new(123);
        hasher1.write_u32(42);
        hasher1.write_u64(9999);
        hasher1.write(b"hello world");
        let hash1 = hasher1.finish();

        // simulate restart by creating a new hasher with the same seed
        let mut hasher2 = SeedableHasher::new(123);
        hasher2.write_u32(42);
        hasher2.write_u64(9999);
        hasher2.write(b"hello world");
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2, "hashes should be identical across 'restarts'");
    }

    #[test]
    fn test_all_write_methods() {
        // test all the write methods to ensure they're consistently delegated
        let seed = 42;

        // first instance
        let mut hasher1 = SeedableHasher::new(seed);
        hasher1.write_u8(1);
        hasher1.write_u16(2);
        hasher1.write_u32(3);
        hasher1.write_u64(4);
        hasher1.write_u128(5);
        hasher1.write_usize(6);
        hasher1.write_i8(-1);
        hasher1.write_i16(-2);
        hasher1.write_i32(-3);
        hasher1.write_i64(-4);
        hasher1.write_i128(-5);
        hasher1.write_isize(-6);
        let hash1 = hasher1.finish();

        // second instance - should get same results
        let mut hasher2 = SeedableHasher::new(seed);
        hasher2.write_u8(1);
        hasher2.write_u16(2);
        hasher2.write_u32(3);
        hasher2.write_u64(4);
        hasher2.write_u128(5);
        hasher2.write_usize(6);
        hasher2.write_i8(-1);
        hasher2.write_i16(-2);
        hasher2.write_i32(-3);
        hasher2.write_i64(-4);
        hasher2.write_i128(-5);
        hasher2.write_isize(-6);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_build_hasher() {
        let seed = 42;
        let builder = SeedableHasher::new(seed);

        // create two hashers from the same builder
        let mut hasher1 = builder.build_hasher();
        let mut hasher2 = builder.build_hasher();

        // write the same data
        hasher1.write(b"test");
        hasher2.write(b"test");

        // should get the same hash
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_trait_stability() {
        // test that the hasher works correctly with rust's Hash trait
        let seed = 100;

        #[derive(Hash)]
        struct TestStruct {
            a: u32,
            b: String,
        }

        let test_value = TestStruct {
            a: 42,
            b: "hello".to_string(),
        };

        // hash using first instance
        let builder1 = SeedableHasher::new(seed);
        let mut hasher1 = builder1.build_hasher();
        test_value.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        // hash using second instance (after "restart")
        let builder2 = SeedableHasher::new(seed);
        let mut hasher2 = builder2.build_hasher();
        test_value.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2, "hash values should be consistent when using with Hash trait");
    }
}
