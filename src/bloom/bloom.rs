// Copyright (c) Dom Dwyer <dom@itsallbroken.com>
// SPDX-License-Identifier: BSD-3-Clause

use std::{
    collections::hash_map::RandomState,
    hash::{
        BuildHasher,
        Hash,
    },
    marker::PhantomData,
};

use crate::bloom::{
    FilterSize,
    bitmap::CompressedBitmap,
};
// TODO(dom): AND, XOR, NOT + examples

// [`Bloom2`]: crate::bloom::Bloom2
// [`BloomFilterBuilder`]: crate::bloom::BloomFilterBuilder
// [`hash`]: std::hash::Hash
// [`FilterSize`]: crate::bloom::FilterSize

/// A trait to abstract bit storage for use in a
/// [`Bloom2`](crate::bloom::Bloom2) filter.
pub trait Bitmap {
    /// Construct a new [`Bitmap`] impl with capacity to hold at least `max_key`
    /// number of bits.
    fn new_with_capacity(max_key: usize) -> Self;

    /// Set bit indexed by `key` to `value`.
    fn set(&mut self, key: usize, value: bool);

    /// Return `true` if the given bit index was previously set to `true`.
    fn get(&self, key: usize) -> bool;
}

/// Construct [`Bloom2`] instances with varying parameters.
///
/// ```rust
/// use std::collections::hash_map::RandomState;
///
/// use crate::bloom::{
///     BloomFilterBuilder,
///     FilterSize,
/// };
///
/// let mut filter = BloomFilterBuilder::hasher(RandomState::default())
///     .size(FilterSize::KeyBytes2)
///     .build();
///
/// filter.insert(&"success!");
/// ```
pub struct BloomFilterBuilder<H, B>
where
    H: BuildHasher,
    B: Bitmap, {
    hasher: H,
    bitmap: B,
    key_size: FilterSize,
}

/// Initialise a `BloomFilterBuilder` that unless changed, will construct a
/// `Bloom2` instance using a [2 byte key] and use Rust's [`DefaultHasher`]
/// ([SipHash] at the time of writing).
///
/// [2 byte key]: crate::bloom::FilterSize::KeyBytes2
/// [`DefaultHasher`]: std::collections::hash_map::RandomState
/// [SipHash]: https://131002.net/siphash/
impl std::default::Default for BloomFilterBuilder<RandomState, CompressedBitmap> {
    fn default() -> BloomFilterBuilder<RandomState, CompressedBitmap> {
        let size = FilterSize::KeyBytes2;
        BloomFilterBuilder {
            hasher: RandomState::default(),
            bitmap: CompressedBitmap::new(key_size_to_bits(size)),
            key_size: size,
        }
    }
}

impl<H, B> BloomFilterBuilder<H, B>
where
    H: BuildHasher,
    B: Bitmap,
{
    /// Set the bit storage (bitmap) for the bloom filter.
    ///
    /// # Panics
    ///
    /// This method may panic if `bitmap` is too small to hold any value in the
    /// range produced by the [key size](FilterSize).
    ///
    /// Providing a `bitmap` instance that is non-empty can be used to restore
    /// the state of a [`Bloom2`] instance (although using `serde` can achieve
    /// this safely too).
    pub fn with_bitmap_data(self, bitmap: B, key_size: FilterSize) -> Self {
        // Invariant: reading the last bit succeeds, ensuring it has sufficient
        // capacity.
        let _ = bitmap.get(key_size as usize);

        Self {
            bitmap,
            key_size,
            ..self
        }
    }

    pub fn with_bitmap<U>(self) -> BloomFilterBuilder<H, U>
    where
        U: Bitmap, {
        BloomFilterBuilder {
            hasher: self.hasher,
            bitmap: U::new_with_capacity(key_size_to_bits(self.key_size)),
            key_size: self.key_size,
        }
    }

    /// Initialise the [`Bloom2`] instance with the provided parameters.
    pub fn build<T: Hash>(self) -> Bloom2<H, B, T> {
        Bloom2 {
            hasher: self.hasher,
            bitmap: self.bitmap,
            key_size: self.key_size,
            _key_type: PhantomData,
        }
    }

    /// Control the in-memory size and false-positive probability of the filter.
    ///
    /// Setting the bitmap size replaces the current `Bitmap` instance with a
    /// new `CompressedBitmap` of the appropriate size.
    ///
    /// See [`FilterSize`].
    pub fn size(self, size: FilterSize) -> Self {
        Self {
            key_size: size,
            bitmap: B::new_with_capacity(key_size_to_bits(size)),
            ..self
        }
    }
}

impl<H> BloomFilterBuilder<H, CompressedBitmap>
where
    H: BuildHasher,
{
    /// Initialise a `BloomFilterBuilder` that unless changed, will construct a
    /// `Bloom2` instance using a [2 byte key] and use the specified hasher.
    ///
    /// [2 byte key]: crate::bloom::FilterSize::KeyBytes2
    pub fn hasher(hasher: H) -> Self {
        let size = FilterSize::KeyBytes2;
        Self {
            hasher,
            bitmap: CompressedBitmap::new(key_size_to_bits(size)),
            key_size: size,
        }
    }
}

fn key_size_to_bits(k: FilterSize) -> usize {
    2_usize.pow(8 * k as u32)
}

/// A fast, memory efficient, sparse bloom filter.
///
/// Most users can quickly initialise a `Bloom2` instance by calling
/// `Bloom2::default()` and start inserting anything that implements the
/// [`Hash`] trait:
///
/// ```rust
/// use crate::bloom::Bloom2;
///
/// let mut b = Bloom2::default();
/// b.insert(&"hello 🐐");
/// assert!(b.contains(&"hello 🐐"));
/// ```
///
/// Initialising a `Bloom2` this way uses some [sensible
/// default](crate::bloom::BloomFilterBuilder) values for a easy-to-use, memory
/// efficient filter. If you want to tune the probability of a false-positive
/// lookup, change the hashing algorithm, memory size of the filter, etc, a
/// [`BloomFilterBuilder`] can be used to initialise a `Bloom2` instance with
/// the desired properties.
///
/// The sparse nature of this filter trades a small amount of insert performance
/// for decreased memory usage. For filters initialised infrequently and held
/// for a meaningful duration of time, this is almost always worth the
/// marginally increased insert latency. When testing performance, be sure to
/// use a release build - there's a significant performance difference!
#[derive(Debug, Clone, PartialEq)]
pub struct Bloom2<H, B, T>
where
    H: BuildHasher,
    B: Bitmap, {
    hasher: H,
    bitmap: B,
    key_size: FilterSize,

    _key_type: PhantomData<T>,
}

/// Initialise a `Bloom2` instance using the default implementation of
/// [`BloomFilterBuilder`].
///
/// It is the equivalent of:
///
/// ```rust
/// use crate::bloom::BloomFilterBuilder;
///
/// let mut b = BloomFilterBuilder::default().build();
/// # b.insert(&42);
/// ```
impl<T> std::default::Default for Bloom2<RandomState, CompressedBitmap, T>
where
    T: Hash,
{
    fn default() -> Self {
        crate::bloom::BloomFilterBuilder::default().build()
    }
}

impl<H, B, T> Bloom2<H, B, T>
where
    H: BuildHasher,
    B: Bitmap,
    T: Hash,
{
    /// Insert places `data` into the bloom filter.
    ///
    /// Any subsequent calls to [`contains`](Bloom2::contains) for the same
    /// `data` will always return true.
    ///
    /// Insertion is significantly faster in release builds.
    ///
    /// The `data` provided can be anything that implements the [`Hash`] trait,
    /// for example:
    ///
    /// ```rust
    /// use crate::bloom::Bloom2;
    ///
    /// let mut b = Bloom2::default();
    /// b.insert(&"hello 🐐");
    /// assert!(b.contains(&"hello 🐐"));
    ///
    /// let mut b = Bloom2::default();
    /// b.insert(&vec!["fox", "cat", "banana"]);
    /// assert!(b.contains(&vec!["fox", "cat", "banana"]));
    ///
    /// let mut b = Bloom2::default();
    /// let data: [u8; 4] = [1, 2, 3, 42];
    /// b.insert(&data);
    /// assert!(b.contains(&data));
    /// ```
    ///
    /// As well as structs if they implement the [`Hash`] trait, which be
    /// helpfully derived:
    ///
    /// ```rust
    /// # use crate::bloom::Bloom2;
    /// # let mut b = Bloom2::default();
    /// #[derive(Hash)]
    /// struct User {
    ///     id: u64,
    ///     email: String,
    /// }
    ///
    /// let user = User {
    ///     id: 42,
    ///     email: "dom@itsallbroken.com".to_string(),
    /// };
    ///
    /// b.insert(&&user);
    /// assert!(b.contains(&&user));
    /// ```
    pub fn insert(&mut self, data: &'_ T) {
        // Generate a hash (u64) value for data and split the u64 hash into
        // several smaller values to use as unique indexes in the bitmap.
        self.hasher
            .hash_one(data)
            .to_be_bytes()
            .chunks(self.key_size as usize)
            .for_each(|chunk| self.bitmap.set(bytes_to_usize_key(chunk), true));
    }

    /// Checks if `data` exists in the filter.
    ///
    /// If `contains` returns true, `hash` has **probably** been inserted
    /// previously. If `contains` returns false, `hash` has **definitely not**
    /// been inserted into the filter.
    pub fn contains(&self, data: &'_ T) -> bool {
        // Generate a hash (u64) value for data
        self.hasher
            .hash_one(data)
            .to_be_bytes()
            .chunks(self.key_size as usize)
            .all(|chunk| self.bitmap.get(bytes_to_usize_key(chunk)))
    }

    pub fn bitmap(&self) -> &B {
        &self.bitmap
    }
}

fn bytes_to_usize_key<'a, I: IntoIterator<Item = &'a u8>>(bytes: I) -> usize {
    bytes
        .into_iter()
        .fold(0, |key, &byte| (key << 8) | byte as usize)
}
