// Copyright (c) Dom Dwyer <dom@itsallbroken.com>
// SPDX-License-Identifier: BSD-3-Clause


// TODO: run test w/ FilterSize3 distribution, try xor with other

/// FilterSize bounds the allocated size and false-positive rate of a
/// [`Bloom2`](crate::bloom::Bloom2) instance.
///
/// The false positive probability for a bloom filter increases as the number of
/// entries increases. This relationship is demonstrated using 64bit hashes as
/// keys for each possible filter configuration below - you should choose a
/// filter size for your expected load level and hash size.
///
/// The value of FilterSize controls the `k` property of the filter: `k =
/// input_length_bytes / FilterSize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterSize {
    /// 2 bytes / 16 bits per key results in a bloom filter with a minimum
    /// memory usage of ~1024 bytes and a maximum memory usage of ~8KB when
    /// fully populated.
    ///
    /// When using a 64bit hash (4x2 byte keys, `k=4`) the probability of a
    /// false positive is:
    ///
    /// ```text
    ///           +--+------------------------+-----------------------+--------------------+
    ///         1 +                                                *                  *    +
    ///           |                                  *                                     |
    ///           |                                                                        |
    ///     P 0.8 +                         *                                              +
    ///     r     |                                                                        |
    ///     o     |                                                                        |
    ///     b 0.6 +                                                                        +
    ///     a     |                  *                                                     |
    ///     b     |                                                                        |
    ///     i 0.4 +                                                                        +
    ///     l     |              *                                                         |
    ///     i     |                                                                        |
    ///     t 0.2 +                                                                        +
    ///     y     |          *                                                             |
    ///           |        *                                                               |
    ///         0 +  *****                                                                 +
    ///           +--+------------------------+-----------------------+--------------------+
    ///              0                      50000                   1e+05
    ///                                       Number of Entries
    /// ```
    ///
    /// The probability of false positives reaches 1-in-2 after 30118 entries.
    ///
    /// An empty sparse bloom filter would require 16x64 bit block map entries
    /// (128 bytes) to map 1024 64 bit blocks, containing a total of 65536 bits
    /// (memory saving: 98.4375%)
    KeyBytes2 = 2,

    /// 3 bytes / 24 bits per key results in a bloom filter with a minimum
    /// memory usage of ~262KB bytes and a maximum memory usage of ~2MB when
    /// fully populated.
    ///
    /// When using a 64bit hash (2x3 byte keys, `k=2`) the probability of a
    /// false positive is:
    ///
    /// ```text
    ///         1 +--+------------------+-------------------+------------------+-----------+
    ///           |                                                                   *    |
    ///           |                                                *                       |
    ///       0.8 +                                                                        +
    ///     P     |                                  *                                     |
    ///     r     |                                                                        |
    ///     o     |                                                                        |
    ///     b 0.6 +                         *                                              +
    ///     a     |                                                                        |
    ///     b     |                                                                        |
    ///     i 0.4 +                  *                                                     +
    ///     l     |                                                                        |
    ///     i     |              *                                                         |
    ///     t 0.2 +                                                                        +
    ///     y     |          *                                                             |
    ///           |      * *                                                               |
    ///         0 +  ****                                                                  +
    ///           +--+------------------+-------------------+------------------+-----------+
    ///              0                1e+07               2e+07              3e+07
    ///                                       Number of Entries
    /// ```
    ///
    /// The probability of false positives reaches 1-in-2 after 10300768
    /// entries.
    ///
    /// An empty sparse bloom filter would require 4096x64 bit block map entries
    /// (32768 bytes) to map 262144 64 bit blocks, containing a total of
    /// 16777216 bits (memory saving: 98.4375%)
    KeyBytes3 = 3,
}
