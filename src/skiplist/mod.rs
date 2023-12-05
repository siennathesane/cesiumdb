//! A skiplist implementation which allows faster random access than a standard
//! linked list. This implementation is courtesy of JP Ellis' [`rust-skiplist`].
//!
//! SkipLists use a probabilistic distribution of nodes over the internal
//! levels, whereby the lowest level (level 0) contains all the nodes, and each
//! level `n > 0` will contain a random subset of the nodes on level `n - 1`.
//!
//! Most commonly, a geometric distribution is used whereby the chance that a
//! node occupies level `n` is `p` times the chance of occupying level `n-1`
//! (with `0 < p < 1`).
//!
//! It is very unlikely that this will need to be changed as the default should
//! suffice, but if need be custom level generators can be implemented.
//!
//! [`rust-skiplist`]: https://github.com/JP-Ellis/rust-skiplist

mod level_generator;
pub mod ordered_skiplist;
pub mod skiplist;
mod skipnode;
