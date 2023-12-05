use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};

mod skiplist;

// Group Benchmarks
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
    crate::skiplist::iter,
    crate::skiplist::push_back,
    crate::skiplist::push_front,
    crate::skiplist::rand_access,
);

// Benchmarks
criterion_main!(benches);
