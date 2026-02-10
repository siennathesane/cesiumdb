use cesiumdb::hlc::{
    HLC,
    HybridLogicalClock,
};
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn clock_gettime(c: &mut Criterion) {
    let clock = HybridLogicalClock::new();
    c.bench_function("HybridLogicalClock::time()", |b| b.iter(|| clock.time()));
}

criterion_group!(benches, clock_gettime);
criterion_main!(benches);
