use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    for size in [2, 4, 8, 16, 32, 64, 128].iter() {
        let mut group = c.benchmark_group("insertion_sort");
        group.throughput(Throughput::Elements(*size));
        group.bench_with_input(
            BenchmarkId::new("insertion_sort", size),
            size,
            |b, &size| {
                let mut rng = rand_pcg::Pcg32::seed_from_u64(123);
                b.iter_batched_ref(
                    || (0..size).map(|_| rng.gen()).collect(),
                    |data: &mut Vec<u64>| ssort::insertion_sort(data),
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(BenchmarkId::new("std::sort", size), size, |b, &size| {
            let mut rng = rand_pcg::Pcg32::seed_from_u64(123);
            b.iter_batched_ref(
                || (0..size).map(|_| rng.gen()).collect(),
                |data: &mut Vec<u64>| data.sort(),
                BatchSize::SmallInput,
            );
        });
        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
