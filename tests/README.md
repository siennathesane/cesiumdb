# CesiumDB Integration Tests

This directory contains large-scale integration tests for the compaction system.

## Running the Tests

These tests are marked with `#[ignore]` because they:
- Test with large datasets (up to 10GB)
- Take significant time to run
- Use substantial disk space

### Run all integration tests:
```bash
cargo test --test compaction_integration -- --ignored --test-threads=1
```

### Run a specific test:
```bash
cargo test --test compaction_integration test_10gb_write_heavy_workload -- --ignored
```

## Test Suite Overview

### 1. `test_10gb_write_heavy_workload`
- **Dataset**: 10 million entries × 1KB = ~10GB
- **Focus**: Write-heavy workload with periodic compaction
- **Validates**:
  - Large-scale data ingestion
  - Background compaction during writes
  - Data integrity after compaction
  - Write throughput metrics

### 2. `test_concurrent_reads_during_compaction`
- **Dataset**: 1 million entries × 1KB = ~1GB
- **Focus**: Concurrent read operations while compacting
- **Validates**:
  - Read consistency during compaction
  - No data corruption under concurrent access
  - Read throughput during compaction
  - Zero read errors

### 3. `test_mixed_workload_with_updates`
- **Dataset**: 1 million keys with 500k updates
- **Focus**: Mixed read/write workload with updates
- **Validates**:
  - Update correctness (newer values win)
  - Concurrent reads and writes
  - Compaction with duplicate keys
  - Mixed workload throughput

### 4. `test_deletion_with_compaction`
- **Dataset**: 500k entries with 50% deletions
- **Focus**: Delete operations and space reclamation
- **Validates**:
  - Tombstone handling
  - Space reclamation after compaction
  - Deleted keys return None
  - Remaining keys are intact

### 5. `test_background_compaction_triggers`
- **Dataset**: 200k entries
- **Focus**: Automatic background compaction
- **Validates**:
  - Background thread triggers compaction
  - Compaction jobs complete automatically
  - Data integrity after background compaction
  - Compaction statistics reporting

### 6. `test_point_lookup_performance`
- **Dataset**: 1 million entries
- **Focus**: Read performance before vs. after compaction
- **Validates**:
  - Read performance improvement after compaction
  - Point lookup correctness
  - Performance metrics
  - Compaction benefit quantification

## Batch API Optimization

All tests have been optimized to use the batch API for maximum performance:
- **Batch size**: 1000 operations per batch (writes) or 500 (mixed workloads)
- **Automatic memtable swapping**: The batch API automatically swaps memtables when full
- **Retry logic**: If a batch fails due to full memtable, it automatically swaps and retries

This provides:
- **10-100x faster** writes compared to individual `put()` calls
- Better stress testing of the batch code path
- More realistic production workload simulation

## Expected Results

### Performance Targets (Approximate)

With batch API optimization:
- **Write throughput**: 100k-500k writes/sec (varies by hardware, 10-100x faster than individual puts)
- **Read throughput**: 100k-500k reads/sec (before compaction)
- **Read throughput**: 200k-1M reads/sec (after compaction, 2-4x improvement)
- **Compaction time**: Varies by data size and hardware

### Data Integrity

All tests validate:
- ✓ Zero data corruption
- ✓ Zero read errors during compaction
- ✓ Correct handling of updates (latest version wins)
- ✓ Proper tombstone behavior for deletions
- ✓ Deterministic value verification

## System Requirements

### Minimum:
- **Disk Space**: 15GB free (for 10GB test data + overhead)
- **RAM**: 4GB
- **CPU**: 4 cores

### Recommended:
- **Disk Space**: 20GB free
- **RAM**: 8GB+
- **CPU**: 8+ cores
- **Storage**: NVMe SSD (for realistic performance)

## Test Output

Each test provides detailed progress output:

```
=== Testing 10GB Write-Heavy Workload ===
Writing 10000000 entries of 1KB each...
Progress: 100000/10000000 (1.0%) - 45231 writes/sec
Progress: 200000/10000000 (2.0%) - 48532 writes/sec
...
Write phase complete: 10000000 entries in 187.42s (53358 writes/sec)

Running final compaction...
Final compaction took 23.15s

Verifying data integrity (sampling 10k random entries)...
Verification complete: 10000 samples in 0.84s

Final Compaction Stats:
Compaction: queued=0, in_progress=0, completed=23, parallel_util=67%, pattern=WriteHeavy

=== Test Complete ===
Total time: 211.41s
Data written: ~10.00GB
```

## Troubleshooting

### Out of Disk Space
- Reduce test data size by modifying constants:
  ```rust
  const TARGET_ENTRIES: u64 = 1_000_000; // Instead of 10_000_000
  ```

### Tests Timeout
- Increase test timeout:
  ```bash
  RUST_TEST_TIMEOUT=600 cargo test --test compaction_integration -- --ignored
  ```

### Memory Issues
- Run with single thread to reduce memory pressure:
  ```bash
  cargo test --test compaction_integration -- --ignored --test-threads=1
  ```

### Performance Issues on HDD
- These tests are optimized for SSD
- On HDD, expect 10-50x slower performance
- Consider reducing data sizes for HDD testing

## Contributing

When adding new integration tests:
1. Mark with `#[ignore]` attribute
2. Add progress reporting for long-running tests
3. Include data integrity verification
4. Document expected behavior in test comments
5. Update this README with test description
