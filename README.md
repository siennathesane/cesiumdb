# CesiumDB

A key-value store focused on flexibility and performance. It is currently under active development and features are either missing or incomplete. Check on the [features](#features) section to see where things are.

## About

CesiumDB is a key-value store. Internally, it's a hybrid sorted vector.

# Pending Features
**## Data Structure Design
1. Optimize segment and level sizes based on NVMe characteristics
2. Implement a more sophisticated growth strategy for dynamic resizing
3. Consider a log-structured approach for write-heavy workloads
4. Implement a B-tree-like structure for better balancing and search performance

## Performance Optimizations
1. Implement batch operations for inserts, deletes, and reads
2. Use SIMD instructions for larger data copies
3. Optimize memory alignment for better cache performance
4. Implement a smart caching strategy for frequently accessed segments
5. Use memory prefetching to reduce latency
6. Optimize the rebalancing strategy to minimize full reconstructions

## Concurrency and Parallelism
1. Implement thread-safe access for concurrent reads and writes
2. Use fine-grained locking or lock-free data structures for better parallelism
3. Implement parallel search and insert operations across levels
4. Consider using a thread pool for background operations like rebalancing

## Memory Management
1. Implement a custom allocator optimized for NVMe access patterns
2. Use memory mapping more efficiently, possibly with huge pages
3. Implement a segmented memory model to handle datasets larger than available RAM

## NVMe Optimizations
1. Align data structures with NVMe page sizes
2. Optimize I/O patterns to match NVMe controller characteristics
3. Implement NVMe command queues for asynchronous operations
4. Utilize NVMe's parallel access capabilities

## Persistence and Durability
1. Implement checkpointing for faster recovery
2. Add journaling for crash consistency
3. Implement incremental persistence to reduce I/O overhead
4. Add configurable durability levels (e.g., fsync on every write vs. periodic syncs)

## Error Handling and Recovery
1. Implement custom error types for more detailed error reporting
2. Add error recovery mechanisms for corrupted data
3. Implement a robust crash recovery system
4. Add data integrity checks (e.g., checksums)

## Scalability
1. Implement sharding for distributing data across multiple NVMe devices
2. Add support for distributed operations across multiple nodes
3. Implement a mechanism to handle datasets much larger than a single NVMe device

## API and Usability
1. Implement range queries for efficient interval searches
2. Add support for composite keys and secondary indexes
3. Implement an iterator interface for more flexible data access
4. Add support for user-defined serialization/deserialization

## Maintenance and Monitoring
1. Implement statistics collection for performance monitoring
2. Add a background process for periodic optimization and defragmentation
3. Implement data compaction to reclaim space from deleted items
4. Add logging and tracing for debugging and performance analysis

## Flexibility and Extensibility
1. Make the data structure generic over different storage backends
2. Implement pluggable comparison and hashing functions
3. Add support for different segment layouts for various use cases
4. Implement versioning for backward compatibility as the structure evolves

## Testing and Validation
1. Implement comprehensive unit and integration tests
2. Add fuzz testing to find edge cases and improve robustness
3. Implement benchmarking suite for performance testing
4. Add stress tests for concurrency and large data volumes

## Specific Feature Improvements
1. Enhance the `get` method with caching for frequently accessed items
2. Implement a more efficient `delete` method with lazy deletion
3. Add bulk insert and delete operations
4. Implement a more sophisticated rebalancing strategy

## Memory Safety and Security
1. Minimize use of unsafe code and add thorough safety checks
2. Implement bounds checking for all memory accesses
3. Add encryption support for sensitive data
4. Implement access control mechanisms

## References

```doi
Giorgos Xanthakis, Giorgos Saloustros, Nikos Batsaras, Anastasios Papagiannis, and Angelos Bilas. 2021. Parallax: Hybrid Key-Value Placement in LSM-based Key-Value Stores. In Proceedings of the ACM Symposium on Cloud Computing (SoCC '21). Association for Computing Machinery, New York, NY, USA, 305–318.
DOI:https://doi.org/10.1145/3472883.3487012
```