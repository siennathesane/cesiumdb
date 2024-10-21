# CesiumDB

A key-value store focused on performance and security.

## About

CesiumDB is a key-value store. Internally, it's an LSM-tree with a whack of optimizations.

- Key-value separations, keys are stored in different files than values
- Key-size separations (pending)
- Direct device access via raw syscalls
- Reference-only memcpys

### Security

If enabled, CesiumDB can use user-provided encryption keys to secure your data the moment the database takes ownership of your data. As this is a non-trivial performance hit, somewhere between 10-30%, it is disabled by default.

Further, we can securely encrypt every internal allocation (including when we load your already encrypted data) so your data will be very safe.

- User-provided encryption key
- Encrypted in-memory the moment CesiumDB takes ownership
- Securely loaded from disk with checksumming
- Encrypted in-memory
