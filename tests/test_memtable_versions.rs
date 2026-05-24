use std::ops::Bound;
use cesiumdb::memtable::Memtable;
use cesiumdb::keypair::{KeyBytes, ValueBytes, DEFAULT_NS};
use bytes::Bytes;

#[test]
fn test_memtable_multiple_versions() {
    let memtable = Memtable::new(0, 64 * 1024 * 1024);
    
    memtable.put(
        KeyBytes::new(DEFAULT_NS, Bytes::from("testkey"), 1),
        ValueBytes::new(DEFAULT_NS, Bytes::from("value1")),
    ).unwrap();
    memtable.put(
        KeyBytes::new(DEFAULT_NS, Bytes::from("testkey"), 2),
        ValueBytes::new(DEFAULT_NS, Bytes::from("value2")),
    ).unwrap();
    memtable.put(
        KeyBytes::new(DEFAULT_NS, Bytes::from("testkey"), 3),
        ValueBytes::new(DEFAULT_NS, Bytes::from("value3")),
    ).unwrap();
    
    let mut count = 0;
    for (k, v) in memtable.scan(Bound::Unbounded, Bound::Unbounded) {
        println!("Entry {}: key={:?}, ts={}, value={:?}", count,
            String::from_utf8_lossy(k.as_ref()), k.ts(),
            String::from_utf8_lossy(v.as_bytes().as_ref()));
        count += 1;
    }
    println!("Total entries: {}", count);
    
    let got = memtable.get(&KeyBytes::new(DEFAULT_NS, Bytes::from("testkey"), 0));
    if let Some(v) = got {
        println!("Direct get: {:?}", String::from_utf8_lossy(v.as_bytes().as_ref()));
    } else {
        println!("Direct get: None");
    }
}
