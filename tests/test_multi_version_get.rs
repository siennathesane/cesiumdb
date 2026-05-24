use std::ops::Bound;
use cesiumdb::{Db, DbOptions};
use tempfile::TempDir;

#[test]
fn test_multi_version_point_read() {
    let temp_dir = TempDir::new().unwrap();
    let mut opts = DbOptions::default();
    opts.data_dir(temp_dir.path().to_path_buf());
    
    let db = Db::open(opts);
    
    db.put(b"testkey", b"value1").unwrap();
    db.put(b"testkey", b"value2").unwrap();
    db.put(b"testkey", b"value3").unwrap();
    
    for i in 0..70_000usize {
        let key = format!("fill_{:08}", i);
        let value = format!("v{:01024}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }
    
    db.sync().unwrap();
    
    // Direct point read
    let result = db.get(b"testkey").unwrap();
    let actual = result.as_ref().map(|b| b.as_ref());
    println!("Direct get: {:?}", actual.map(|b| String::from_utf8_lossy(b)));
    
    // Scan to see all versions in the segment
    let start = b"testkey".to_vec();
    let end = b"testkey".to_vec();
    let mut count = 0;
    for (k, v) in db.scan(Bound::Included(&start), Bound::Included(&end)) {
        println!("Scan entry {}: key={:?}, value={:?}", count, 
            String::from_utf8_lossy(&k), 
            String::from_utf8_lossy(&v));
        count += 1;
    }
    println!("Total scan entries for testkey: {}", count);
    
    assert_eq!(actual, Some(b"value3" as &[u8]), 
        "Expected latest value 'value3', got {:?}", 
        actual.map(|b| String::from_utf8_lossy(b)));
}
