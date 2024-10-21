extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // TODO(@siennathesane): enable ffi support.
    let enable_str = env::var("ENABLE_FFI").unwrap_or_else(|_| "false".to_string());
    if enable_str != "true" {
        return;
    }

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("cesiumdb.h");
}
