extern crate flatc_rust; // or just `use flatc_rust;` with Rust 2018 edition.

use std::{
    io::Result,
    path::Path,
};

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=src/types/table.fbs");
    flatc_rust::run(flatc_rust::Args {
        inputs: &[Path::new("src/types/table.fbs")],
        out_dir: Path::new("src/types/"),
        ..Default::default()
    })
    .expect("flatc");

    println!();
    let mut prost_build = prost_build::Config::new();
    prost_build.out_dir(Path::new("src/types/"));

    prost_build.compile_protos(&["src/types/internals.proto"], &["src/"])?;

    Ok(())
}
