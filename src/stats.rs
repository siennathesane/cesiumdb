// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

use std::sync::{
    atomic::AtomicUsize,
    LazyLock,
};

pub(crate) static STATS: LazyLock<Stats> = LazyLock::new(|| Stats::default());

#[derive(Debug, Default)]
pub(crate) struct Stats {
    pub(crate) current_threads: AtomicUsize,
}
