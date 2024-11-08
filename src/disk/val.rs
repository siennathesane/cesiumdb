// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

#[repr(C)]
#[derive(Debug, Eq, Clone, Copy, PartialEq)]
pub(crate) struct Value<T: AsRef<[u8]>> {
    size: u32,
    data: T,
}

impl<T: AsRef<[u8]>> Value<T> {
    fn new(data: T) -> Self {
        Self {
            size: size_of_val(&data) as u32,
            data,
        }
    }

    pub(crate) fn size(&self) -> u32 {
        self.size
    }
}
