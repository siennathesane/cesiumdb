// Copyright (c) Sienna Satterwhite, CesiumDB Contributors
// SPDX-License-Identifier: GPL-3.0-only WITH Classpath-exception-2.0

/// A peekable adapter for an iterator. This allows us to peek at the next item
/// without consuming it. It's designed for use with the iterators.
pub(crate) struct Peekable<I: Iterator> {
    iter: I,
    peeked: Option<Option<I::Item>>,
}

impl<I: Iterator> Peekable<I> {
    pub(crate) fn new(iter: I) -> Self {
        Self { iter, peeked: None }
    }

    #[inline]
    pub(crate) fn peek(&mut self) -> Option<&I::Item> {
        if self.peeked.is_none() {
            self.peeked = Some(self.iter.next());
        }
        self.peeked.as_ref().unwrap().as_ref()
    }

    #[inline]
    pub(crate) fn peek_mut(&mut self) -> Option<&mut I::Item> {
        if self.peeked.is_none() {
            self.peeked = Some(self.iter.next());
        }
        self.peeked.as_mut().unwrap().as_mut()
    }
}

impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(peeked) = self.peeked.take() {
            peeked
        } else {
            self.iter.next()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        match &self.peeked {
            | Some(Some(_)) => (lower + 1, upper.map(|u| u + 1)),
            | _ => (lower, upper),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Peekable;

    #[test]
    fn test_peek() {
        let vec = vec![1, 2, 3];
        let mut peekable = Peekable::new(vec.into_iter());

        assert_eq!(peekable.peek(), Some(&1));
        assert_eq!(peekable.peek(), Some(&1)); // Peek again to ensure it doesn't consume
        assert_eq!(peekable.next(), Some(1)); // Now consume
        assert_eq!(peekable.peek(), Some(&2));
    }

    #[test]
    fn test_peek_mut() {
        let vec = vec![1, 2, 3];
        let mut peekable = Peekable::new(vec.into_iter());

        if let Some(value) = peekable.peek_mut() {
            *value = 10;
        }
        assert_eq!(peekable.next(), Some(10));
        assert_eq!(peekable.peek(), Some(&2));
    }

    #[test]
    fn test_next() {
        let vec = vec![1, 2, 3];
        let mut peekable = Peekable::new(vec.into_iter());

        assert_eq!(peekable.next(), Some(1));
        assert_eq!(peekable.next(), Some(2));
        assert_eq!(peekable.next(), Some(3));
        assert_eq!(peekable.next(), None);
    }

    #[test]
    fn test_size_hint() {
        let vec = vec![1, 2, 3];
        let mut peekable = Peekable::new(vec.into_iter());

        assert_eq!(peekable.size_hint(), (3, Some(3)));
        peekable.peek();
        assert_eq!(peekable.size_hint(), (3, Some(3)));
        peekable.next();
        assert_eq!(peekable.size_hint(), (2, Some(2)));
    }
}
