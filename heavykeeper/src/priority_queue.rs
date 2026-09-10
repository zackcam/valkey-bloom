use crate::sip::SipState;
use hashbrown::HashTable;
use std::borrow::Borrow;
use std::hash::Hash;

use crate::cuckoo::{realloc_large_heap_allocated_object, Reallocator};

/// Relocate `vec`'s backing allocation through `reallocator`, in place. Trimmed
/// to a boxed slice first to drop spare capacity; the rebuilt `Vec` has
/// capacity equal to its length.
fn realloc_vec<E, R: Reallocator>(vec: &mut Vec<E>, reallocator: &mut R) {
    let mut boxed = std::mem::take(vec).into_boxed_slice();
    realloc_large_heap_allocated_object(&mut boxed, reallocator);
    *vec = boxed.into_vec();
}

#[derive(Clone)]
struct Slot<T> {
    item: T,
    count: u64,
    sequence: u32,
    heap_pos: u32,
}

/// A specialized priority queue for HeavyKeeper that maintains top-k items by count
///
/// - `linear = true` (default): linear scan over `item_store`. Better cache
///   locality for small `k` and avoids hashing on the lookup path; the hash
///   table is left unallocated.
/// - `linear = false`: a `hashbrown` hash table maps items to slot indices for
///   O(1) lookup, at the cost of the table's memory and per-op hashing.
#[derive(Clone)]
pub(crate) struct TopKQueue<T> {
    item_store: Vec<Slot<T>>,
    heap: Vec<u32>,        // slot indices, min-heap ordered by count
    table: HashTable<u32>, // hash -> slot index into `item_store` (unused when `linear`)
    linear: bool,
    capacity: usize,
    /// Monotonic insertion counter used to break count ties (older wins). On
    /// wrap at `u32::MAX` the live slots are renumbered in relative order via
    /// [`Self::next_sequence`], so tie ordering never degrades.
    sequence: u32,
    hasher: SipState,
}

impl<T: Ord + Clone + Hash + PartialEq> TopKQueue<T> {
    /// Build a queue using the hash-table lookup strategy.
    pub(crate) fn with_capacity_and_hasher(capacity: usize, hasher: SipState) -> Self {
        Self::with_capacity_hasher_linear(capacity, hasher, false)
    }

    /// Build a queue, choosing the lookup strategy: `linear` scans `item_store`
    /// and leaves the hash table unallocated; otherwise the hash table is used.
    pub(crate) fn with_capacity_hasher_linear(
        capacity: usize,
        hasher: SipState,
        linear: bool,
    ) -> Self {
        Self {
            item_store: Vec::with_capacity(capacity),
            heap: Vec::with_capacity(capacity + 1),
            // Linear lookup never touches the table; leave it unallocated.
            table: if linear {
                HashTable::new()
            } else {
                HashTable::with_capacity(capacity)
            },
            linear,
            capacity,
            sequence: 0,
            hasher,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_hasher_linear(capacity, SipState::random(), true)
    }

    pub(crate) fn len(&self) -> usize {
        self.item_store.len()
    }

    /// Whether this queue uses the linear-scan lookup strategy.
    pub(crate) fn linear(&self) -> bool {
        self.linear
    }

    /// Worst-case structural bytes to budget per tracked entry, for sizing a
    /// queue from `capacity` before it exists: the slot and heap-index cell
    /// (exact, from the real types) plus the optional hash table's share
    /// (upper bound: hashbrown documents a 7/8 max load factor and
    /// power-of-two bucket counts at 5 bytes per bucket -- one `u32` index
    /// plus one control byte -- so at most `2 * 8/7 * 5 < 12` bytes per
    /// entry). Excludes the item's own heap bytes, which depend on runtime
    /// data. Actual usage is reported exactly by [`Self::mem_bytes`].
    pub(crate) const fn entry_mem_bytes() -> usize {
        const TABLE_SHARE_UPPER: usize = 12;
        std::mem::size_of::<Slot<T>>() + std::mem::size_of::<u32>() + TABLE_SHARE_UPPER
    }

    /// Returns the heap memory (in bytes) used by this queue's containers.
    ///
    /// Computed from the allocated *capacity* of the slots, heap vector,
    /// and hash table, plus the heap each live item owns beyond
    /// its inline `size_of::<T>()`. `item_heap(t)` should return the bytes `t`
    /// points to (e.g. `String::capacity`).
    pub(crate) fn mem_bytes<F>(&self, item_heap: F) -> usize
    where
        F: Fn(&T) -> usize,
    {
        use std::mem::size_of;
        let store_bytes = self.item_store.capacity() * size_of::<Slot<T>>();
        let heap_bytes = self.heap.capacity() * size_of::<u32>();
        // hashbrown reports its own allocation, so the estimate cannot drift
        // from the crate's internal layout across upgrades.
        let table_bytes = self.table.allocation_size();
        let item_bytes: usize = self.item_store.iter().map(|s| item_heap(&s.item)).sum();
        store_bytes + heap_bytes + table_bytes + item_bytes
    }

    /// Relocate the `heap` and `item_store` vectors through `reallocator`. For
    /// `item_store` only the outer buffer moves; any heap a `T` owns (e.g. a
    /// `Vec<u8>` key's bytes) stays put, as elements are copied byte-for-byte.
    /// The hash table owns its own allocation and is not relocated.
    pub(crate) fn realloc_large_heap_allocated_objects<R: Reallocator>(
        &mut self,
        reallocator: &mut R,
    ) {
        realloc_vec(&mut self.heap, reallocator);
        realloc_vec(&mut self.item_store, reallocator);
    }

    pub(crate) fn get<Q>(&self, item: &Q) -> Option<u64>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ToOwned<Owned = T> + ?Sized,
    {
        self.find_slot(item).map(|idx| self.item_store[idx].count)
    }

    pub(crate) fn contains<Q>(&self, item: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find_slot(item).is_some()
    }

    /// Update an existing entry's count to `count`, if `item` is tracked.
    ///
    /// Returns `true` if `item` is present (whether or not this call
    /// actually raised its count), `false` if it isn't tracked at all —
    #[inline]
    pub(crate) fn update_if_present<Q>(&mut self, item: &Q, count: u64) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(slot_idx) = self.find_slot(item) {
            let slot = &mut self.item_store[slot_idx];
            // The count-min-sketch estimate can fall below the value already
            // tracked (a later add where every row decayed yields a smaller
            // max_count). The PQ keeps the high-water estimate, so a
            // non-increasing update is a no-op, not an error.
            if count <= slot.count {
                return true;
            }
            slot.count = count;
            let pos = slot.heap_pos as usize;
            self.sift_down(pos);
            true
        } else {
            false
        }
    }

    pub(crate) fn min_count(&self) -> u64 {
        // If heap is empty, return 0
        // Otherwise return count from root node (index 0)
        if self.item_store.is_empty() {
            0
        } else {
            self.item_store[self.heap[0] as usize].count
        }
    }

    pub(crate) fn is_full(&self) -> bool {
        self.item_store.len() >= self.capacity
    }

    /// Return the next tie-break sequence value. When the counter would wrap,
    /// renumber the live slots to 1..=len preserving their relative order, so
    /// count-tie ordering stays correct instead of silently degrading. The
    /// renumber is O(k log k) once per 2^32 assignments, amortized nothing.
    fn next_sequence(&mut self) -> u32 {
        if self.sequence == u32::MAX {
            let mut order: Vec<u32> = (0..self.item_store.len() as u32).collect();
            order.sort_unstable_by_key(|&idx| self.item_store[idx as usize].sequence);
            for (new_seq, &idx) in order.iter().enumerate() {
                self.item_store[idx as usize].sequence = new_seq as u32 + 1;
            }
            // Slot count is bounded by capacity (well below u32::MAX), so the
            // counter has full headroom again.
            self.sequence = self.item_store.len() as u32;
        }
        self.sequence += 1;
        self.sequence
    }

    /// Insert or update `item` to `count`.
    ///
    /// Returns `Some(evicted)` when a previously tracked item is displaced
    /// by this call, otherwise `None`.
    pub(crate) fn upsert(&mut self, item: T, count: u64) -> Option<T> {
        let hash = if self.linear {
            0
        } else {
            self.hasher.hash_one(&item)
        };
        // Fast path: update existing item
        let existing = if self.linear {
            self.find_slot_linear(&item)
        } else {
            self.find_slot_with_hash(&item, hash)
        };
        if let Some(slot_idx) = existing {
            let slot = &mut self.item_store[slot_idx];
            if count == slot.count {
                return None;
            }
            slot.count = count;
            let pos = slot.heap_pos as usize;
            self.sift_down(pos);
            self.sift_up(pos);
            return None;
        }

        // For new items, if we have space just add it
        if self.item_store.len() < self.capacity {
            // Restore capacity to k after a defrag trimmed it, so it stays a
            // known constant for memory tracking.
            if self.heap.capacity() < self.capacity + 1 {
                self.heap.reserve_exact(self.capacity + 1 - self.heap.len());
            }
            if self.item_store.capacity() < self.capacity {
                self.item_store
                    .reserve_exact(self.capacity - self.item_store.len());
            }

            let slot_idx = self.item_store.len() as u32;
            let heap_pos = slot_idx;
            let sequence = self.next_sequence();

            self.item_store.push(Slot {
                item,
                count,
                sequence,
                heap_pos,
            });
            self.heap.push(slot_idx);

            if !self.linear {
                self.table.insert_unique(hash, slot_idx, |&idx| {
                    self.hasher.hash_one(&self.item_store[idx as usize].item)
                });
            }
            self.sift_up(heap_pos as usize);
            return None;
        }

        // Queue is full - check if new count beats minimum
        if !self.item_store.is_empty() {
            let min_slot_idx = self.heap[0] as usize;
            let min_count = self.item_store[min_slot_idx].count;
            if count > min_count {
                if !self.linear {
                    let old_hash = self.hasher.hash_one(&self.item_store[min_slot_idx].item);
                    match self
                        .table
                        .find_entry(old_hash, |&idx| idx == min_slot_idx as u32)
                    {
                        Ok(entry) => {
                            entry.remove();
                        }
                        // A miss means the table and item_store have desynced;
                        // surface it in debug builds instead of leaving a stale
                        // entry pointing at the reused slot.
                        Err(_) => debug_assert!(
                            false,
                            "evicted min slot {min_slot_idx} missing from lookup table"
                        ),
                    }
                }

                let old_item =
                    std::mem::replace(&mut self.item_store[min_slot_idx].item, item);
                self.item_store[min_slot_idx].count = count;
                self.item_store[min_slot_idx].sequence = self.next_sequence();

                if !self.linear {
                    self.table.insert_unique(hash, min_slot_idx as u32, |&idx| {
                        self.hasher.hash_one(&self.item_store[idx as usize].item)
                    });
                }
                self.sift_down(0);
                return Some(old_item);
            }
        }
        None
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&T, u64)> {
        let mut items: Vec<_> = self
            .item_store
            .iter()
            .map(|s| (&s.item, s.count, s.sequence))
            .collect();

        // Sort by count descending, then by sequence ascending.
        items.sort_unstable_by(|(_, c1, s1), (_, c2, s2)| match c2.cmp(c1) {
            std::cmp::Ordering::Equal => s1.cmp(s2),
            other => other,
        });

        // Return an iterator over (&T, count), preserving sorted order.
        items.into_iter().map(|(k, count, _)| (k, count))
    }

    /// Iterate items in ascending insertion-`sequence` order.
    ///
    /// Serialization uses this so restore (re-`upsert` in this order) reassigns
    /// sequences that preserve the count-tie ordering.
    pub(crate) fn iter_by_sequence(&self) -> impl Iterator<Item = (&T, u64)> {
        let mut items: Vec<_> = self
            .item_store
            .iter()
            .map(|s| (&s.item, s.count, s.sequence))
            .collect();
        items.sort_unstable_by_key(|(_, _, seq)| *seq);
        items.into_iter().map(|(k, count, _)| (k, count))
    }

    fn find_slot<Q>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.linear {
            return self.find_slot_linear(item);
        }
        let hash = self.hasher.hash_one(item);
        self.find_slot_with_hash(item, hash)
    }

    /// Linear scan over `item_store`. Used when `linear` is set.
    #[inline]
    fn find_slot_linear<Q>(&self, item: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.item_store
            .iter()
            .position(|s| s.item.borrow() == item)
    }

    #[inline]
    fn find_slot_with_hash<Q>(&self, item: &Q, hash: u64) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.table
            .find(hash, |&idx| {
                self.item_store[idx as usize].item.borrow() == item
            })
            .map(|&idx| idx as usize)
    }

    // Binary heap helper methods using Eytzinger layout (0-based indexing)
    fn parent(i: usize) -> usize {
        (i - 1) >> 1
    }
    fn left(i: usize) -> usize {
        2 * i + 1
    }
    fn right(i: usize) -> usize {
        2 * i + 2
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = Self::parent(pos);
            if self.item_store[self.heap[parent] as usize].count
                > self.item_store[self.heap[pos] as usize].count
            {
                self.swap_nodes(parent, pos);
                pos = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        loop {
            let mut smallest = pos;
            let left = Self::left(pos);
            let right = Self::right(pos);

            if left < self.heap.len()
                && self.item_store[self.heap[left] as usize].count
                    < self.item_store[self.heap[smallest] as usize].count
            {
                smallest = left;
            }
            if right < self.heap.len()
                && self.item_store[self.heap[right] as usize].count
                    < self.item_store[self.heap[smallest] as usize].count
            {
                smallest = right;
            }

            if smallest == pos {
                break;
            }

            self.swap_nodes(pos, smallest);
            pos = smallest;
        }
    }

    fn swap_nodes(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        // Update heap positions in item_store
        self.item_store[self.heap[i] as usize].heap_pos = i as u32;
        self.item_store[self.heap[j] as usize].heap_pos = j as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion() {
        let mut queue = TopKQueue::with_capacity(2);
        queue.upsert("a", 1);
        queue.upsert("b", 2);

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"b", 2), (&"a", 1)]);
    }

    #[test]
    fn test_update_existing() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, SipState::random());
        queue.upsert("a", 1);
        queue.upsert("b", 2);
        queue.upsert("a", 3); // Update a's count

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"a", 3), (&"b", 2)]);
    }

    #[test]
    fn test_heap_cleanup() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, SipState::random());

        // Insert initial items
        queue.upsert("a", 1);
        queue.upsert("b", 2);

        // Update 'a' multiple times
        queue.upsert("a", 3);
        queue.upsert("a", 4);
        queue.upsert("a", 5);

        // Insert new item with higher count
        queue.upsert("c", 6);

        // Check heap size vs items size
        assert_eq!(queue.heap.len(), 2, "Expected 2 items");

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"c", 6), (&"a", 5)]);
    }

    #[test]
    fn test_insertion_order() {
        let mut queue = TopKQueue::with_capacity_and_hasher(3, SipState::random());

        // Insert items with same count in specific order
        queue.upsert("a", 1);
        queue.upsert("b", 1);
        queue.upsert("c", 1);

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"a", 1), (&"b", 1), (&"c", 1)]);
    }

    #[test]
    fn test_heap_consistency() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, SipState::random());

        // Fill queue
        queue.upsert("a", 1);
        queue.upsert("b", 2);

        // Update existing item multiple times
        for i in 3..10 {
            queue.upsert("a", i);
        }

        // Try to insert new item
        queue.upsert("c", 5);

        // Verify min_count is accurate
        assert_eq!(queue.min_count(), 5);
    }

    #[test]
    fn test_capacity_overflow() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, SipState::random());

        // Insert more items than capacity
        queue.upsert("a", 1);
        queue.upsert("b", 2);
        queue.upsert("c", 3);
        queue.upsert("d", 4);
        queue.upsert("e", 5);

        assert_eq!(queue.len(), 2, "Queue should maintain capacity");

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"e", 5), (&"d", 4)]);
    }

    #[test]
    fn test_upsert_returns_evicted_item() {
        // Both lookup strategies must report the displaced item and keep
        // lookups consistent afterwards.
        for linear in [true, false] {
            let mut queue =
                TopKQueue::with_capacity_hasher_linear(2, SipState::random(), linear);
            assert_eq!(queue.upsert("a", 1), None);
            assert_eq!(queue.upsert("b", 2), None);
            // Full queue, does not beat the min: rejected, nothing evicted.
            assert_eq!(queue.upsert("c", 1), None);
            // Full queue, beats the min: "a" is displaced and returned.
            assert_eq!(queue.upsert("c", 3), Some("a"));
            assert!(!queue.contains(&"a"), "linear={linear}");
            assert_eq!(queue.get(&"c"), Some(3), "linear={linear}");
            assert_eq!(queue.get(&"b"), Some(2), "linear={linear}");
        }
    }

    // Regression: when the sequence counter wraps, live slots are renumbered
    // so count-tie ordering is preserved instead of the new item sorting first.
    #[test]
    fn test_sequence_wrap_preserves_tie_order() {
        let mut queue = TopKQueue::with_capacity_and_hasher(3, SipState::random());
        queue.upsert("first", 5);
        queue.upsert("second", 5);
        // Force the next assignment to hit the wrap point.
        queue.sequence = u32::MAX;
        queue.upsert("third", 5);

        // Insertion order among equal counts must survive the wrap.
        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"first", 5), (&"second", 5), (&"third", 5)]);
        // The counter was renumbered down to the live-slot range.
        assert_eq!(queue.sequence, 3);
    }

    #[test]
    fn test_repeated_updates() {
        let mut queue = TopKQueue::with_capacity_and_hasher(2, SipState::random());

        // Insert and update same item repeatedly
        for i in 1..100 {
            queue.upsert("a", i);
        }

        queue.upsert("b", 50);

        assert_eq!(queue.len(), 2);

        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![(&"a", 99), (&"b", 50)]);
    }

    #[test]
    fn test_heap_property() {
        let mut queue = TopKQueue::with_capacity_and_hasher(10, SipState::random());

        // Insert in reverse order to test heap maintenance
        for i in (0..=10).rev() {
            queue.upsert(format!("item{}", i), i as u64);
        }

        // Verify heap property: parent should be <= children for min-heap
        for i in 1..queue.heap.len() {
            let parent_idx = TopKQueue::<String>::parent(i);
            if parent_idx > 0 {
                // Skip root's parent
                let parent_count = queue.item_store[queue.heap[parent_idx] as usize].count;
                let child_count = queue.item_store[queue.heap[i] as usize].count;
                assert!(
                    parent_count <= child_count,
                    "Heap property violated: parent count {} at index {} is greater than child count {} at index {}",
                    parent_count,
                    parent_idx,
                    child_count,
                    i
                );
            }
        }

        // Verify items are stored in descending order (highest counts first)
        let items: Vec<_> = queue.iter().collect();
        for i in 0..items.len() - 1 {
            assert!(
                items[i].1 >= items[i + 1].1,
                "Items not properly ordered by count: {} before {}",
                items[i].1,
                items[i + 1].1
            );
        }
    }
}
