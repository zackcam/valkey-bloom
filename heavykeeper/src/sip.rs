//! Portable seeded hasher for sketch placement and lookups.

use siphasher::sip::SipHasher13;
use std::hash::{BuildHasher, Hash};

/// Seeded `BuildHasher` producing SipHash-1-3.
///
/// SipHash is a frozen specification: a given key produces identical output on
/// every architecture, endianness, and crate version. Sketch payloads hashed
/// with a given seed can therefore be serialized on one platform and restored
/// on another. (`ahash`, used previously, varies by CPU feature set and
/// release, which made serialized sketches non-portable.)
#[derive(Clone, Debug)]
pub struct SipState {
    k0: u64,
    k1: u64,
}

impl SipState {
    /// Derive both SipHash keys from `seed` with splitmix64 so the key halves
    /// are decorrelated even for small or sequential seeds.
    pub fn with_seed(seed: u64) -> Self {
        let mut state = seed;
        let k0 = splitmix64(&mut state);
        let k1 = splitmix64(&mut state);
        Self { k0, k1 }
    }

    /// Build from a random seed. For sketches that are never serialized (or
    /// whose seed is recorded elsewhere); seeded construction is the norm.
    pub fn random() -> Self {
        Self::with_seed(fastrand::u64(..))
    }

    /// Hash a single value. Inherent so call sites do not need the
    /// `BuildHasher` trait in scope.
    #[inline]
    pub fn hash_one<T: Hash>(&self, value: T) -> u64 {
        BuildHasher::hash_one(self, value)
    }
}

impl BuildHasher for SipState {
    type Hasher = SipHasher13;

    #[inline]
    fn build_hasher(&self) -> SipHasher13 {
        SipHasher13::new_with_keys(self.k0, self.k1)
    }
}

/// splitmix64 step (public-domain constant sequence from Vigna's SplitMix64).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins portability: SipHash-1-3 with fixed keys is a frozen spec, so these
    // values must hold on every architecture, endianness, and crate version.
    // If this test ever fails, sketch hashing has silently changed and the
    // serialization VERSION must be bumped.
    #[test]
    fn test_hash_values_are_stable_and_portable() {
        assert_eq!(SipState::with_seed(0).hash_one(0u64), 7815940735901830603);
        assert_eq!(SipState::with_seed(42).hash_one(0u64), 4981565277819135878);
        assert_eq!(
            SipState::with_seed(42).hash_one(b"heavykeeper".as_slice()),
            4456609196736051103
        );
    }

    #[test]
    fn test_same_seed_same_hash_different_seed_different_hash() {
        let a = SipState::with_seed(7);
        let b = SipState::with_seed(7);
        let c = SipState::with_seed(8);
        assert_eq!(a.hash_one("x"), b.hash_one("x"));
        assert_ne!(a.hash_one("x"), c.hash_one("x"));
    }
}
