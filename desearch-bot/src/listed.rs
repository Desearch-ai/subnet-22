//! What one sitemap listed when it was last read: a Golomb-coded set of entry fingerprints, about 4.2 bytes per entry.

use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest};

/// Bits in each remainder; a fingerprint outside the set matches by chance once in 2^32 lookups.
const P: u32 = 32;

/// One entry as the store records it: its key and its lastmod.
pub fn fingerprint(key: &[u8], lastmod: u32) -> u64 {
    let mut hash = Blake2b::<U8>::new();
    hash.update(key);
    hash.update(lastmod.to_le_bytes());
    u64::from_le_bytes(hash.finalize().into())
}

#[derive(Debug, PartialEq, Eq)]
pub struct ListedSet {
    range: u64,
    values: Vec<u64>,
}

impl ListedSet {
    pub fn encode(fingerprints: &[u64]) -> Vec<u8> {
        let range = (fingerprints.len() as u64) << P;
        let mut values: Vec<u64> = fingerprints.iter().map(|&f| reduce(f, range)).collect();
        values.sort_unstable();
        values.dedup();
        let mut out = Vec::with_capacity(values.len() * 17 / 4 + 16);
        varint(&mut out, fingerprints.len() as u64);
        varint(&mut out, values.len() as u64);
        let mut bits = BitWriter { bytes: out, acc: 0, filled: 0 };
        let mut previous = 0;
        for value in values {
            let delta = value - previous;
            previous = value;
            bits.unary(delta >> P);
            bits.push(delta & ((1 << P) - 1), P);
        }
        bits.finish()
    }

    pub fn decode(data: &[u8]) -> Option<ListedSet> {
        let mut at = 0;
        let entries = read_varint(data, &mut at)?;
        let count = read_varint(data, &mut at)?;
        let mut bits = BitReader { bytes: &data[at..], at: 0, acc: 0, filled: 0 };
        let mut values = Vec::with_capacity(count.min(1 << 24) as usize);
        let mut previous = 0u64;
        for _ in 0..count {
            let quotient = bits.unary()?;
            let remainder = bits.take(P)?;
            previous = previous.checked_add(quotient.checked_mul(1 << P)? | remainder)?;
            values.push(previous);
        }
        Some(ListedSet { range: entries << P, values })
    }

    pub fn contains(&self, fingerprint: u64) -> bool {
        self.values.binary_search(&reduce(fingerprint, self.range)).is_ok()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Map a 64-bit hash evenly onto 0..range without a division.
fn reduce(hash: u64, range: u64) -> u64 {
    ((hash as u128 * range as u128) >> 64) as u64
}

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(data: &[u8], at: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *data.get(*at)?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Some(value);
        }
    }
    None
}

struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    filled: u32,
}

impl BitWriter {
    fn push(&mut self, value: u64, bits: u32) {
        self.acc |= value << self.filled;
        self.filled += bits;
        while self.filled >= 8 {
            self.bytes.push(self.acc as u8);
            self.acc >>= 8;
            self.filled -= 8;
        }
    }

    fn unary(&mut self, mut ones: u64) {
        while ones >= 32 {
            self.push(u32::MAX as u64, 32);
            ones -= 32;
        }
        self.push((1 << ones) - 1, ones as u32 + 1);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.bytes.push(self.acc as u8);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    at: usize,
    acc: u64,
    filled: u32,
}

impl BitReader<'_> {
    fn refill(&mut self) {
        while self.filled <= 56 && self.at < self.bytes.len() {
            self.acc |= u64::from(self.bytes[self.at]) << self.filled;
            self.at += 1;
            self.filled += 8;
        }
    }

    fn consume(&mut self, bits: u32) {
        self.acc = if bits >= 64 { 0 } else { self.acc >> bits };
        self.filled -= bits;
    }

    fn take(&mut self, bits: u32) -> Option<u64> {
        self.refill();
        if self.filled < bits {
            return None;
        }
        let value = self.acc & ((1u64 << bits) - 1);
        self.consume(bits);
        Some(value)
    }

    fn unary(&mut self) -> Option<u64> {
        let mut ones = 0;
        loop {
            self.refill();
            if self.filled == 0 {
                return None;
            }
            let run = self.acc.trailing_ones().min(self.filled);
            ones += u64::from(run);
            if run < self.filled {
                self.consume(run + 1);
                return Some(ones);
            }
            self.consume(run);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn members_are_found_and_strangers_are_not() {
        let members: Vec<u64> = (0..50_000u32).map(|i| fingerprint(format!("example.com\0example.com/item-{i}").as_bytes(), i)).collect();
        let encoded = ListedSet::encode(&members);
        assert!(encoded.len() < members.len() * 9 / 2, "{} bytes for {} entries", encoded.len(), members.len());
        let set = ListedSet::decode(&encoded).unwrap();
        assert_eq!(set.len(), members.len());
        assert!(members.iter().all(|&m| set.contains(m)));
        let strangers = (0..200_000u32).filter(|&i| set.contains(fingerprint(format!("example.com\0example.com/other-{i}").as_bytes(), i)));
        assert_eq!(strangers.count(), 0);
        assert!(!set.contains(fingerprint(b"example.com\0example.com/item-7", 8)), "a new lastmod is a new entry");
    }

    #[test]
    fn small_and_empty_sets_round_trip() {
        for size in [0usize, 1, 2, 3, 40] {
            let members: Vec<u64> = (0..size as u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
            let set = ListedSet::decode(&ListedSet::encode(&members)).unwrap();
            assert_eq!(set.len(), size);
            assert!(members.iter().all(|&m| set.contains(m)));
        }
        assert!(ListedSet::decode(&[0x80]).is_none());
    }
}
