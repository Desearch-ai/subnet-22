//! The Python crawler's language identifier: py3langid's naive Bayes model, read from `tools/export_langid.py` output.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use unicode_normalization::UnicodeNormalization;

const MAGIC: &[u8; 8] = b"LANGID01";
/// The score of a class ruled out, as py3langid floors it.
const FLOOR: f32 = f32::MIN;

pub struct LangId {
    classes: Vec<String>,
    pc: Vec<f32>,
    ptc: Vec<f32>,
    nextmove: Vec<u32>,
    rowbase: Vec<usize>,
    out: Vec<i32>,
    aliases: Vec<(usize, usize)>,
}

impl LangId {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        LangId::parse(&data).with_context(|| format!("reading the language model in {}", path.display()))
    }

    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut at = 0;
        let mut take = |len: usize| -> Result<&[u8]> {
            let chunk = data.get(at..at + len).context("the model file is cut short")?;
            at += len;
            Ok(chunk)
        };
        if take(8)? != MAGIC {
            bail!("not a model written by tools/export_langid.py");
        }
        let header: Vec<usize> = take(16)?.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
        let (features, classes, moves, states) = (header[0], header[1], header[2], header[3]);
        let classes: Vec<String> =
            take(classes * 4)?.chunks(4).map(|c| String::from_utf8_lossy(c).trim_end_matches('\0').to_string()).collect();
        let pc = take(classes.len() * 4)?.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let ptc = take(features * classes.len() * 2)?.chunks(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        let nextmove = take(moves * 4)?.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let rowbase = take(states * 2)?.chunks(2).map(|c| (u16::from_le_bytes([c[0], c[1]]) as usize) << 8).collect();
        let out = take(states * 4)?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut first: HashMap<&str, usize> = HashMap::new();
        let mut aliases = Vec::new();
        for (i, code) in classes.iter().enumerate() {
            match first.get(code.as_str()) {
                Some(&j) => aliases.push((j, i)),
                None => {
                    first.insert(code, i);
                }
            }
        }
        Ok(LangId { classes, pc, ptc, nextmove, rowbase, out, aliases })
    }

    /// The most likely language, as `py3langid.classify(text)[0]` gives it.
    pub fn classify(&self, text: &str) -> &str {
        let lowered;
        let text = if is_upper(text) {
            lowered = text.to_lowercase();
            &lowered
        } else {
            text
        };
        let normalized: String = text.nfc().collect();
        let mut order: Vec<usize> = Vec::new();
        let mut counts: HashMap<usize, u32> = HashMap::new();
        let mut state = 0;
        for &byte in normalized.as_bytes() {
            state = self.nextmove[self.rowbase[state] + byte as usize] as usize;
            if let Ok(feature) = usize::try_from(self.out[state]) {
                let count = counts.entry(feature).or_default();
                if *count == 0 {
                    order.push(feature);
                }
                *count += 1;
            }
        }
        let n = self.classes.len();
        let mut scores = vec![FLOOR; n];
        if !order.is_empty() {
            let mut sums = vec![0f32; n];
            for feature in order {
                let weight = (counts[&feature] as f32).ln_1p();
                for (sum, p) in sums.iter_mut().zip(&self.ptc[feature * n..(feature + 1) * n]) {
                    *sum += weight * p;
                }
            }
            for ((score, sum), prior) in scores.iter_mut().zip(sums).zip(&self.pc) {
                *score = sum + prior;
            }
        }
        for &(i, j) in &self.aliases {
            scores[i] = scores[i].max(scores[j]);
            scores[j] = FLOOR;
        }
        let best = scores.iter().enumerate().fold(0, |best, (i, &s)| if s > scores[best] { i } else { best });
        &self.classes[best]
    }
}

/// Python's `str.isupper`: some cased character, and none lowercase or titlecase.
fn is_upper(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        let title = matches!(c, '\u{1c5}' | '\u{1c8}' | '\u{1cb}' | '\u{1f2}' | '\u{1f88}'..='\u{1f8f}' | '\u{1f98}'..='\u{1f9f}'
            | '\u{1fa8}'..='\u{1faf}' | '\u{1fbc}' | '\u{1fcc}' | '\u{1ffc}');
        if c.is_lowercase() || title {
            return false;
        }
        cased |= c.is_uppercase();
    }
    cased
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let value = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            let (mut e, mut m) = (113u32, mantissa);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => sign | 0x7fc0_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_widen_exactly() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x0200), 2f32.powi(-15));
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert!(half_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn upper_is_pythons() {
        assert!(is_upper("HELLO WORLD 42") && !is_upper("Hello") && !is_upper("123") && !is_upper("ǅA"));
    }
}
