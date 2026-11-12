//! Counts kept in the order each key was first counted, as Python's Counter keeps them.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts(Vec<(String, u64)>);

impl Counts {
    pub fn add(&mut self, key: &str, n: u64) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some((_, count)) => *count += n,
            None => self.0.push((key.to_string(), n)),
        }
    }

    /// `counter[key] = n`: the key keeps its place, or comes last when new.
    pub fn set(&mut self, key: &str, n: u64) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some((_, count)) => *count = n,
            None => self.0.push((key.to_string(), n)),
        }
    }

    pub fn get(&self, key: &str) -> u64 {
        self.0.iter().find(|(k, _)| k == key).map_or(0, |(_, n)| *n)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.0.iter().map(|(k, n)| (k.as_str(), *n))
    }

    /// `Counter.update(other)`: other's counts added, new keys after the existing ones.
    pub fn merge(&mut self, other: &Counts) {
        for (key, n) in other.iter() {
            self.add(key, n);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_first_counted_order() {
        let mut counts = Counts::default();
        counts.add("taken", 1);
        counts.add("skipped", 0);
        counts.add("taken", 2);
        assert_eq!(counts.iter().collect::<Vec<_>>(), [("taken", 3), ("skipped", 0)]);
        assert_eq!(counts.get("missing"), 0);
        let mut other = Counts::default();
        other.add("new", 1);
        other.add("taken", 1);
        counts.merge(&other);
        assert_eq!(counts.iter().collect::<Vec<_>>(), [("taken", 4), ("skipped", 0), ("new", 1)]);
    }
}
