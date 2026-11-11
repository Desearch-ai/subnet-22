//! Counts kept in the order each key was first counted, as Python's Counter keeps them.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts(Vec<(String, u64)>);

impl Counts {
    pub fn add(&mut self, key: &str, n: u64) {
        todo!("{key} {n}")
    }

    pub fn get(&self, key: &str) -> u64 {
        todo!("{key}")
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.0.iter().map(|(k, n)| (k.as_str(), *n))
    }
}
