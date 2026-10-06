//! The SN22 publisher: uploads that passed validation into the numbered change feed and the version index.

pub mod blocked;
pub mod canonical;
pub mod changes;
pub mod entities;
pub mod index;
pub mod local;
pub mod outcomes;
pub mod reading;
pub mod records;
pub mod text;
pub mod worker;
