//! What every bulk source shares: v2 text conventions, upload-shaped rows, run folders, HTTP and run diffs.

pub mod counts;
pub mod http;
pub mod rows;
pub mod run;
pub mod runs;
pub mod text;

pub use counts::Counts;
pub use rows::{authors_line, paper_page, upload_row, Batches, Page, PaperMeta, Row};
pub use run::RunArgs;
pub use text::Line;
