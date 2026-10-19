//! The embedding models embed tasks name, and the vectors miners upload for them.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Model {
    pub name: &'static str,
    pub dims: u64,
}

pub const DESEARCH_4B_V1: Model = Model { name: "desearch-4b-v1", dims: 2560 };
pub const MODELS: [Model; 1] = [DESEARCH_4B_V1];
/// Each value of a vector is a little-endian float16.
pub const VALUE_BYTES: u64 = 2;
pub const VECTORS_SUFFIX: &str = ".f16";
pub const VECTORS_TYPE: &str = "application/octet-stream";

pub fn model(name: &str) -> Option<Model> {
    MODELS.into_iter().find(|model| model.name == name)
}

/// The size of an upload holding one vector for each of `texts` texts.
pub fn upload_bytes(model: Model, texts: u64) -> u64 {
    texts * model.dims * VALUE_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_upload_holds_one_float16_vector_per_text() {
        assert_eq!(upload_bytes(model("desearch-4b-v1").unwrap(), 3), 3 * 5_120);
        assert_eq!(model("qwen3-embedding-8b"), None);
    }
}
