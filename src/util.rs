use std::{path::PathBuf, sync::OnceLock};

use hf_hub::api::sync::{ApiBuilder, ApiError};

static TOKEN: OnceLock<String> = OnceLock::new();

/// Set the huggingface hub token to access "protected" models. See <https://huggingface.co/settings/tokens>
pub fn set_hf_token(token: &str) {
    let _ = TOKEN.set(token.to_owned());
}

pub(crate) fn download_file_hf_hub(model: &str, file: &str) -> Result<PathBuf, ApiError> {
    let token = TOKEN.get().map(|token| token.to_owned());
    let repo = ApiBuilder::new()
        .with_token(token)
        .build()?
        .model(model.to_string());
    repo.get(file)
}
