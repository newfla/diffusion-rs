use std::{
    path::PathBuf,
    sync::{OnceLock, RwLock},
};

use hf_hub::api::sync::{ApiBuilder, ApiError};

static TOKEN: OnceLock<RwLock<String>> = OnceLock::new();

/// Set the huggingface hub token to access "protected" models. See <https://huggingface.co/settings/tokens>
pub fn set_hf_token(token: &str) {
    let guard = TOKEN.get_or_init(|| RwLock::new(Default::default()));
    let mut data = guard.write().unwrap();
    *data = token.to_owned();
}

pub(crate) fn download_file_hf_hub(model: &str, file: &str) -> Result<PathBuf, ApiError> {
    let token = TOKEN.get().map(|token| token.read().unwrap().to_owned());
    let repo = ApiBuilder::new()
        .with_token(token)
        .build()?
        .model(model.to_string());
    repo.get(file)
}
