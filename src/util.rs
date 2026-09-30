use std::{
    env,
    path::PathBuf,
    sync::{OnceLock, RwLock},
};

use hf_hub::{
    HFClientBuilder, HFError,
    progress::{DownloadEvent, ProgressEvent, ProgressHandler},
};
use human_units::FormatSize;

static TOKEN: OnceLock<RwLock<String>> = OnceLock::new();

/// Set the huggingface hub token to access "protected" models. See <https://huggingface.co/settings/tokens>
pub fn set_hf_token(token: &str) {
    let guard = TOKEN.get_or_init(|| RwLock::new(Default::default()));
    let mut data = guard.write().unwrap();
    *data = token.to_owned();
}

/// Download file from huggingface hub
pub fn download_file_hf_hub(repo: &str, file: &str) -> Result<PathBuf, HFError> {
    let (owner, repo) = repo.split_once("/").unwrap_or_default();
    let mut hf_client =
        if let Some(token) = TOKEN.get().map(|token| token.read().unwrap().to_owned()) {
            HFClientBuilder::new().token(token)
        } else {
            HFClientBuilder::new()
        };
    if let Some(home_dir) = env::home_dir() {
        let cache_dir = home_dir.join(".cache/huggingface/hub");
        hf_client = hf_client.cache_dir(cache_dir);
    }
    hf_client
        .build_sync()?
        .model(owner, repo)
        .download_file()
        .filename(file)
        .progress(PrintProgressHandler(repo.to_string(), file.to_string()))
        .send()
}

struct PrintProgressHandler(String, String);

impl ProgressHandler for PrintProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(dl) = event {
            match dl {
                DownloadEvent::Start {
                    total_files: _,
                    total_bytes,
                } => {
                    println!(
                        "\nStarting download: {}/{}, {}",
                        self.0,
                        self.1,
                        (*total_bytes).format_size()
                    );
                }
                DownloadEvent::Progress { files } => {
                    for f in files {
                        let pct = (f.bytes_completed * 100)
                            .checked_div(f.total_bytes)
                            .unwrap_or(0);
                        print!("\r");
                        print!(
                            "  {}: {pct}% ({}/{})",
                            format_args!("{}/{}", self.0, self.1),
                            f.bytes_completed.format_size(),
                            f.total_bytes.format_size()
                        );
                    }
                }
                DownloadEvent::Complete => {
                    println!("Download complete.");
                }
                _ => {}
            }
        }
    }
}
