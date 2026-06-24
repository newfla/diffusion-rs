pub mod api;
pub mod bridge;
pub mod gui_params;

// frb_generated is produced by flutter_rust_bridge_codegen.
// Until codegen runs, we provide a minimal stub so the crate compiles.
mod frb_generated;

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}
