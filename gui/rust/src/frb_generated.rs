// Stub module: replaced by flutter_rust_bridge_codegen generate.
// Provides placeholder types so the crate compiles before FRB codegen runs.

/// Placeholder StreamSink that will be replaced by FRB codegen.
/// Implements the same interface shape so api.rs compiles.
pub struct StreamSink<T> {
    _marker: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
impl<T> StreamSink<T> {
    /// Emit a value to the Dart stream.
    pub fn add(&self, _value: T) -> anyhow::Result<()> {
        Ok(())
    }

    /// Emit an error to the Dart stream.
    pub fn add_error(&self, _error: anyhow::Error) -> anyhow::Result<()> {
        Ok(())
    }
}

// StreamSink must be Clone so it can be shared between threads
impl<T> Clone for StreamSink<T> {
    fn clone(&self) -> Self {
        StreamSink {
            _marker: std::marker::PhantomData,
        }
    }
}
