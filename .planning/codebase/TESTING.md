---
title: Testing
last_mapped: 2026-06-18
---

# Testing

## Framework

- **Test runner:** Rust built-in (`cargo test`) — no external runner
- **Assertion style:** Standard `assert!` / `assert_eq!` macros
- **No mocking framework** — tests invoke real C++ library calls, or are `#[ignore]`d when models are unavailable
- **No fixtures library** — test setup is inline helper functions

## Test Organization

Tests are **co-located** with source code using Rust's `#[cfg(test)]` module pattern:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    // ...
}
```

Files with test modules:
- `src/preset.rs` — 39 tests (all `#[ignore]`)
- `src/modifier.rs` — 22 tests (all `#[ignore]`, plus a few non-ignored)
- `src/api.rs` — 3 tests

No separate `tests/` integration test directory exists.

## Test Structure

### Preset Tests (`src/preset.rs`)
All tests are `#[ignore]` because they require:
1. Real model weights downloaded from HuggingFace
2. A valid HuggingFace token in `token.txt` for gated models

Pattern:
```rust
fn run(preset: Preset) {
    let mut model_config = ModelConfigBuilder::default().preset(preset).build().unwrap();
    let config = GenerationParamsBuilder::default()
        .prompt(PROMPT)
        .build()
        .unwrap();
    gen_img(&config, &mut model_config).unwrap();
}

#[ignore]
#[test]
fn test_flux_1_schnell() {
    set_hf_token(include_str!("../token.txt"));
    run(Preset::Flux1Schnell(Flux1Weight::Q2_K));
}
```

Each test exercises one specific model preset end-to-end (model load → inference → image save).

### Modifier Tests (`src/modifier.rs`)
Same pattern — require real model weights. Test combinations of modifiers applied to a base preset.

### API Tests (`src/api.rs`)
3 tests — details require closer inspection but likely test FFI boundary behavior.

## How to Run Tests

```bash
# Run all non-ignored tests (fast, no model weights needed)
cargo test

# Run a specific ignored test (requires model + token)
cargo test test_flux_1_schnell -- --ignored

# Run all tests including ignored (requires all models)
cargo test -- --include-ignored

# Run tests with a GPU feature
cargo test --features metal
cargo test --features cuda
cargo test --features vulkan
```

## CI Test Strategy

Defined in `.github/workflows/test.yml`:

1. **Formatting gate** (`cargo-fmt` job) — runs first on Ubuntu, blocks all other jobs
2. **Build + test matrix** (`build-no-features`) — Ubuntu, macOS, Windows; runs `cargo test` (non-ignored only)
3. **Feature builds** — `build-vulkan`, `build-metal`, `build-cuda`, `build-rocm` — compile-only verification for GPU backends; only Vulkan on Linux and Metal on macOS also run `cargo test`
4. SYCL build is commented out in CI

## Coverage

- **~64 total tests** across the library, all integration-style (no unit tests with mocks)
- **~61 are `#[ignore]`d** — only run manually when model weights are present
- **~3 non-ignored tests** run in CI — very low automated test coverage
- No coverage tooling (no `cargo-tarpaulin`, `cargo-llvm-cov`, etc.) configured

## Key Gaps

- No non-ignored unit tests for builder validation (field defaults, error cases)
- No integration tests in `tests/` directory
- No property-based testing
- `token.txt` must exist at repo root for HF-gated model tests (even when `#[ignore]`d, `include_str!` is evaluated at compile time)
