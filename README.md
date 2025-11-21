# diffusion-rs
[![Latest version](https://img.shields.io/crates/v/diffusion-rs.svg)](https://crates.io/crates/diffusion-rs)
[![Documentation](https://docs.rs/diffusion-rs/badge.svg)](https://docs.rs/diffusion-rs)

Rust bindings to <https://github.com/leejet/stable-diffusion.cpp>

## Features Matrix
| | Windows | Mac | Linux |
| --- | :---: | :---: | :---: |
|vulkan| ❌ | ⛓️‍💥 | ✅️ |
|metal| - | ✅️ | - |
|cuda| ❌ | - | ✅️ |
|rocm| ❌ | - | ⛓️‍💥 |
|sycl| ❌ | - | ✅️ |

✅️: Working

❌: See this [cargo issue](https://github.com/rust-lang/cargo/issues/15137)

⛓️‍💥 : Issues when linking libraries

## Usage 
``` rust no_run
use diffusion_rs::{api::gen_img, preset::{Preset,PresetBuilder}};
let (mut config, mut model_config) = PresetBuilder::default()
            .preset(Preset::SDXLTurbo1_0Fp16)
            .prompt("a lovely duck drinking water from a bottle")
            .build()
            .unwrap();
gen_img(config, model_config).unwrap();
```

## Troubleshooting

* Something other than Windows/Linux isn't working!
    * I don't have a way to test these platforms, so I can't really help you.
    * If you can fix the issue, please open a PR!

## Roadmap
1. ~~Ensure that the underline cpp library compiles on supported platforms~~
2. ~~Build an easy to use library with model presets~~
3. ~~Automatic library publishing on crates.io by gh actions~~
4. _Maybe_ prebuilt CLI app binaries