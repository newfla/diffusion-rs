# diffusion-rs
[![Latest version](https://img.shields.io/crates/v/diffusion-rs.svg)](https://crates.io/crates/diffusion-rs)
[![Documentation](https://docs.rs/diffusion-rs/badge.svg)](https://docs.rs/diffusion-rs)

Rust bindings to <https://github.com/leejet/stable-diffusion.cpp>

## Features Matrix
| | Windows | Mac | Linux |
| --- | :---: | :---: | :---: |
|vulkan| ✅️ | ✅️ | ✅️ |
|metal| - | ✅️ | - |
|cuda| ✅️ | - | ✅️ |
|rocm| ❔ | - | ⛓️‍💥 |
|sycl| ❔ | - | ✅️ |

✅️: Working

❔: Not tested

❌: See this [cargo issue](https://github.com/rust-lang/cargo/issues/15137)

⛓️‍💥 : Issues when linking libraries

## Usage 
``` rust no_run
use diffusion_rs::{api::gen_img, preset::{Preset,PresetBuilder}};
let (config, mut model_config) = PresetBuilder::default()
            .preset(Preset::SDXLTurbo1_0Fp16)
            .prompt("a lovely duck drinking water from a bottle")
            .build()
            .unwrap();
gen_img(&config, &mut model_config).unwrap();
```

## Troubleshooting

* Something other than Windows/Linux isn't working!
    * I don't have a way to test these platforms, so I can't really help you.
    * If you can fix the issue, please open a PR!
