# diffusion-rs
[![Latest version](https://img.shields.io/crates/v/diffusion-rs.svg)](https://crates.io/crates/diffusion-rs)
[![Documentation](https://docs.rs/diffusion-rs/badge.svg)](https://docs.rs/diffusion-rs)

Rust bindings to <https://github.com/leejet/stable-diffusion.cpp>

## Features Matrix
| | Windows | Mac | Linux |
| --- | :---: | :---: | :---: |
|vulkan| ✅️ | ✅️ | ✅️ |
|metal| ❌️ | ✅️ | ❌️ |
|cuda| ✅️ | ❌️ | ✅️ |
|rocm| ❔️ | ❌️ | ✅️ |
|sycl| ❔️ | ❌️ | ✅️ |

❔️: Not tested, should be supported 

## Usage 
``` rust no_run
use diffusion_rs::{api::txt2img, preset::{Preset,PresetBuilder}};
let config = PresetBuilder::default()
            .preset(Preset::SDXLTurbo1_0Fp16)
            .prompt("a lovely duck drinking water from a bottle")
            .build()
            .unwrap();
txt2img(config).unwrap();
```

## Troubleshooting

* Something other than Windows/Linux isn't working!
    * I don't have a way to test these platforms, so I can't really help you.
* I get a panic during binding generation build!
    * You can attempt to fix it yourself, or you can set the `DIFFUSION_SKIP_BINDINGS` environment variable.
      This skips attempting to build the bindings whatsoever and copies the existing ones. They may be out of date,
      but it's better than nothing.
        * `DIFFUSION_SKIP_BINDINGS=1 cargo build`
    * If you can fix the issue, please open a PR!

## Roadmap
1. ~~Ensure that the underline cpp library compiles on supported platforms~~
2. ~~Build an easy to use library with model presets~~
3. Automatic library publishing on crates.io by gh actions
4. _Maybe_ prebuilt CLI app binaries