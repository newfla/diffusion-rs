# diffusion-rs
Rust bindings to https://github.com/leejet/stable-diffusion.cpp

## Features Matrix
| | Windows | Mac | Linux |
| --- | :---: | :---: | :---: |
|vulkan| ✅️ | ✅️ | ✅️ |
|metal| ❌️ | ✅️ | ❌️ |
|cuda| ✅️ | ❌️ | ✅️ |
|rocm| ❔️ | ❌️ | ✅️ |
|sycl| ❔️ | ❌️ | ✅️ |

❔️: Not tested, should be supported 

## Roadmap
1. ~~Ensure that the underline cpp library compiles on supported platforms~~
2. Build an easy to use library with models download and async interface
3. Automatic library publishing on crates.io by gh actions
4. _Maybe_ prebuilt CLI app binaries