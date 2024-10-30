use std::{
    env,
    fs::{self, create_dir_all, read_dir},
    path::{Path, PathBuf},
};

use cmake::Config;
use fs_extra::dir;

// Heavily ispired by https://github.com/tazz4843/whisper-rs/blob/master/sys/build.rs

fn main() {
    // Link C++ standard library
    let target = env::var("TARGET").unwrap();
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
    }

    println!("cargo:rerun-if-changed=wrapper.h");

    // Copy stable-diffusion code into the build script directory
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let diffusion_root = out.join("stable-diffusion.cpp/");
    let stb_write_image_src = diffusion_root.join("thirdparty/stb_image_write.c");

    if !diffusion_root.exists() {
        create_dir_all(&diffusion_root).unwrap();
        dir::copy("./stable-diffusion.cpp", &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy stable-diffusion sources into {}: {}",
                diffusion_root.display(),
                e
            )
        });
        fs::copy("./stb_image_write.c", &stb_write_image_src).unwrap_or_else(|e| {
            panic!(
                "Failed to copy stb_image_write to {}: {}",
                stb_write_image_src.display(),
                e
            )
        });

        remove_default_params_stb(&diffusion_root.join("thirdparty/stb_image_write.h"))
            .unwrap_or_else(|e| panic!("Failed to remove default parameters from stb: {}", e));
    }

    // Bindgen
    if env::var("DIFFUSION_SKIP_BINDINGS").is_ok() {
        fs::copy("src/bindings.rs", out.join("bindings.rs")).expect("Failed to copy bindings.rs");
    } else {
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg("-I./stable-diffusion.cpp")
            .clang_arg("-I./stable-diffusion.cpp/ggml/include")
            .rustified_non_exhaustive_enum(".*")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .unwrap()
            .write_to_file(out.join("bindings.rs"));

        if let Err(e) = bindings {
            println!("cargo:warning=Unable to generate bindings: {}", e);
            println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
            // copy src/bindings.rs to OUT_DIR
            fs::copy("src/bindings.rs", out.join("bindings.rs"))
                .expect("Unable to copy bindings.rs");
        }
    }

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // Configure cmake for building
    let mut config = Config::new(&diffusion_root);

    //Enable cmake feature flags
    #[cfg(feature = "cublas")]
    {
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");

        if target.contains("msvc") {
            let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap()).join("lib/x64");
            println!("cargo:rustc-link-search={}", cuda_path.display());
        } else {
            println!("cargo:rustc-link-lib=culibos");
            println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
            println!("cargo:rustc-link-search=/opt/cuda/lib64");
            println!("cargo:rustc-link-search=/opt/cuda/lib64/stubs");
        }

        config.define("SD_CUBLAS", "ON");
        if let Ok(target) = env::var("CUDA_COMPUTE_CAP") {
            config.define("CUDA_COMPUTE_CAP", target);
        }
    }

    #[cfg(feature = "hipblas")]
    {
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");

        config.generator("Ninja");
        config.define("CMAKE_C_COMPILER", "clang");
        config.define("CMAKE_CXX_COMPILER", "clang++");
        let hip_lib_path = if target.contains("msvc") {
            let hip_path = env::var("HIP_PATH").expect("Missing HIP_PATH env variable");
            PathBuf::from(hip_path).join("lib")
        } else {
            let hip_path = match env::var("HIP_PATH") {
                Ok(path) => PathBuf::from(path),
                Err(_) => PathBuf::from("/opt/rocm"),
            };
            hip_path.join("lib")
        };
        println!("cargo:rustc-link-search={}", hip_lib_path.display());
        config.define("SD_HIPBLAS", "ON");
        if let Ok(target) = env::var("AMDGPU_TARGETS") {
            config.define("AMDGPU_TARGETS", target);
        }
    }

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        config.define("SD_METAL", "ON");
    }

    #[cfg(feature = "vulkan")]
    {
        if target.contains("msvc") {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan-1");
            let vulkan_path = match env::var("VULKAN_SDK") {
                Ok(path) => PathBuf::from(path),
                Err(_) => panic!(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                ),
            };
            let vulkan_lib_path = vulkan_path.join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=vulkan");
        }
        config.define("SD_VULKAN", "ON");
    }

    #[cfg(feature = "sycl")]
    {
        env::var("ONEAPI_ROOT").expect("Please load the oneAPi environment before building. See https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md");

        if target.contains("msvc") {
            config.generator("Ninja");
            config.define("CMAKE_C_COMPILER", "cl");
            config.define("CMAKE_CXX_COMPILER", "icx");
        } else {
            config.define("CMAKE_C_COMPILER", "icx");
            config.define("CMAKE_CXX_COMPILER", "icpx");
        }
        config.define("SD_SYCL", "ON");
    }

    #[cfg(feature = "flashattn")]
    {
        config.define("SD_FLASH_ATTN", "ON");
    }

    // Build stable-diffusion
    config
        .profile("Release")
        .define("SD_BUILD_SHARED_LIBS", "OFF")
        .define("SD_BUILD_EXAMPLES", "OFF")
        .define("GGML_OPENMP", "OFF")
        .very_verbose(true)
        .pic(true);

    let destination = config.build();

    // Build stb write image
    let mut builder = cc::Build::new();

    builder.file(stb_write_image_src).compile("stbwriteimage");

    add_link_search_path(&out.join("lib")).unwrap();
    add_link_search_path(&out.join("build")).unwrap();
    add_link_search_path(&out).unwrap();

    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=stable-diffusion");
    println!("cargo:rustc-link-lib=static=ggml");
}

fn add_link_search_path(dir: &std::path::Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        for entry in read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}

fn remove_default_params_stb(file: &Path) -> std::io::Result<()> {
    let data = fs::read_to_string(file)?;
    let new_data = data.replace("const char* parameters = NULL", "const char* parameters");
    fs::write(file, new_data)
}
