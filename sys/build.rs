use std::{
    env,
    fs::{create_dir_all, read_dir},
    path::PathBuf,
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

    if !diffusion_root.exists() {
        create_dir_all(&diffusion_root).unwrap();
        dir::copy("./stable-diffusion.cpp", &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy stable-diffusion sources into {}: {}",
                diffusion_root.display(),
                e
            )
        });
    }

    // Bindgen
    let bindings = bindgen::Builder::default().header("wrapper.h");

    bindings
        .clang_arg("-I./stable-diffusion.cpp")
        .clang_arg("-I./stable-diffusion.cpp/ggml/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap()
        .write_to_file(out.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // Configure cmake for building
    let mut config = Config::new(&diffusion_root);

    //Enable cmake feature flags
    #[cfg(feature = "cublas")]
    {
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
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");

        if target.contains("msvc") {
            panic!("Due to a problem with the last revision of the ROCm 5.7 library, it is not possible to compile the library for the windows environment.\nSee https://github.com/ggerganov/whisper.cpp/issues/2202 for more details.")
        } else {
            println!("cargo:rerun-if-env-changed=HIP_PATH");

            let hip_path = match env::var("HIP_PATH") {
                Ok(path) => PathBuf::from(path),
                Err(_) => PathBuf::from("/opt/rocm"),
            };
            let hip_lib_path = hip_path.join("lib");

            println!("cargo:rustc-link-search={}", hip_lib_path.display());
        }

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
        config.define("SD_SYCL", "ON");
        panic!("Not yet supported!");
    }

    #[cfg(feature = "flashattn")]
    {
        config.define("SD_FLASH_ATTN", "ON");
        panic!("Broken in 2024/09/02 release!");
    }

    config
        .profile("Release")
        .define("SD_BUILD_SHARED_LIBS", "OFF")
        .define("SD_BUILD_EXAMPLES", "OFF")
        .very_verbose(true)
        .pic(true);

    let destination = config.build();

    add_link_search_path(&out.join("lib")).unwrap();
    add_link_search_path(&out.join("build")).unwrap();

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
