use std::{
    env,
    fs::{self, create_dir_all, read_dir},
    path::{Path, PathBuf},
};

use cmake::Config;
use fs_extra::dir;

// Inspired by https://github.com/tazz4843/whisper-rs/blob/master/sys/build.rs

fn main() {
    // Link C++ standard library
    let target = env::var("TARGET").unwrap();
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={cpp_stdlib}");
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
            println!("cargo:warning=Unable to generate bindings: {e}");
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
    #[cfg(feature = "cuda")]
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

        config.define("SD_CUDA", "ON");
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
        config.define("SD_METAL", "ON");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }

    #[cfg(feature = "vulkan")]
    {
        let vulkan_path = env::var("VULKAN_SDK").map(|path| PathBuf::from(path));
        if target.contains("msvc") {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan-1");

            let vulkan_lib_path = vulkan_path
                .expect("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set")
                .join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else {
            if let Ok(vulkan_path) = vulkan_path {
                let vulkan_lib_path = vulkan_path.join("lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            }
            println!("cargo:rustc-link-lib=vulkan");
        }
        config.define("SD_VULKAN", "ON");
    }

    #[cfg(feature = "sycl")]
    {
        env::var("ONEAPI_ROOT").expect("Please load the oneAPi environment before building. See https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md");
        let sycl_lib_path = PathBuf::from(env::var("ONEAPI_ROOT").unwrap()).join("mkl/latest/lib");
        println!("cargo:rustc-link-search={}", sycl_lib_path.display());

        println!("cargo:rustc-link-lib=static=mkl_sycl");
        println!("cargo:rustc-link-lib=static=mkl_core");
        println!("cargo:rustc-link-lib=static=mkl_scalapack_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_intel_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_blacs_intelmpi_ilp64");
        println!("cargo:rustc-link-lib=static=mkl_tbb_thread");

        println!("cargo:rustc-link-lib=tbb");
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=svml");
        println!("cargo:rustc-link-lib=imf");
        println!("cargo:rustc-link-lib=intlc");
        println!("cargo:rustc-link-lib=ur_loader");
        println!("cargo:rustc-link-lib=m");
        println!("cargo-rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=sycl");
        println!("cargo:rustc-link-lib=dnnl");

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

    add_link_search_path(&out.join("lib")).unwrap();
    add_link_search_path(&out.join("build")).unwrap();
    add_link_search_path(&out).unwrap();

    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=stable-diffusion");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(feature = "cuda")]
    println!("cargo:rustc-link-lib=static=ggml-cuda");

    #[cfg(feature = "hipblas")]
    println!("cargo:rustc-link-lib=static=ggml-hip");

    #[cfg(feature = "metal")]
    println!("cargo:rustc-link-lib=static=ggml-metal");

    #[cfg(feature = "vulkan")]
    println!("cargo:rustc-link-lib=static=ggml-vulkan");

    #[cfg(feature = "sycl")]
    println!("cargo:rustc-link-lib=static=ggml-sycl");
}

fn add_link_search_path(dir: &Path) -> std::io::Result<()> {
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
