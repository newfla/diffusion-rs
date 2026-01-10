use std::path::PathBuf;

use chrono::Local;
use clap::{Parser, ValueEnum};

use diffusion_rs::{
    api::{PreviewType, gen_img},
    preset::{
        ChromaRadianceWeight, ChromaWeight, DiffInstructStarWeight, Flux1MiniWeight, Flux1Weight,
        Flux2Weight, NitroSDRealismWeight, NitroSDVibrantWeight, OvisImageWeight, Preset,
        PresetBuilder, PresetDiscriminants, QwenImageWeight, SSD1BWeight,
        TwinFlowZImageTurboExpWeight, WeightType, ZImageTurboWeight,
    },
    util::set_hf_token,
};
use execution_time::ExecutionTime;
use strum::VariantNames;

macro_rules! clap_enum_variants {
    ($e: ty) => {{
        use clap::builder::TypedValueParser;
        clap::builder::PossibleValuesParser::new(<$e>::VARIANTS).map(|s| s.parse::<$e>().unwrap())
    }};
}

#[derive(Clone, Debug, ValueEnum)]
enum PreviewMode {
    Fast,
    Accurate,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// The prompt to render
    prompt: String,

    /// Negative prompt
    #[arg(short, long)]
    negative: Option<String>,

    /// The preset to use
    #[arg(short, long, ignore_case = true, value_parser = clap_enum_variants!(PresetDiscriminants))]
    preset: PresetDiscriminants,

    /// Optionally which type of quantization to use
    #[arg(short, long, ignore_case = true, value_parser = clap_enum_variants!(WeightType))]
    weights: Option<WeightType>,

    /// Numer of inference steps
    #[arg(short, long)]
    steps: Option<i32>,

    /// Width
    #[arg(short, long)]
    width: Option<i32>,

    /// Height
    #[arg(short, long)]
    height: Option<i32>,

    /// Number of images to generate
    #[arg(short, long, default_value_t = 1)]
    batch: i32,

    /// Output Folder
    #[arg(short, long, default_value = "./")]
    output: PathBuf,

    /// Enable preview
    #[arg(short, long, ignore_case = true)]
    preview: Option<PreviewMode>,

    /// Set Huggingface Hub token
    #[arg(short, long)]
    token: Option<String>,

    /// Enable optimization for gpu with lower GB
    #[arg(short, long, default_value_t = false)]
    low_vram: bool,

    /// Enable Random Seed: different runs will produce different results
    #[arg(short, long, default_value_t = false)]
    random_seed: bool,
}

fn main() {
    let timer = ExecutionTime::start();
    let args = Args::parse();

    if let Some(token) = &args.token {
        set_hf_token(token);
    }

    let preset = get_preset(&args);
    let (file_name, preview_filename) = get_output_file_name(&args);

    println!();
    if args.batch > 1 {
        println!("Images will be saved to {:#?}", args.output)
    } else {
        println!("Image will be saved as {file_name:#?}");
    }
    if args.preview.is_some() {
        println!("Image preview between inference steps will be saved as {preview_filename:#?}");
    }
    println!();

    let (config, mut model_config) = PresetBuilder::default()
        .preset(preset)
        .prompt(&args.prompt)
        .with_modifier(move |(mut config, mut model_config)| {
            // atch request?
            if args.batch > 1 {
                config.batch_count(args.batch);
                config.output(args.output);
            } else {
                config.output(file_name);
            }

            if let Some(steps) = &args.steps {
                config.steps(*steps);
            }

            if args.random_seed {
                config.seed(-1);
            }

            if let Some(width) = args.width {
                config.width(width);
            }

            if let Some(height) = args.height {
                config.height(height);
            }

            if let Some(negative) = args.negative {
                config.negative_prompt(negative);
            }

            if args.low_vram {
                model_config
                    .clip_on_cpu(true)
                    .vae_tiling(true)
                    .offload_params_to_cpu(true);
            }

            match args.preview {
                Some(PreviewMode::Fast) => config.preview_mode(PreviewType::PREVIEW_PROJ),
                Some(PreviewMode::Accurate) => config.preview_mode(PreviewType::PREVIEW_VAE),
                None => config.preview_mode(PreviewType::PREVIEW_NONE),
            };
            config.preview_output(preview_filename);

            Ok((config, model_config))
        })
        .build()
        .unwrap();
    gen_img(&config, &mut model_config).unwrap();

    println!();
    timer.print_elapsed_time();
    println!();
}

fn get_output_file_name(args: &Args) -> (PathBuf, PathBuf) {
    let ts = Local::now().format("%Y.%m.%d-%H.%M.%S");
    let file_name = format!("output_{}.png", ts);
    let preview = format!("preview_output_{}.png", ts);
    (args.output.join(file_name), args.output.join(preview))
}

fn get_preset(args: &Args) -> Preset {
    let preset = match args.preset {
        PresetDiscriminants::StableDiffusion1_4 => Preset::StableDiffusion1_4,
        PresetDiscriminants::StableDiffusion1_5 => Preset::StableDiffusion1_5,
        PresetDiscriminants::StableDiffusion2_1 => Preset::StableDiffusion2_1,
        PresetDiscriminants::StableDiffusion3Medium => Preset::StableDiffusion3Medium,
        PresetDiscriminants::StableDiffusion3_5Medium => Preset::StableDiffusion3_5Medium,
        PresetDiscriminants::StableDiffusion3_5Large => Preset::StableDiffusion3_5Large,
        PresetDiscriminants::StableDiffusion3_5LargeTurbo => Preset::StableDiffusion3_5LargeTurbo,
        PresetDiscriminants::SDXLBase1_0 => Preset::SDXLBase1_0,
        PresetDiscriminants::SDTurbo => Preset::SDTurbo,
        PresetDiscriminants::SDXLTurbo1_0 => Preset::SDXLTurbo1_0,
        PresetDiscriminants::Flux1Dev => Preset::Flux1Dev(
            args.weights
                .unwrap_or_else(|| Flux1Weight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::Flux1Schnell => Preset::Flux1Schnell(
            args.weights
                .unwrap_or_else(|| Flux1Weight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::Flux1Mini => Preset::Flux1Mini(
            args.weights
                .unwrap_or_else(|| Flux1MiniWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::JuggernautXL11 => Preset::JuggernautXL11,
        PresetDiscriminants::Chroma => Preset::Chroma(
            args.weights
                .unwrap_or_else(|| ChromaWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::NitroSDRealism => Preset::NitroSDRealism(
            args.weights
                .unwrap_or_else(|| NitroSDRealismWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::NitroSDVibrant => Preset::NitroSDVibrant(
            args.weights
                .unwrap_or_else(|| NitroSDVibrantWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::DiffInstructStar => Preset::DiffInstructStar(
            args.weights
                .unwrap_or_else(|| DiffInstructStarWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::ChromaRadiance => Preset::ChromaRadiance(
            args.weights
                .unwrap_or_else(|| ChromaRadianceWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::SSD1B => Preset::SSD1B(
            args.weights
                .unwrap_or_else(|| SSD1BWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::Flux2Dev => Preset::Flux2Dev(
            args.weights
                .unwrap_or_else(|| Flux2Weight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::ZImageTurbo => Preset::ZImageTurbo(
            args.weights
                .unwrap_or_else(|| ZImageTurboWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::QwenImage => Preset::QwenImage(
            args.weights
                .unwrap_or_else(|| QwenImageWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::OvisImage => Preset::OvisImage(
            args.weights
                .unwrap_or_else(|| OvisImageWeight::default().into())
                .try_into()
                .unwrap(),
        ),
        PresetDiscriminants::DreamShaperXL2_1Turbo => Preset::DreamShaperXL2_1Turbo,
        PresetDiscriminants::TwinFlowZImageTurboExp => Preset::TwinFlowZImageTurboExp(
            args.weights
                .unwrap_or_else(|| TwinFlowZImageTurboExpWeight::default().into())
                .try_into()
                .unwrap(),
        ),
    };
    preset
}
