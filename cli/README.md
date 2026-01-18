# Diffusion-RS CLI

## Usage
```
cli.exe --preset flux1schnell --random-seed "a lovely duck holding a sign says 'drink your water'"
```
## Help
```
Usage: cli.exe [OPTIONS] --preset <PRESET> <PROMPT>

Arguments:
  <PROMPT>  The prompt to render

Options:
  -n, --negative <NEGATIVE>  Negative prompt
  -p, --preset <PRESET>      The preset to use [possible values: StableDiffusion1_4, StableDiffusion1_5, StableDiffusion2_1, StableDiffusion3Medium, StableDiffusion3_5Medium, StableDiffusion3_5Large, StableDiffusion3_5LargeTurbo, SDXLBase1_0, SDTurbo, SDXLTurbo1_0, Flux1Dev, Flux1Schnell, Flux1Mini, JuggernautXL11, Chroma, NitroSDRealism, NitroSDVibrant, DiffInstructStar, ChromaRadiance, SSD1B, Flux2Dev, ZImageTurbo, QwenImage, OvisImage, DreamShaperXL2_1Turbo, TwinFlowZImageTurboExp, SDXS512DreL2_1Turbo, TwinFlowZImageTurboExp, SDXS512DreamShaper]
  -w, --weights <WEIGHTS>    Optionally which type of quantization to use [possible values: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, I8, I16, I32, I64, F64, IQ1_M, BF16, TQ1_0, TQ2_0, MXFP4, F8_E4M3]
  -s, --steps <STEPS>        Override the preset default number of inference steps
  -w, --width <WIDTH>        Override the preset default width
  -h, --height <HEIGHT>      Override the preset default height
  -b, --batch <BATCH>        Number of images to generate (default 1) [default: 1]
  -o, --output <OUTPUT>      Output Folder [default: ./]
  -p, --preview <PREVIEW>    Enable preview [possible values: fast, accurate]
  -t, --token <TOKEN>        Set Huggingface Hub token
  -l, --low-vram             Enable optimization for gpu with lower GB
  -r, --random-seed          Enable Random Seed: different runs will produce different results
  -h, --help                 Print help
  -V, --version              Print version
  ```