# Feature Landscape: AI Image Generation Desktop GUIs

**Domain:** Desktop GUI for AI image generation (Stable Diffusion / Flux family)
**Researched:** 2026-06-18
**Reference tools:** AUTOMATIC1111 (WebUI), ComfyUI, InvokeAI, Fooocus, DiffusionBee, Draw Things
**Confidence:** HIGH (based on direct tool knowledge through August 2025)

---

## Table Stakes

Features every serious SD/Flux GUI has. Missing = users leave immediately or rate it "broken."

| Feature | Why Expected | Complexity | Notes for diffusion-rs GUI |
|---------|--------------|------------|---------------------------|
| Prompt text area | Core input; no alt exists | Low | Multi-line, no fixed max width, scrollable |
| Negative prompt field | Essential for SD 1.x/2.x/SDXL quality control; less critical for Flux but still expected | Low | Separate from positive; can be collapsed for Flux |
| Steps slider/field | Every user tweaks this constantly | Low | Numeric input + optional slider; range 1–150 typical |
| Width / height selection | Every generation requires it | Low | Common presets (512×512, 768×768, 1024×1024, 1024×768, etc.) via dropdown or free int fields |
| Model / preset selector | Users have multiple models; switching is the most frequent config action | Low | Dropdown with human-readable names; must be prominent |
| CFG / guidance scale | Used universally across SD and Flux; Flux uses a different scale but concept identical | Low | Float field or slider; range 1–20 typical |
| Seed field with randomize button | Power users always fix seed for reproducibility; "–1 = random" is universal convention | Low | Int field + dice/shuffle icon button to clear to –1 |
| Generate / Run button | Obvious | Low | Prominent, primary button; disables during generation |
| Progress display during generation | Users expect visual feedback; blank screen during 30-second runs is a hard fail | Medium | Step counter + progress bar; see below for conventions |
| Live preview image | Expected by anyone who used A1111 or ComfyUI; absence feels like regression | Medium | Intermediate decoded latent shown every N steps |
| Output image display | The result must appear in the app, not just be saved to disk | Low | Right-side panel; must show at native/fit resolution |
| Save image button | Users want explicit control over which outputs to keep | Low | Triggered manually post-generation; saves from temp dir |
| Batch count field | Generating multiple images per run is a core workflow | Low | Int field, minimum 1 |

---

## Parameter Panel UX Patterns

How good GUIs handle 15+ inputs without clutter — synthesized from A1111, InvokeAI, Fooocus, Draw Things.

### What works: collapsible accordion sections

A1111 (and InvokeAI) group parameters into collapsible sections. The canonical grouping:

- **Core** (always visible): prompt, negative, preset/model, steps, size, generate button
- **Advanced** (collapsed by default): seed, CFG, sampler, scheduler, clip skip
- **Post-processing** (collapsed by default): upscaler, hires fix, refiner

Rationale: 80% of users only touch core params. Advanced users expand what they need. No one is overwhelmed.

### What works: contextual visibility

Fooocus pioneered "hide everything except what matters for this model." The approach:

- Show upscaler_scale only when an upscaler is selected (not "none")
- Show weights dropdown only for presets that support selectable weights
- Hide negative prompt for Flux models (they ignore it) or show it grayed out

This is directly relevant to diffusion-rs: the `upscaler_scale` field should be invisible when upscaler = "none". The `weights` dropdown should appear only for applicable presets.

### What works: inline validation

InvokeAI shows width/height constraints inline (e.g., "must be multiple of 8" or "must be multiple of 64 for SDXL"). No modal dialogs. Error text appears below the field in red.

### What works: size presets dropdown/chips

Rather than free int fields for width/height, offer a preset dropdown (512×512, 768×512, 1024×1024) with an "Advanced" toggle that reveals free int fields. Reduces input errors from non-multiple-of-8 values.

### What does not work: flat list of 20+ fields

Raw CLI-style layout with no grouping. Users scroll endlessly to find the one parameter they need. A1111's early tabs-per-feature approach created confusion; the accordion approach won.

---

## Progress Display Conventions

Based on A1111, ComfyUI, InvokeAI, and Draw Things behavior:

| Convention | Standard behavior | Notes |
|------------|-------------------|-------|
| Step counter | "Step 12/30" displayed as text | Always shown during generation |
| Progress bar | Fills linearly from 0→100% as steps complete | Determinate bar, not spinner |
| Preview update frequency | Every 5 steps or every ~1 second, whichever is less frequent | Too frequent = UI jank; too infrequent = feels stalled |
| Preview decoding | Decoded from latent space — blurry at first, sharpens by step 15–20 | Expected visual artifact; users understand it |
| Preview size | Same panel as final output, same dimensions | Not a thumbnail — full panel |
| Time elapsed / ETA | "12s elapsed / ~18s remaining" shown during generation | ComfyUI and InvokeAI do this; A1111 does not in base |
| Button state during generation | Generate button changes to "Cancel" or "Stop" | A1111: interrupt button; ComfyUI: cancel node button |

### diffusion-rs specific

The CLI already emits step callbacks. The GUI should:
- Show "Step N / total" (from callback data)
- Refresh preview image on each callback where an intermediate image is provided
- Not re-render the entire left panel on each update (performance)

---

## Image Saving Patterns

How good GUIs handle saving output:

| Pattern | Used by | Notes |
|---------|---------|-------|
| Auto-save to output folder with sequential name | A1111, ComfyUI | Always saves; user picks folder in settings |
| Temp-then-explicit-save | DiffusionBee, Draw Things | Only keeps what user explicitly saves; cleaner for casual users |
| Metadata embedding in PNG | A1111, InvokeAI | Embeds full generation params in PNG chunks for reproducibility |
| Filename includes prompt excerpt + seed | A1111 default | `00042-3456789012-a fantasy landscape.png` |
| Filename is timestamp | ComfyUI default | `ComfyUI_00042_.png` |
| Separate outputs gallery | InvokeAI, Draw Things | App maintains a history panel; not just filesystem |

### Recommendation for diffusion-rs GUI

The PROJECT.md specifies temp dir + explicit save button. This matches the DiffusionBee/Draw Things pattern. When the user clicks Save:
- Default to user's Pictures folder or last-used folder (persisted in app settings)
- Default filename: `{preset}_{seed}_{timestamp}.png` — no prompt excerpt (avoids filesystem special chars)
- PNG format only for initial version; no metadata embedding needed in v1

---

## Preset / Model Selection UX

| Pattern | Used by | Notes |
|---------|---------|-------|
| Single flat dropdown | Fooocus, Draw Things | Simple; works when < 20 models |
| Grouped dropdown (by architecture) | InvokeAI | SD 1.x / SDXL / Flux as subgroups; good for 20–50 models |
| Search-filtered dropdown | A1111 (with extensions), ComfyUI | Needed at 50+ models |
| Model card grid | InvokeAI | Visual thumbnails; overkill for v1 |
| Context-sensitive weights sub-dropdown | diffusion-rs CLI | Already modeled in PROJECT.md; show only when preset has selectable weights |

### Recommendation

For diffusion-rs (~35 presets as of v0.1.20): a grouped dropdown organized by architecture family (SD 1.x, SD 2.x, SDXL, SD3, Flux) with the weights sub-dropdown appearing contextually. No search needed at this count.

---

## Differentiators

Features that separate good GUIs from adequate ones. Users notice but do not necessarily demand:

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Seed locking with visual indicator | Easy reproducibility; "lock" icon next to seed field shows whether seed is fixed or random | Low | Shows intent clearly; avoids confusion about why results vary |
| "Copy seed from last result" button | One-click reproducibility after finding a good result | Low | Very high UX value per implementation cost |
| Parameter change highlighting | Fields that differ from defaults shown in a different color | Medium | Helps users understand what they changed |
| Keyboard shortcut for generate | Cmd+Enter or Ctrl+Enter to generate without mouse | Low | Power users expect this |
| Prompt history (last 10) | Users iterate on prompts; typing the same thing twice is friction | Medium | Dropdown or up-arrow recall in prompt field |
| Generation queue / cancel mid-run | Cancel without closing the app | Medium | diffusion-rs backend must support interrupt |
| Theme persistence | System/light/dark preference remembered across sessions | Low | PROJECT.md already calls this out |
| Drag-to-resize panels | Two-panel split should be resizable | Low | Flutter SplitView or custom drag handle |
| Compact / expanded panel toggle | Some users want to maximize the image preview area | Low | Hide parameter panel entirely when not needed |
| Output folder quick-open | Button to reveal output folder in Finder/Explorer | Low | One-click access to saved images |

---

## Anti-Features

Things that make AI image generation GUIs annoying. Deliberately avoid:

| Anti-Feature | Why Annoying | What to Do Instead |
|--------------|--------------|-------------------|
| Regenerating on every parameter change | Any slider move triggers a full generation — wastes GPU and time | Generate only on explicit button press or Enter |
| Modal dialogs for generation errors | Blocks UI; forces interaction before user can fix and retry | Inline error message in the panel; non-blocking |
| Settings buried in menu submenus | HuggingFace token, output folder, theme toggle should be first-class | Put persistent settings in a settings panel or sidebar, not buried 3 levels deep |
| No progress feedback for >5 second operations | Users assume the app crashed | Always show a progress bar or spinner for any blocking operation |
| Requiring app restart to apply model/preset changes | A1111 legacy behavior; infuriating | Model load is triggered on next generation, not on close/reopen |
| Clearing the prompt on new generation | Losing the prompt after hitting Generate is a hard fail | Preserve all inputs between generations |
| Fixed window size / non-resizable | Clashes with different monitor configurations | Fully resizable; remember last window size and position |
| Exposing raw CLI flags as text fields with no labels | Direct CLI argument names (--cfg_scale, --n_iter) are meaningless to GUI users | Human-readable labels: "Guidance Scale", "Batch Size" |
| Saving output to cwd silently | Files appear wherever the app binary is; confusing | Explicit output folder with visible path |
| No "open containing folder" for saved images | Users want to find and share images immediately | Button to reveal file in system file manager |
| Upscaler requiring user to know the right cache mode | Technical dependency between upscaler and cache is an internal concern | UI should enforce or auto-select the required cache mode when upscaler is chosen |
| Password/token field with no toggle | HuggingFace token visible in plain text risks shoulder surfing; fully hidden token is unusable | Password field with show/hide toggle (PROJECT.md already handles this correctly) |
| Prompt field that does not grow | Fixed-height text box scrolls horizontally for long prompts; unreadable | Multi-line growing text area with scroll |

---

## Feature Dependencies

```
Upscaler dropdown (non-none) → upscaler_scale field becomes visible
Upscaler active              → cache mode must be non-none (enforce in UI)
Weights dropdown             → visible only when selected preset supports selectable weights
Seed field                   → dice button clears to –1 (randomize); lock icon shows fixed state
Progress bar                 → requires generation to have started (hidden at rest)
Preview panel                → requires generation to have started (shows placeholder at rest)
Save button                  → enabled only after a generation has completed
```

---

## MVP Recommendation

For diffusion-rs GUI v1 (the scope defined in PROJECT.md), prioritize:

**Must have (table stakes — already in PROJECT.md scope):**
1. Preset dropdown + contextual weights sub-dropdown
2. Prompt text area (multi-line, growing)
3. Negative prompt field
4. Steps, width, height, CFG, batch, seed fields with randomize button
5. Cache dropdown, preview dropdown, upscaler dropdown, upscaler_scale (conditional)
6. HuggingFace token password field with toggle
7. low_vram toggle
8. Generate button (disables all inputs during generation)
9. Progress bar with step counter
10. Live preview panel updating during generation
11. Final image display panel
12. Explicit save button with folder picker

**Add in v1 for quality (low cost, high value):**
- Keyboard shortcut Cmd/Ctrl+Enter to generate
- Output folder quick-open button (reveal in Finder/Explorer) post-save
- Seed "dice" randomize button (clear to –1)
- Contextual hiding of upscaler_scale when upscaler = none
- Auto-enforce non-none cache when upscaler is selected (or show warning)

**Defer to v2 (not in scope per PROJECT.md):**
- Prompt history / recall
- Parameter change highlighting
- Generation queue
- Outputs gallery / history panel
- Grouped/searchable model list (relevant only at 50+ models)
- Metadata embedding in PNG

---

## Sources

Based on direct knowledge of the following tools (knowledge cutoff August 2025):
- AUTOMATIC1111 WebUI (github.com/AUTOMATIC1111/stable-diffusion-webui) — reference for A1111 conventions
- ComfyUI (github.com/comfyanonymous/ComfyUI) — reference for node graph and progress display
- InvokeAI (github.com/invoke-ai/InvokeAI) — reference for UX patterns and contextual UI
- Fooocus (github.com/lllyasviel/Fooocus) — reference for simplified/opinionated UI design
- DiffusionBee (diffusionbee.com) — reference for macOS-native temp-then-save pattern
- Draw Things (drawthings.ai) — reference for mobile-to-desktop portable UI patterns

Confidence: HIGH for table stakes and anti-features (universal across all tools); MEDIUM for differentiator ordering (subjective UX judgment).
