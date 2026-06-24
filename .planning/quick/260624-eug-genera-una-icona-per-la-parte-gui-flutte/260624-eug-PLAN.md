---
phase: quick
plan: 260624-eug
type: execute
wave: 1
depends_on: []
files_modified:
  - gui/assets/app_icon.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_16.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_32.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_64.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_128.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_256.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_512.png
  - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_1024.png
  - gui/windows/runner/resources/app_icon.ico
autonomous: true
requirements: []
must_haves:
  truths:
    - "App icon shows a red flame on white background across all desktop platforms"
    - "macOS app icon uses the custom flame icon at all required sizes"
    - "Windows app icon uses the custom flame icon in ICO format"
  artifacts:
    - path: "gui/assets/app_icon.png"
      provides: "High-resolution source icon (1024x1024)"
    - path: "gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_1024.png"
      provides: "macOS icon at largest size"
    - path: "gui/windows/runner/resources/app_icon.ico"
      provides: "Windows ICO icon"
  key_links: []
---

<objective>
Generate a custom app icon for the DiffusionRS GUI Flutter application: a red flame on a white background. Replace the default Flutter icons on all desktop platforms (macOS, Windows, Linux).

Purpose: Give the app a distinctive, recognizable identity instead of the default Flutter logo.
Output: Custom flame icon files for macOS (7 PNG sizes), Windows (ICO), and a high-res source PNG in assets/.
</objective>

<execution_context>
@/Users/flavio.bizzarri/repo/diffusion-rs/.claude/gsd-core/workflows/execute-plan.md
@/Users/flavio.bizzarri/repo/diffusion-rs/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Generate flame icon at all required sizes using Python/Pillow</name>
  <files>
    gui/assets/app_icon.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_16.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_32.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_64.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_128.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_256.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_512.png,
    gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_1024.png,
    gui/windows/runner/resources/app_icon.ico
  </files>
  <action>
    Write a Python script using Pillow (PIL) that programmatically draws a red flame icon on a white background. The flame should be drawn using polygon shapes with bezier-like curves via ImageDraw, centered on the canvas. Use a rich red color (e.g. RGB 220, 40, 40) for the main flame body, with an optional orange-yellow inner highlight (RGB 255, 160, 30) for depth. The background must be pure white (RGB 255, 255, 255).

    Design the flame at 1024x1024 resolution first. The flame shape should be a stylized teardrop/fire silhouette — wide at the base, tapering to a point at the top, with a slight fork or curve at the tip for visual interest. Keep the design simple and bold so it reads well at 16x16.

    Generate all required output files:
    - gui/assets/app_icon.png at 1024x1024 (high-res source, also usable by Linux)
    - gui/macos/Runner/Assets.xcassets/AppIcon.appiconset/ at sizes: 16, 32, 64, 128, 256, 512, 1024 (overwrite existing default Flutter icons)
    - gui/windows/runner/resources/app_icon.ico containing embedded sizes 16, 32, 48, 64, 128, 256 (overwrite existing default Flutter icon)

    Use LANCZOS resampling when downscaling from 1024 to smaller sizes for best quality. The macOS icon files must use the existing naming convention: app_icon_{size}.png. The Contents.json in the macOS appiconset does not need modification since the filenames remain the same.

    Run the script, verify outputs exist and have correct dimensions, then delete the script (it is a one-shot generator, not a project artifact).
  </action>
  <verify>
    <automated>python3 -c "
from PIL import Image
import os
base = '/Users/flavio.bizzarri/repo/diffusion-rs/gui'
checks = [
    (f'{base}/assets/app_icon.png', 1024),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_1024.png', 1024),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_512.png', 512),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_256.png', 256),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_128.png', 128),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_64.png', 64),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_32.png', 32),
    (f'{base}/macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_16.png', 16),
    (f'{base}/windows/runner/resources/app_icon.ico', None),
]
for path, expected_size in checks:
    assert os.path.exists(path), f'Missing: {path}'
    img = Image.open(path)
    if expected_size:
        assert img.size == (expected_size, expected_size), f'{path}: expected {expected_size}x{expected_size}, got {img.size}'
    # Verify not default Flutter blue - check center pixel is not blue
    px = img.convert('RGB').getpixel((img.width//2, img.height//2))
    assert px[2] < 200 or px[0] > 150, f'{path}: still looks like default Flutter icon (center pixel {px})'
print('All icon files verified successfully')
"
    </automated>
  </verify>
  <done>
    All icon files exist at correct sizes, center pixels confirm the flame design (not default Flutter blue), and the generator script has been cleaned up. macOS Contents.json unchanged (same filenames). Windows ICO contains multi-resolution icon data.
  </done>
</task>

</tasks>

<verification>
- All 8 macOS icon PNGs replaced with flame icon at correct dimensions
- Windows app_icon.ico replaced with flame icon containing multiple sizes
- gui/assets/app_icon.png exists at 1024x1024 as high-res source
- No default Flutter blue logo remains in any icon file
- No generator script left in the repository
</verification>

<success_criteria>
The DiffusionRS GUI shows a red flame on white background as its app icon on macOS, Windows, and Linux instead of the default Flutter logo.
</success_criteria>

<output>
Create `.planning/quick/260624-eug-genera-una-icona-per-la-parte-gui-flutte/260624-eug-SUMMARY.md` when done
</output>
