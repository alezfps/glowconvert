## Installation

### Dependencies
```bash
pip install opencv-python pillow numpy
```

### Alternative Installation
```bash
pip install -r requirements.txt
```

## Usage

### Basic Syntax
```bash
python glow_effects.py <input_image> [OPTIONS]
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_path` | string | required | Path to input image file |
| `-o, --output` | string | auto-generated | Output file path |
| `-r, --radius` | integer | 40 | Glow radius in pixels |
| `-i, --intensity` | float | 3.0 | Glow intensity multiplier |
| `--type` | choice | strong | Effect type: strong, colored, neon |
| `--color` | RGB | 0 150 255 | Color values for colored effect (R G B) |

### Usage Examples

**Strong White Glow Effect:**
```bash
python glow_effects.py input.png --type strong --radius 50 --intensity 4.0
```

**Custom Colored Glow:**
```bash
python glow_effects.py logo.jpg --type colored --color 255 100 0 --radius 75
```

**Neon Effect Processing:**
```bash
python glow_effects.py text.png --type neon -o neon_output.png
```

**Batch Processing Example:**
```bash
for file in *.jpg; do python glow_effects.py "$file" --type strong; done
```

## Technical Implementation

### Processing Pipeline
1. **Image Loading**: BGR to RGB conversion with error handling
2. **Mask Generation**: Adaptive binary thresholding at 80-pixel intensity
3. **Morphological Processing**: Close and open operations for noise reduction
4. **Multi-Layer Blur**: Gaussian blur application with varying kernel sizes
5. **Layer Composition**: Weighted combination of blur layers
6. **Output Rendering**: High-quality image export with format preservation

### Algorithm Details
- **Threshold Value**: 80 (optimized for text and bright objects)
- **Kernel Sizes**: Dynamic calculation based on image dimensions
- **Blur Layers**: Up to 7 layers with exponentially increasing radii
- **Intensity Scaling**: Linear interpolation across blur layers

### Output Specifications
- Format preservation from input
- JPEG quality: 95%
- RGB color space
- Original resolution maintained

## Output Naming Convention

When no output path is specified, files are automatically named using the following pattern:

```
<original_filename>_<effect_type><original_extension>
```

**Examples:**
- `image.jpg` → `image_strong_glow.jpg`
- `logo.png` → `logo_colored_glow.png`
- `sign.png` → `sign_neon_effect.png`
