import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse
import os

def create_strong_glow_effect(image_path, output_path=None, glow_radius=40, glow_intensity=3.0):
    try:
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        print(f"Mask created - white pixels found: {np.sum(mask > 0)}")
        
        result = np.zeros_like(image_rgb)
        
        mask_bool = mask > 0
        result[mask_bool] = [255, 255, 255]
        
        glow_image = np.zeros_like(image_rgb, dtype=np.float32)
        
        glow_radii = [5, 10, 20, 35, 50, 75, 100]
        glow_intensities = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
        
        for radius, intensity in zip(glow_radii, glow_intensities):
            if radius * 2 + 1 <= min(height, width):
                kernel_size = radius * 2 + 1
                blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), radius/3)
                
                blurred_mask = blurred_mask / 255.0 * intensity * glow_intensity
                
                for i in range(3):
                    glow_image[:, :, i] += blurred_mask
        
        glow_image = np.clip(glow_image * 255, 0, 255).astype(np.uint8)
        
        final_result = np.maximum(result, glow_image)
        
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_strong_glow{ext}"
        
        final_pil = Image.fromarray(final_result)
        final_pil.save(output_path, quality=95)
        print(f"Strong glow effect created! Saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def create_colored_glow_effect(image_path, output_path=None, glow_color=(0, 150, 255), glow_radius=50, glow_intensity=4.0):
    try:
        print(f"Creating colored glow effect for: {image_path}")
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        result = np.zeros_like(image_rgb, dtype=np.float32)
        
        glow_layers = [10, 20, 35, 50, 75]
        intensities = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        for radius, intensity in zip(glow_layers, intensities):
            kernel_size = radius * 2 + 1
            if kernel_size <= min(image_rgb.shape[:2]):
                blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), radius/3)
                blurred_norm = blurred / 255.0 * intensity * glow_intensity
                
                result[:, :, 0] += blurred_norm * glow_color[0] / 255.0
                result[:, :, 1] += blurred_norm * glow_color[1] / 255.0  
                result[:, :, 2] += blurred_norm * glow_color[2] / 255.0
        
        mask_bool = mask > 0
        result[mask_bool] = [1.0, 1.0, 1.0]
        
        final_result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_colored_glow{ext}"
        
        Image.fromarray(final_result).save(output_path, quality=95)
        print(f"Colored glow effect created! Saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def create_neon_effect(image_path, output_path=None):
    try:
        print(f"Creating neon effect for: {image_path}")
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        result = np.zeros_like(image_rgb, dtype=np.float32)
        
        colors = [
            (255, 255, 255),
            (100, 200, 255),
            (50, 150, 255),
            (0, 100, 255),
            (0, 50, 200),
        ]
        
        radii = [3, 15, 30, 50, 80]
        intensities = [1.0, 0.8, 0.6, 0.4, 0.3]
        
        for (r, g, b), radius, intensity in zip(colors, radii, intensities):
            kernel_size = radius * 2 + 1
            if kernel_size <= min(image_rgb.shape[:2]):
                blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), radius/3)
                blurred_norm = blurred / 255.0 * intensity
                
                result[:, :, 0] += blurred_norm * r / 255.0
                result[:, :, 1] += blurred_norm * g / 255.0
                result[:, :, 2] += blurred_norm * b / 255.0
        
        final_result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_neon_effect{ext}"
        
        Image.fromarray(final_result).save(output_path, quality=95)
        print(f"Neon effect created! Saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Create strong glow effects on images')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output path (optional)')
    parser.add_argument('-r', '--radius', type=int, default=40, help='Glow radius (default: 40)')
    parser.add_argument('-i', '--intensity', type=float, default=3.0, help='Glow intensity (default: 3.0)')
    parser.add_argument('--type', choices=['strong', 'colored', 'neon'], default='strong',
                       help='Type of glow effect (default: strong)')
    parser.add_argument('--color', nargs=3, type=int, metavar=('R', 'G', 'B'),
                       default=[0, 150, 255], help='Glow color for colored effect (default: blue)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found!")
        return
    
    if args.type == 'strong':
        create_strong_glow_effect(args.input_path, args.output, args.radius, args.intensity)
    elif args.type == 'colored':
        create_colored_glow_effect(args.input_path, args.output, tuple(args.color), args.radius, args.intensity)
    elif args.type == 'neon':
        create_neon_effect(args.input_path, args.output)

if __name__ == "__main__":
    main()
