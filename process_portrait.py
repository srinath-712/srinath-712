import os
import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# Input image path
IMG_PATH = r"C:\Users\Srinath\.gemini\antigravity-ide\brain\7c3ce351-f273-4b71-aa3c-737894880a98\media__1785245801072.jpg"

def load_and_preprocess():
    img = Image.open(IMG_PATH).convert('L')
    w, h = img.size
    print(f"Original image size: {w}x{h}")
    
    # Head and shoulders crop
    # The image is a full body shot. Let's crop head and shoulders: upper 55% of height, centered width.
    crop_w = int(w * 0.75)
    crop_h = int(h * 0.55)
    left = int((w - crop_w) / 2)
    top = int(h * 0.08) # start slightly below top
    right = left + crop_w
    bottom = top + crop_h
    
    cropped = img.crop((left, top, right, bottom))
    
    # Resize to working grid 300x340
    resized = cropped.resize((300, 340), Image.Resampling.LANCZOS)
    
    # Contrast 1.3x
    enhancer = ImageEnhance.Contrast(resized)
    img_contrast = enhancer.enhance(1.3)
    
    # Autocontrast cutoff=1
    img_auto = ImageOps.autocontrast(img_contrast, cutoff=1)
    
    # UnsharpMask radius=3, percent=140
    img_sharp = img_auto.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
    return img_sharp

def serpentine_floyd_steinberg(img_gray):
    """1-bit Floyd-Steinberg dither with Serpentine error diffusion."""
    arr = np.array(img_gray, dtype=float)
    h, w = arr.shape
    output = np.zeros((h, w), dtype=int)
    
    for y in range(h):
        # Serpentine: alternate left-to-right and right-to-left
        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1
            
        for x in x_range:
            old_val = arr[y, x]
            new_val = 255 if old_val > 127 else 0
            output[y, x] = new_val
            err = old_val - new_val
            
            # Distribute error to neighbors
            # Right neighbor
            nx = x + direction
            if 0 <= nx < w:
                arr[y, nx] += err * (7 / 16)
            # Bottom neighbors
            if y + 1 < h:
                arr[y + 1, x] += err * (5 / 16)
                if 0 <= nx < w:
                    arr[y + 1, nx] += err * (1 / 16)
                prev_x = x - direction
                if 0 <= prev_x < w:
                    arr[y + 1, prev_x] += err * (3 / 16)
                    
    return output

if __name__ == "__main__":
    processed = load_and_preprocess()
    dithered = serpentine_floyd_steinberg(processed)
    print(f"Dithered grid generated: {dithered.shape}, white pixels: {np.sum(dithered == 255)}, black pixels: {np.sum(dithered == 0)}")
