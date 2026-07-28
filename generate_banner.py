import os
import math
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

IMG_PATH = r"C:\Users\Srinath\.gemini\antigravity-ide\brain\7c3ce351-f273-4b71-aa3c-737894880a98\media__1785245801072.jpg"

def load_and_preprocess_portrait():
    img = Image.open(IMG_PATH).convert('L')
    w, h = img.size
    
    # Head and shoulders crop (upper 55% of height, centered width)
    crop_w = int(w * 0.75)
    crop_h = int(h * 0.55)
    left = int((w - crop_w) / 2)
    top = int(h * 0.08)
    right = left + crop_w
    bottom = top + crop_h
    
    cropped = img.crop((left, top, right, bottom))
    
    # 320x360 working grid
    resized = cropped.resize((320, 360), Image.Resampling.LANCZOS)
    
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
        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1
            
        for x in x_range:
            old_val = arr[y, x]
            new_val = 255 if old_val > 120 else 0
            output[y, x] = new_val
            err = old_val - new_val
            
            nx = x + direction
            if 0 <= nx < w:
                arr[y, nx] += err * (7 / 16)
            if y + 1 < h:
                arr[y + 1, x] += err * (5 / 16)
                if 0 <= nx < w:
                    arr[y + 1, nx] += err * (1 / 16)
                prev_x = x - direction
                if 0 <= prev_x < w:
                    arr[y + 1, prev_x] += err * (3 / 16)
                    
    return output

def get_portrait_segmentation_mask(img_gray_pil):
    arr = np.array(img_gray_pil, dtype=np.uint8)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = closed
        
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [c], -1, 255, -1)
        
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.erode(filled_mask, kernel_erode, iterations=1)
    
    return final_mask > 0

def build_clean_svg(is_dark=True):
    if is_dark:
        bg_color = "#0A101F"
        border_color = "#1E293B"
        header_bg = "#0F172A"
        header_text = "#94A3B8"
        title_color = "#22D3EE"
        label_color = "#22D3EE"
        dots_color = "#334155"
        val_color = "#F1F5F9"
        val_highlight = "#10B981"
        pill_bg = "#1E293B"
        pill_border = "#22D3EE"
        pill_text = "#22D3EE"
        live_color = "#10B981"
        dot_color = "#A78BFA"
        visual_map_bg = "#0D1527"
    else:
        bg_color = "#F8FAFC"
        border_color = "#CBD5E1"
        header_bg = "#E2E8F0"
        header_text = "#475569"
        title_color = "#0891B2"
        label_color = "#0891B2"
        dots_color = "#CBD5E1"
        val_color = "#0F172A"
        val_highlight = "#059669"
        pill_bg = "#E2E8F0"
        pill_border = "#0891B2"
        pill_text = "#0891B2"
        live_color = "#059669"
        dot_color = "#7C3AED"
        visual_map_bg = "#F1F5F9"

    # Load & Dither Portrait
    img_gray = load_and_preprocess_portrait()
    mask = get_portrait_segmentation_mask(img_gray)
    dithered = serpentine_floyd_steinberg(img_gray)
    
    grid_h, grid_w = dithered.shape
    start_px, start_py = 55, 125
    scale_x = 350.0 / grid_w
    scale_y = 410.0 / grid_h
    
    portrait_dots = []
    step = 1 if is_dark else 2
    for y in range(0, grid_h, step):
        for x in range(0, grid_w, step):
            if is_dark:
                # Dark Mode: illuminated subject dots inside mask
                if dithered[y, x] == 255 and mask[y, x]:
                    px = start_px + x * scale_x
                    py = start_py + y * scale_y
                    portrait_dots.append((px, py))
            else:
                # Light Mode: density shading dots (dithered black pixels)
                if dithered[y, x] == 0:
                    px = start_px + x * scale_x
                    py = start_py + y * scale_y
                    portrait_dots.append((px, py))
                    
    portrait_dots = np.array(portrait_dots)
    print(f"[{'Dark' if is_dark else 'Light'} Mode] Total Clean Portrait Dots: {len(portrait_dots)}")

    # Group portrait dots into 94 height bands
    num_bands = 94
    y_min, y_max = np.min(portrait_dots[:, 1]), np.max(portrait_dots[:, 1])
    band_edges = np.linspace(y_min, y_max + 1, num_bands + 1)
    
    drift_bands = []
    for b in range(num_bands):
        in_band = (portrait_dots[:, 1] >= band_edges[b]) & (portrait_dots[:, 1] < band_edges[b+1])
        pts_b = portrait_dots[in_band]
        if len(pts_b) > 0:
            drift_bands.append(pts_b)

    cx_map, cy_map = 230, 330

    # SVG Construction
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">')
    svg.append(f'  <defs>')
    svg.append(f'    <style>')
    svg.append(f'      @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}')
    svg.append(f'      .live-dot {{ animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }}')
    svg.append(f'    </style>')
    svg.append(f'  </defs>')
    
    # Outer Background
    svg.append(f'  <rect width="1180" height="610" fill="{bg_color}" rx="12"/>')
    
    # Terminal Window Container
    svg.append(f'  <rect x="15" y="15" width="1150" height="580" fill="{bg_color}" stroke="{border_color}" stroke-width="2" rx="10"/>')
    
    # Header Bar
    svg.append(f'  <path d="M 15 25 A 10 10 0 0 1 25 15 L 1155 15 A 10 10 0 0 1 1165 25 L 1165 55 L 15 55 Z" fill="{header_bg}"/>')
    svg.append(f'  <line x1="15" y1="55" x2="1165" y2="55" stroke="{border_color}" stroke-width="1.5"/>')
    
    # Window Buttons
    svg.append(f'  <circle cx="40" cy="35" r="6" fill="#FF5F56"/>')
    svg.append(f'  <circle cx="60" cy="35" r="6" fill="#FFBD2E"/>')
    svg.append(f'  <circle cx="80" cy="35" r="6" fill="#27C93F"/>')
    
    # Title
    svg.append(f'  <text x="590" y="39" font-family="ui-monospace, Consolas, monospace" font-size="14" font-weight="600" fill="{title_color}" text-anchor="middle">profile.sh --live</text>')
    
    # Split Line
    split_x = 445
    svg.append(f'  <line x1="{split_x}" y1="55" x2="{split_x}" y2="595" stroke="{border_color}" stroke-width="1.5"/>')
    
    # -------------------------------------------------------------
    # LEFT PANEL: VISUAL.MAP (CLEAN PORTRAIT DITHER)
    # -------------------------------------------------------------
    svg.append(f'  <!-- VISUAL.MAP Panel -->')
    svg.append(f'  <text x="35" y="85" font-family="ui-monospace, Consolas, monospace" font-size="13" font-weight="bold" fill="{label_color}">VISUAL.MAP</text>')
    svg.append(f'  <text x="425" y="85" font-family="ui-monospace, Consolas, monospace" font-size="11" fill="{header_text}" text-anchor="end">[STREAM: 60 FPS]</text>')
    
    # Inner map frame
    svg.append(f'  <rect x="35" y="98" width="390" height="472" fill="{visual_map_bg}" stroke="{border_color}" stroke-width="1" rx="6"/>')
    
    # Subtle crosshair guides
    svg.append(f'  <line x1="{cx_map}" y1="110" x2="{cx_map}" y2="558" stroke="{border_color}" stroke-width="0.75" stroke-dasharray="3,3" opacity="0.4"/>')
    svg.append(f'  <line x1="45" y1="{cy_map}" x2="415" y2="{cy_map}" stroke="{border_color}" stroke-width="0.75" stroke-dasharray="3,3" opacity="0.4"/>')
    
    # DITHERED PORTRAIT
    svg.append(f'  <!-- Dithered Portrait (Illuminated Subject) -->')
    svg.append(f'  <g fill="{dot_color}">')
    
    for idx, band_pts in enumerate(drift_bands):
        # Subtle ambient breathing drift along height bands
        dx = math.sin(idx * 0.15) * 1.5
        dy = math.cos(idx * 0.15) * 1.2
        
        path_d = []
        for px, py in band_pts:
            path_d.append(f"M {px:.1f} {py:.1f} h 1.4 v 1.4 h -1.4 Z")
        d_str = " ".join(path_d)
        
        # Smooth subtle breathing animation loop
        svg.append(f'    <path d="{d_str}">')
        svg.append(f'      <animateTransform attributeName="transform" type="translate" values="0,0; {dx:.1f},{dy:.1f}; 0,0; {-dx:.1f},{-dy:.1f}; 0,0" keyTimes="0;0.25;0.5;0.75;1" dur="10s" repeatCount="indefinite"/>')
        svg.append(f'    </path>')
    svg.append(f'  </g>')

    # -------------------------------------------------------------
    # RIGHT PANEL: SYSTEM.INFO
    # -------------------------------------------------------------
    svg.append(f'  <!-- SYSTEM.INFO Panel -->')
    svg.append(f'  <text x="470" y="85" font-family="ui-monospace, Consolas, monospace" font-size="13" font-weight="bold" fill="{label_color}">SYSTEM.INFO</text>')
    svg.append(f'  <circle cx="585" cy="81" r="4" fill="{live_color}" class="live-dot"/>')
    svg.append(f'  <text x="595" y="85" font-family="ui-monospace, Consolas, monospace" font-size="12" font-weight="bold" fill="{live_color}">LIVE</text>')
    
    pill_x = 1030
    svg.append(f'  <rect x="{pill_x}" y="67" width="115" height="24" fill="{pill_bg}" stroke="{pill_border}" stroke-width="1.2" rx="12"/>')
    svg.append(f'  <text x="{pill_x + 57.5}" y="83" font-family="ui-monospace, Consolas, monospace" font-size="13" font-weight="bold" fill="{pill_text}" text-anchor="middle">srinath-712</text>')

    rows_data = [
        ("Subject", "SRINATH A P", val_color, True),
        ("Role", "Full Stack Developer &amp; AI Engineer", val_highlight, True),
        ("Origin", "Chennai", val_color, False),
        ("Education", "B.Tech Computer Science Engineering", val_color, False),
        ("Status", "Building Privacy-First Mobile Apps and AI Projects", val_highlight, True),
        ("ToolChain", "VS Code · Git", val_color, False),
        ("Core.Lang", "Python · C · C++ · Java · JavaScript", val_color, False),
        ("Core.Frontend", "React · Flutter · React Native", val_color, False),
        ("Core.Backend", "Node.js", val_color, False),
        ("Core.Database", "PostgreSQL", val_color, False),
        ("Core.Infra", "AWS", val_color, False),
        ("Grid.Mail", "srinath.a712@gmail.com", val_color, False),
        ("Grid.Portfolio", "Coming Soon", header_text, False),
        ("Grid.LinkedIn", "linkedin.com/in/srinath-a-p", val_color, False),
        ("Grid.GitHub", "github.com/srinath-712", val_color, False),
        ("Grid.Instagram", "instagram.com/srinath_712", val_color, False),
    ]

    start_y = 122
    row_height = 28
    left_x = 470
    right_x = 1145
    char_w = 8.4
    
    for idx, (label, val, val_c, is_bold) in enumerate(rows_data):
        y = start_y + idx * row_height
        svg.append(f'    <text x="{left_x}" y="{y}" font-family="ui-monospace, Consolas, monospace" font-size="14" font-weight="bold" fill="{label_color}">{label}</text>')
        
        lbl_len = len(label)
        val_len = len(val.replace("&amp;", "&"))
        l_end = left_x + int(lbl_len * char_w) + 12
        v_start = right_x - int(val_len * char_w) - 12
        dots_w = max(20, v_start - l_end)
        
        leader_dots = "..........................................................................................."
        svg.append(f'    <text x="{l_end}" y="{y}" font-family="ui-monospace, Consolas, monospace" font-size="14" fill="{dots_color}" textLength="{dots_w:.1f}" lengthAdjust="spacingAndGlyphs">{leader_dots}</text>')
        
        font_wt = "bold" if is_bold else "500"
        svg.append(f'    <text x="{right_x}" y="{y}" font-family="ui-monospace, Consolas, monospace" font-size="14" font-weight="{font_wt}" fill="{val_c}" text-anchor="end">{val}</text>')

    svg.append(f'</svg>')
    return "\n".join(svg)

if __name__ == "__main__":
    dark_svg = build_clean_svg(is_dark=True)
    with open("dark.svg", "w", encoding="utf-8") as f:
        f.write(dark_svg)
    print("Generated clean dark.svg!")
    
    light_svg = build_clean_svg(is_dark=False)
    with open("light.svg", "w", encoding="utf-8") as f:
        f.write(light_svg)
    print("Generated clean light.svg!")
