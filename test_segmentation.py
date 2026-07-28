import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def get_portrait_segmentation_mask_cv2(img_gray_pil):
    arr = np.array(img_gray_pil, dtype=np.uint8)
    
    # Otsu thresholding or adaptive thresholding
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Binary closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    
    # Filter out background label 0
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = closed
        
    # Fill holes inside largest component using contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [c], -1, 255, -1)
        
    # Erode slightly to remove border diffusion artifacts
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.erode(filled_mask, kernel_erode, iterations=1)
    
    return final_mask > 0

if __name__ == "__main__":
    from process_portrait import load_and_preprocess, serpentine_floyd_steinberg
    img = load_and_preprocess()
    mask = get_portrait_segmentation_mask_cv2(img)
    dithered = serpentine_floyd_steinberg(img)
    
    dark_mode_dots = (dithered == 255) & mask
    print("Dark mode subject dots count:", np.sum(dark_mode_dots))
    
    light_mode_dots = (dithered == 0)
    print("Light mode density dots count:", np.sum(light_mode_dots))
