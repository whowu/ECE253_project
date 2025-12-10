import os
import cv2
import numpy as np
from skimage import img_as_float


def blur_score(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def enhance_local_contrast_rgb(img_rgb_float, clip_limit=2.0):
    img_uint8 = (img_rgb_float * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    rgb_eq_float = rgb_eq.astype(np.float32) / 255.0
    return rgb_eq_float



def highlight_aware_sharpen(img_rgb_float, amount, sigma):
    base = cv2.GaussianBlur(img_rgb_float, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = img_rgb_float - base

    img_uint8 = (img_rgb_float * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    highlight_mask = (v > 0.85).astype(np.float32)
    shadow_mask = (v < 0.1).astype(np.float32)
    mid_mask = 1.0 - np.clip(highlight_mask + shadow_mask, 0.0, 1.0)
    mid_mask = mid_mask[:, :, None]

    sharpened = img_rgb_float + amount * detail * mid_mask
    sharpened = np.clip(sharpened, 0.0, 1.0)
    return sharpened


def process_spatial_adaptive(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = blur_score(gray)

    too_blurry_thresh = 15.0
    if score < too_blurry_thresh:
        enhanced = enhance_local_contrast_rgb(img_float, clip_limit=2.0)
        return enhanced, "too_blurry"

    if score > 150.0:
        level = "light"
        amount = 0.5
        sigma = 0.7
        clip_limit = 2.0
    elif score > 60.0:
        level = "medium"
        amount = 0.8
        sigma = 1.0
        clip_limit = 2.2
    else:
        level = "heavy"
        amount = 1.4
        sigma = 1.6
        clip_limit = 3.0

    contrast_boost = enhance_local_contrast_rgb(img_float, clip_limit=clip_limit)
    sharpened = highlight_aware_sharpen(contrast_boost, amount=amount, sigma=sigma)

    if level == "heavy":
        sharpened = highlight_aware_sharpen(sharpened, amount=0.6, sigma=0.8)

    return sharpened, level



def save_img(out_path, img_float_rgb):
    img_uint8 = (img_float_rgb * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_bgr)


def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(all_files)
    c_light = c_med = c_heavy = c_skip = 0

    print(f"Found {total} spatial blur images")
    print("Starting processing\n")

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(input_dir, fname)

        try:
            enhanced, level = process_spatial_adaptive(in_path)
            out_path = os.path.join(output_dir, fname)
            save_img(out_path, enhanced)

            if level == "light":
                c_light += 1
            elif level == "medium":
                c_med += 1
            elif level == "heavy":
                c_heavy += 1
            else:
                c_skip += 1

            progress = idx / total * 100.0
            print(f"[{idx}/{total}] {fname} | level: {level} | {progress:.1f}%")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\nDone")
    print(f" light:  {c_light}")
    print(f" medium: {c_med}")
    print(f" heavy:  {c_heavy}")
    print(f" too_blurry: {c_skip}")
    print(f"output dir: {output_dir}")



if __name__ == "__main__":
    in_dir = "data/D'/D'_annotated_base/spatial_blur/images"
    out_dir= "data/D'/D'_enhanced/spatial_blur/images"
    batch_process(in_dir, out_dir)
