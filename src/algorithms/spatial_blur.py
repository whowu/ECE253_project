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


def save_img(out_path, img_float_rgb):
    img_uint8 = (img_float_rgb * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_bgr)


# -------------------------------------------------------------------
#  algo 1: CLAHE
#  def spatial_contrast(src_dir, dst_dir, param)
# -------------------------------------------------------------------
def _contrast_single(image_path, clip_limit):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)
    enhanced = enhance_local_contrast_rgb(img_float, clip_limit=clip_limit)
    return enhanced


def spatial_contrast(src_dir, dst_dir, param):
    """
    param:
        None → clip_limit = 2.0
        list → [clip_limit]
    """
    if param is None:
        clip_limit = 2.0
    else:
        if not isinstance(param, list) or len(param) != 1:
            raise ValueError("spatial_contrast: param 应为 [clip_limit]")
        clip_limit = float(param[0])

    os.makedirs(dst_dir, exist_ok=True)

    all_files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(all_files)
    print(f"Found {total} images for spatial_contrast")

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(src_dir, fname)
        try:
            enhanced = _contrast_single(in_path, clip_limit)
            save_img(os.path.join(dst_dir, fname), enhanced)
            print(f"[{idx}/{total}] {fname} done")
        except Exception as e:
            print(f"Error {fname}: {e}")

    print(f"spatial_contrast output dir: {dst_dir}")


# -------------------------------------------------------------------
#  algo 2: highlight aware sharpening
#  def spatial_sharpen(src_dir, dst_dir, param)
# -------------------------------------------------------------------
def _sharpen_single(image_path, amount, sigma):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)
    sharpened = highlight_aware_sharpen(img_float, amount=amount, sigma=sigma)
    return sharpened


def spatial_sharpen(src_dir, dst_dir, param):
    """
    param:
        None → amount = 0.8, sigma = 1.0
        list → [amount, sigma]
    """
    if param is None:
        amount, sigma = 0.8, 1.0
    else:
        if not isinstance(param, list) or len(param) != 2:
            raise ValueError("spatial_sharpen: param 应为 [amount, sigma]")
        amount = float(param[0])
        sigma = float(param[1])

    os.makedirs(dst_dir, exist_ok=True)

    all_files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(all_files)
    print(f"Found {total} images for spatial_sharpen")

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(src_dir, fname)
        try:
            sharpened = _sharpen_single(in_path, amount, sigma)
            save_img(os.path.join(dst_dir, fname), sharpened)
            print(f"[{idx}/{total}] {fname} done")
        except Exception as e:
            print(f"Error {fname}: {e}")

    print(f"spatial_sharpen output dir: {dst_dir}")


# -------------------------------------------------------------------
#  algo 3: adaptive spatial pipeline
#  def spatial_adaptive(src_dir, dst_dir, param)
# -------------------------------------------------------------------
def _adaptive_single(image_path, param):
    """

    param:
        None → default
        list[params]:
            [
              too_blurry_thresh,
              thresh_light, thresh_medium,
              light_amount,  light_sigma,  light_clip,
              med_amount,    med_sigma,    med_clip,
              heavy_amount,  heavy_sigma,  heavy_clip,
              extra_heavy_amount, extra_heavy_sigma
            ]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    score = blur_score(gray)

    if param is None:
        too_blurry_thresh = 15.0
        thresh_light = 150.0
        thresh_medium = 60.0
        light_amount, light_sigma, light_clip = 0.5, 0.7, 2.0
        med_amount, med_sigma, med_clip = 0.8, 1.0, 2.2
        heavy_amount, heavy_sigma, heavy_clip = 1.4, 1.6, 3.0
        extra_heavy_amount, extra_heavy_sigma = 0.6, 0.8
    else:
        if not isinstance(param, list) or len(param) != 14:
            raise ValueError(
                "spatial_adaptive: params:\n"
                "[too_blurry_thresh, thresh_light, thresh_medium,"
                " light_amount, light_sigma, light_clip,"
                " med_amount, med_sigma, med_clip,"
                " heavy_amount, heavy_sigma, heavy_clip,"
                " extra_heavy_amount, extra_heavy_sigma]"
            )
        (
            too_blurry_thresh,
            thresh_light, thresh_medium,
            light_amount, light_sigma, light_clip,
            med_amount, med_sigma, med_clip,
            heavy_amount, heavy_sigma, heavy_clip,
            extra_heavy_amount, extra_heavy_sigma
        ) = param

    if score < too_blurry_thresh:
        enhanced = enhance_local_contrast_rgb(img_float, clip_limit=2.0)
        return enhanced, "too_blurry"

    if score > thresh_light:
        level = "light"
        amount = light_amount
        sigma = light_sigma
        clip_limit = light_clip
    elif score > thresh_medium:
        level = "medium"
        amount = med_amount
        sigma = med_sigma
        clip_limit = med_clip
    else:
        level = "heavy"
        amount = heavy_amount
        sigma = heavy_sigma
        clip_limit = heavy_clip

    contrast_boost = enhance_local_contrast_rgb(img_float, clip_limit=clip_limit)
    sharpened = highlight_aware_sharpen(contrast_boost, amount=amount, sigma=sigma)

    if level == "heavy":
        sharpened = highlight_aware_sharpen(
            sharpened,
            amount=extra_heavy_amount,
            sigma=extra_heavy_sigma,
        )

    return sharpened, level


def spatial_adaptive(src_dir, dst_dir, param):
    os.makedirs(dst_dir, exist_ok=True)

    all_files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(all_files)
    c_light = c_med = c_heavy = c_skip = 0

    print(f"Found {total} spatial blur images")
    print("Starting spatial_adaptive\n")

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(src_dir, fname)

        try:
            enhanced, level = _adaptive_single(in_path, param)
            out_path = os.path.join(dst_dir, fname)
            save_img(out_path, enhanced)

            if level == "light":
                c_light += 1
            elif level == "medium":
                c_med += 1
            elif level == "heavy":
                c_heavy += 1
            else:
                c_skip += 1

            progress = idx / total * 100.0 if total > 0 else 100.0
            print(f"[{idx}/{total}] {fname} | level: {level} | {progress:.1f}%")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\nDone spatial_adaptive")
    print(f" light:      {c_light}")
    print(f" medium:     {c_med}")
    print(f" heavy:      {c_heavy}")
    print(f" too_blurry: {c_skip}")
    print(f"output dir:  {dst_dir}")


# 老的 batch_process 保留 直接调用自适应 algo
def batch_process(input_dir, output_dir):
    return spatial_adaptive(input_dir, output_dir, param=None)


if __name__ == "__main__":
    in_dir = "data/D'/D'_annotated_base/spatial_blur/images"
    out_dir = "data/D'/D'_enhanced/spatial_blur/images"

    func = spatial_adaptive
    param = None

    func(in_dir, out_dir, param)
