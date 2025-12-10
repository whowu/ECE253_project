import os
import cv2
import numpy as np
from skimage import restoration, img_as_float


def estimate_blur_level(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    score = lap.var()
    if score > 150.0:
        return "light"
    elif score > 60.0:
        return "medium"
    else:
        return "heavy"


def build_psf(size, thickness=3):
    psf = np.zeros((size, size))
    center = size // 2
    cv2.line(psf, (0, center), (size, center), 1, thickness=thickness)
    psf /= psf.sum()
    return psf


def process_adaptive(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_level = estimate_blur_level(gray)

    if blur_level == "light":
        psf_size = 9
        rl_iters = 6
    elif blur_level == "medium":
        psf_size = 15
        rl_iters = 15
    else:
        psf_size = 21
        rl_iters = 22

    psf = build_psf(psf_size)

    final_deconv = np.zeros_like(img_float)

    for i in range(3):
        res = restoration.richardson_lucy(
            img_float[:, :, i],
            psf,
            num_iter=rl_iters
        )
        res = np.nan_to_num(res)
        final_deconv[:, :, i] = np.clip(res, 0, 1)

    return final_deconv, blur_level


def save_img(out_path, img_float_rgb):
    img_uint8 = (img_float_rgb * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_bgr)


def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_files = [
        fname for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    total = len(all_files)
    count_light, count_medium, count_heavy = 0, 0, 0

    print(f"Found {total} images")
    print("Starting adaptive processing...\n")

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(input_dir, fname)

        try:
            deconv_img, level = process_adaptive(in_path)

            # Save result using exactly same name
            save_img(os.path.join(output_dir, fname), deconv_img)

            if level == "light":
                count_light += 1
            elif level == "medium":
                count_medium += 1
            else:
                count_heavy += 1

            progress = (idx / total) * 100
            print(f"[{idx}/{total}] {fname} processed | blur level: {level} | {progress:.1f}% done")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\nAll processing completed.")
    print("Summary:")
    print(f" Light blur images:  {count_light}")
    print(f" Medium blur images: {count_medium}")
    print(f" Heavy blur images:  {count_heavy}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    in_folder = "data/D'/D'_annotated_base/motion_blur/images"
    out_folder = "data/D'/D'_enhanced/motion_blur/images"
    batch_process(in_folder, out_folder)
