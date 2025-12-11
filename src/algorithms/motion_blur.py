import os
import cv2
import numpy as np
from skimage import restoration, img_as_float


def estimate_blur_level(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    score = lap.var()
    if score > 220.0:
        return "light"
    elif score > 120.0:
        return "medium"
    else:
        return "heavy"

def _build_psf_impl(size, thickness=3):
    psf = np.zeros((size, size))
    center = size // 2
    cv2.line(psf, (0, center), (size, center), 1, thickness=thickness)
    psf /= psf.sum()
    return psf

def build_psf(src_dir, dst_dir, param):
    """
    - src_dir: images input directory
    - dst_dir: images output directory
    - param: list:
        [size] or [size, thickness]
    """
    if param is None:
        # by default
        param = [15, 3]

    if not isinstance(param, list):
        raise ValueError("build_psf: param 必须是 list 或 None")

    if len(param) == 1:
        size = int(param[0])
        thickness = 3
    elif len(param) == 2:
        size = int(param[0])
        thickness = int(param[1])
    else:
        raise ValueError("build_psf: param 应该是 [size] 或 [size, thickness]")

    return _build_psf_impl(size, thickness=thickness)


def _process_single_image(image_path, param):
    """
    None: take default
    list: [psf_light, psf_medium, psf_heavy,
            iter_light, iter_medium, iter_heavy]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_level = estimate_blur_level(gray)

    # 解析 param
    if param is not None:
        if not isinstance(param, list):
            raise ValueError("process_adaptive: param 必须是 list 或 None")

        if len(param) == 6:
            psf_light, psf_medium, psf_heavy, iter_light, iter_medium, iter_heavy = param
        else:
            raise ValueError(
                "process_adaptive: param 列表格式应为 "
                "[psf_light, psf_medium, psf_heavy, iter_light, iter_medium, iter_heavy]"
            )
    else:
        # use default
        psf_light, psf_medium, psf_heavy = 9, 15, 21
        iter_light, iter_medium, iter_heavy = 6, 15, 22

    if blur_level == "light":
        psf_size = psf_light
        rl_iters = iter_light
    elif blur_level == "medium":
        psf_size = psf_medium
        rl_iters = iter_medium
    else:
        psf_size = psf_heavy
        rl_iters = iter_heavy

    psf = _build_psf_impl(psf_size)

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


def process_adaptive(src_dir, dst_dir, param):
    """
    - src_dir: input images directory
    - dst_dir: output images directory
    - param:
        None: use default
        list: [psf_light, psf_medium, psf_heavy,
               iter_light, iter_medium, iter_heavy]
    """
    os.makedirs(dst_dir, exist_ok=True)

    all_files = [
        fname for fname in os.listdir(src_dir)
        if fname.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    total = len(all_files)
    count_light, count_medium, count_heavy = 0, 0, 0

    print(f"Found {total} images")
    print("Starting adaptive processing...\n")

    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(src_dir, fname)

        try:
            deconv_img, level = _process_single_image(in_path, param)

            save_img(os.path.join(dst_dir, fname), deconv_img)

            if level == "light":
                count_light += 1
            elif level == "medium":
                count_medium += 1
            else:
                count_heavy += 1

            progress = (idx / total) * 100 if total > 0 else 100.0
            print(
                f"[{idx}/{total}] {fname} processed | "
                f"blur level: {level} | {progress:.1f}% done"
            )

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\nAll processing completed.")
    print("Summary:")
    print(f" Light blur images:  {count_light}")
    print(f" Medium blur images: {count_medium}")
    print(f" Heavy blur images:  {count_heavy}")
    print(f"\nOutput saved to: {dst_dir}")


def batch_process(input_dir, output_dir):
    return process_adaptive(input_dir, output_dir, param=None)


if __name__ == "__main__":
    in_folder = "data/D'/D'_annotated_base/motion_blur"
    out_folder = "data/D'/D'_enhanced/motion_blur/ablation_study"
    process_adaptive(in_folder, out_folder, param=None)

    experiments = {
    "default_9_15_21_6_15_22": [9, 15, 21, 6, 15, 22],

    "heavier_heavy_psf25_iter30": [9, 15, 25, 6, 15, 30],

    "soft_light_strong_heavy": [7, 13, 27, 4, 12, 32],

    "smaller_psf_more_iter": [7, 11, 17, 8, 18, 28],
    }


    for name, param in experiments.items():
        out_dir = os.path.join(out_folder, name)
        print(f"\n====== Running experiment: {name} ======")
        print(f"param = {param}")
        process_adaptive(in_folder, out_dir, param)
