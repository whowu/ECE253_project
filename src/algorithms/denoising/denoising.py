import os
import cv2
import numpy as np
from bm3d import bm3d, BM3DStages

def estimate_noise_level(img_gray):
    laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
    sigma = np.median(np.abs(laplacian)) / 0.6745 
    
    if sigma < 20.0:
        return "light"
    elif sigma < 40.0:
        return "medium"
    else:
        return "heavy"



def apply_bilateral(img_bgr, noise_level):
    if noise_level == "light":
        d = 5
        sigmaColor = 50
        sigmaSpace = 50

    elif noise_level == "medium":
        d = 9
        sigmaColor = 75
        sigmaSpace = 75

    else:
        d = 15
        sigmaColor = 100
        sigmaSpace = 100
    
    denoised = cv2.bilateralFilter(
        img_bgr,
        d=d,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace
    )
    return denoised



def apply_cbm3d(img_bgr, noise_level):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    if noise_level == "light":
        sigma_psd = 15.0 / 255.0 
    elif noise_level == "medium":
        sigma_psd = 25.0 / 255.0 
    else:
        sigma_psd = 35.0 / 255.0 
    
    try:
        denoised_rgb = bm3d(img_rgb, sigma_psd=sigma_psd, stage_arg=BM3DStages.ALL_STAGES)
    except (AttributeError, TypeError):
        try:
            denoised_rgb = bm3d(img_rgb, sigma_psd=sigma_psd, stage_arg='all')
        except:
            denoised_rgb = bm3d(img_rgb, sigma_psd=sigma_psd)
    
    denoised_rgb = np.clip(denoised_rgb, 0.0, 1.0)
    denoised_bgr = cv2.cvtColor((denoised_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return denoised_bgr



def choose_technique(image_path, method='bilateral'):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_level = estimate_noise_level(gray)
    
    if method.lower() == 'bilateral':
        denoised_img = apply_bilateral(img, noise_level)
    elif method.lower() == 'cbm3d':
        denoised_img = apply_cbm3d(img, noise_level)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bilateral' or 'cbm3d'")
    
    return denoised_img, noise_level




def batch_process(input_dir, output_dir, method='bilateral'):
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = [
        fname for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    total = len(all_files)
    count_light, count_medium, count_heavy = 0, 0, 0
    
    print(f"Found {total} images")
    print(f"Starting adaptive {method.upper()} denoising...\n")
    
    for idx, fname in enumerate(all_files, start=1):
        in_path = os.path.join(input_dir, fname)
        
        try:
            denoised_img, noise_level = choose_technique(in_path, method)
            
  
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, denoised_img)
            
            if noise_level == "light":
                count_light += 1
            elif noise_level == "medium":
                count_medium += 1
            else:
                count_heavy += 1
            
            progress = (idx / total) * 100
            print(f"[{idx}/{total}] {fname} processed | noise level: {noise_level} | {progress:.1f}% done")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    
    print("\nAll processing completed.")
    print("Summary:")
    print(f" Light noise images:  {count_light}")
    print(f" Medium noise images: {count_medium}")
    print(f" Heavy noise images:  {count_heavy}")
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    in_folder = "data/D'/D'_annotated_base/noise/images"
    
 
    out_folder_bilateral = "data/D'/D'_enhanced/noise/images/bilateral"
    batch_process(in_folder, out_folder_bilateral, method='bilateral')
    
    print("\n" + "="*60 + "\n")
    
 
    out_folder_cbm3d = "data/D'/D'_enhanced/noise/images/cbm3d"
    batch_process(in_folder, out_folder_cbm3d, method='cbm3d')