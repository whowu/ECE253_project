import os
import cv2
import numpy as np
from bm3d import bm3d, BM3DStages


def apply_bilateral(src_dir, dst_dir, param):
    os.makedirs(dst_dir, exist_ok=True)
    
    d, sigmaColor, sigmaSpace = param
    
    image_files = [
        fname for fname in os.listdir(src_dir)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    total = len(image_files)
    print(f"Processing {total} images using Bilateral Filtering")
    print(f"Parameters: d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}\n")
    
    for idx, fname in enumerate(image_files, start=1):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        
        try:
            img = cv2.imread(src_path)
            if img is None:
                print(f"[{idx}/{total}] Error: Could not read {fname}")
                continue
            
            denoised = cv2.bilateralFilter(
                img,
                d=d,
                sigmaColor=sigmaColor,
                sigmaSpace=sigmaSpace
            )
            
            cv2.imwrite(dst_path, denoised)
            
            print(f"[{idx}/{total}] {fname} processed")
            
        except Exception as e:
            print(f"[{idx}/{total}] Error processing {fname}: {e}")
    
    print(f"\nBilateral filtering completed. Output saved to: {dst_dir}")



def apply_cbm3d(src_dir, dst_dir, param):
    os.makedirs(dst_dir, exist_ok=True)
    
    sigma_psd = param
    
    image_files = [
        fname for fname in os.listdir(src_dir)
        if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    total = len(image_files)
    print(f"Processing {total} images using CBM3D")
    print(f"Parameters: sigma_psd={sigma_psd:.4f}\n")
    
    
    for idx, fname in enumerate(image_files, start=1):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        
        try:
            img = cv2.imread(src_path)
            if img is None:
                print(f"[{idx}/{total}] Error: Could not read {fname}")
                continue
            
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            
            try:
                denoised_rgb = bm3d(img_rgb, sigma_psd=sigma_psd, stage_arg=BM3DStages.ALL_STAGES)
            except (AttributeError, TypeError):
                try:
                    denoised_rgb = bm3d(img_rgb, sigma_psd=sigma_psd, stage_arg='all')
                except:
                    denoised_rgb = bm3d(img_rgb, sigma_psd=sigma_psd)
            
            
            denoised_rgb = np.clip(denoised_rgb, 0.0, 1.0)
            denoised_bgr = cv2.cvtColor((denoised_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            
            cv2.imwrite(dst_path, denoised_bgr)
            
            print(f"[{idx}/{total}] {fname} processed")
            
        except Exception as e:
            print(f"[{idx}/{total}] Error processing {fname}: {e}")
    
    print(f"\nCBM3D denoising completed. Output saved to: {dst_dir}")