#!/usr/bin/env python3
import os
import shutil

def create_numbered_symlinks():
    source_dir = "/home/multy-surya/Grounded-SAM-2/lars_dataset/train/images"
    target_dir = "/home/multy-surya/Grounded-SAM-2/lars_temp_numbered"
    
    # Clean target directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files and sort them
    image_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # Create numbered symbolic links
    for i, filename in enumerate(image_files):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, f"{i:06d}.jpg")
        os.symlink(source_path, target_path)
        if i % 100 == 0:
            print(f"Created {i+1} symlinks...")
    
    print(f"Created {len(image_files)} numbered symbolic links in {target_dir}")
    return target_dir

if __name__ == "__main__":
    create_numbered_symlinks()