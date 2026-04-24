import os
import shutil
import random
from bing_image_downloader import downloader

def setup_data():
    base_dir = "data"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    # 1. Clean existing directories (done manually via Powershell)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
            
    animals = ["Dog", "Elephant", "Lion", "Tiger"]
    
    # 2. Download Images to a temp dir
    temp_dir = "temp_downloads"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    for animal in animals:
        print(f"Downloading images for {animal}...")
        downloader.download(f"{animal} animal high quality", 
                            limit=25,  
                            output_dir=temp_dir, 
                            adult_filter_off=True, 
                            force_replace=False, 
                            timeout=60, 
                            verbose=False)
                            
        # The downloader creates a folder with the exact query name
        query_folder = os.path.join(temp_dir, f"{animal} animal high quality")
        
        if not os.path.exists(query_folder):
            print(f"Failed to find downloaded folder for {animal}")
            continue
            
        # Get all downloaded images
        images = [f for f in os.listdir(query_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # Split 5 for val, rest for train
        val_images = images[:5]
        train_images = images[5:]
        
        os.makedirs(os.path.join(train_dir, animal), exist_ok=True)
        os.makedirs(os.path.join(val_dir, animal), exist_ok=True)
        
        for img in train_images:
            shutil.move(os.path.join(query_folder, img), os.path.join(train_dir, animal, img))
            
        for img in val_images:
            shutil.move(os.path.join(query_folder, img), os.path.join(val_dir, animal, img))
            
    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    print("Data download and split completed successfully!")

if __name__ == "__main__":
    setup_data()
