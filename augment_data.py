import os
from PIL import Image, ImageEnhance
import random

def augment_image(img, i):
    factor_color = random.uniform(0.8, 1.2)
    factor_contrast = random.uniform(0.8, 1.2)
    factor_brightness = random.uniform(0.8, 1.2)
    
    img = ImageEnhance.Color(img).enhance(factor_color)
    img = ImageEnhance.Contrast(img).enhance(factor_contrast)
    img = ImageEnhance.Brightness(img).enhance(factor_brightness)
    
    if random.choice([True, False]):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    return img

animals_and_files = {
    "Lion": "Lion.jpeg",
    "Elephant": "Elephant.jpg",
    "Dog": "Dog.jpeg",
    "Cheetah": "Cheetah.jpg",
    "Cat": "Cat.jpeg",
    "Monkey": "Monkey.jpg"
}

train_dir = os.path.join("data", "train")
val_dir = os.path.join("data", "val")

for animal, filename in animals_and_files.items():
    if not os.path.exists(filename):
        print(f"Source file {filename} not found!")
        continue
        
    print(f"Generating data for {animal}...")
    os.makedirs(os.path.join(train_dir, animal), exist_ok=True)
    os.makedirs(os.path.join(val_dir, animal), exist_ok=True)
    
    img = Image.open(filename).convert('RGB')
    
    # 25 train images
    for i in range(25):
        aug_img = augment_image(img.copy(), i)
        aug_img.save(os.path.join(train_dir, animal, f"aug_train_{i}.jpg"))
        
    # 5 val images
    for i in range(5):
        aug_img = augment_image(img.copy(), i)
        aug_img.save(os.path.join(val_dir, animal, f"aug_val_{i}.jpg"))
        
print("Data generation complete!")
