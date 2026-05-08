import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 1. Load the pre-trained ImageNet model
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)

# 2. Save its state dict exactly like train.py does
torch.save(model.state_dict(), "mobilenet_v3.pt")
print("Saved 1000-class mobilenet_v3.pt successfully!")

# 3. Generate the CLASS_MAPPING for app.py
categories = weights.meta["categories"]
with open("class_mapping.txt", "w", encoding='utf-8') as f:
    f.write("CLASS_MAPPING = {\n")
    for idx, category in enumerate(categories):
        # Escape any quotes
        safe_cat = category.replace('"', '\\"')
        f.write(f'    {idx}: "{safe_cat}",\n')
    f.write("}\n")
print("Generated class_mapping.txt successfully!")
