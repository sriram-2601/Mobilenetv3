import re

# Read app.py
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Read the mapping
with open("class_mapping.txt", "r", encoding="utf-8") as f:
    mapping_code = f.read()

# Replace the CLASS_MAPPING block using regex
# It starts with CLASS_MAPPING = { and ends with } followed by an empty line or something
# Let's find the exact string
old_mapping_pattern = re.compile(r"CLASS_MAPPING\s*=\s*\{.*?\n\}", re.DOTALL)
new_content = old_mapping_pattern.sub(mapping_code.strip(), content, count=1)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Updated app.py with 1000 classes!")
