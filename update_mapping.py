import re

classes = ['Bear', 'Cat', 'Cheetah', 'Dog', 'Elephant', 'Fox', 'Giraffe', 'Hippo', 'Kangaroo', 'Lion', 'Monkey', 'Panda', 'Penguin', 'Rhino', 'Wolf', 'Zebra']
mapping = "CLASS_MAPPING = {\n"
for i, cls in enumerate(classes):
    mapping += f'    {i}: "{cls}",\n'
mapping += "}"

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace whatever CLASS_MAPPING is currently there
content = re.sub(r'CLASS_MAPPING = \{.*?\}', mapping, content, flags=re.DOTALL)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
