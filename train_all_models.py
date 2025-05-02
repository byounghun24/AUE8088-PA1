import os

model_list = [
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
]

for model_name in model_list:
    print(f"\n===== Training model: {model_name} =====")
    os.system(f"python efficientnet_b0tob7.py --model_name {model_name}")