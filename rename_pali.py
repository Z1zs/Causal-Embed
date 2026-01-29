import os
import json
import shutil
from safetensors.torch import load_file, save_file


path_list = [
    './models/causalpali/checkpoint-8865',
]

KEY_MAPPINGS = {
    "model.language_model.lm_head": "model.lm_head",
    "model.language_model.model": "model.model.language_model",
    "model.vision_tower": "model.model.vision_tower",
    "model.multi_modal_projector": "model.model.multi_modal_projector",
}

def rename_key(key):
    for old_prefix, new_prefix in KEY_MAPPINGS.items():
        if key.startswith(old_prefix):
            return key.replace(old_prefix, new_prefix, 1)
    return key

for checkpoint_path in path_list:
    output_path = checkpoint_path.replace("/check", "_fixed/check")
    os.makedirs(output_path, exist_ok=True)
    print(f"\nProcessing: {checkpoint_path} -> {output_path}")

    # 1. process .safetensors
    for filename in os.listdir(checkpoint_path):
        if filename.endswith(".safetensors"):
            print(f"  Converting {filename}...")
            file_path = os.path.join(checkpoint_path, filename)
            
            state_dict = load_file(file_path)
            new_state_dict = {}
            
            for k, v in state_dict.items():
                new_key = rename_key(k)
                new_state_dict[new_key] = v
                
                if k != new_key and len(new_state_dict) == 1:
                   print(f"    [Example] {k} -> {new_key}")
            
            save_file(new_state_dict, os.path.join(output_path, filename))

    # 2. process index.json
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print("  Converting model.safetensors.index.json...")
        with open(index_file, "r") as f:
            index_data = json.load(f)
        
        new_weight_map = {}
        for k, v in index_data["weight_map"].items():
            new_key = rename_key(k)
            new_weight_map[new_key] = v
        
        index_data["weight_map"] = new_weight_map
        
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)

    # 3. copy .json and .bin files
    print("  Copying other files...")
    for filename in os.listdir(checkpoint_path):
        if not filename.endswith(".safetensors") and filename != "model.safetensors.index.json":
            src = os.path.join(checkpoint_path, filename)
            dst = os.path.join(output_path, filename)
            if os.path.isfile(src):
                shutil.copy(src, dst)

    print("Done!")