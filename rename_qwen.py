import os
import json
from safetensors.torch import load_file, save_file
path_list = ['./models/causalqwen/checkpoint-9850']

for checkpoint_path in path_list:
    output_path = checkpoint_path.replace("/check","_fixed/check")
    os.makedirs(output_path, exist_ok=True)

    # 1. process safetensors
    for filename in os.listdir(checkpoint_path):
        if filename.endswith(".safetensors"):
            print(f"Processing {filename}...")
            state_dict = load_file(os.path.join(checkpoint_path, filename))
            new_state_dict = {}
            
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_key = k.replace("model.", "language_model.", 1)
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            save_file(new_state_dict, os.path.join(output_path, filename))

    # 2. process index.json
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index_data = json.load(f)
        
        new_weight_map = {}
        for k, v in index_data["weight_map"].items():
            if k.startswith("model."):
                new_weight_map[k.replace("model.", "language_model.", 1)] = v
            else:
                new_weight_map[k] = v
        
        index_data["weight_map"] = new_weight_map
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)

    # 3. copy .json and .bin
    import shutil
    for filename in os.listdir(checkpoint_path):
        if not filename.endswith(".safetensors") and filename != "model.safetensors.index.json":
            shutil.copy(os.path.join(checkpoint_path, filename), os.path.join(output_path, filename))

    print("Done!")