
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

import argparse
parser = argparse.ArgumentParser(description="Generate mutated models with inactive neurons.")
parser.add_argument("--num_mutants", type=int, default=20, help="Number of mutated models to generate.")
args = parser.parse_args()

num = args.num_mutants

def fuzz_weights(state_dict, sigma=1.5):

    noise = torch.normal(0, sigma, size=state_dict.size())
    state_dict = state_dict + noise
    
    return state_dict


data = torch.load("./Metamorphic/mutant/original_agent/TD3_actor.pht", map_location=torch.device('cpu'))


np.random.seed(0)


for i in range(1, num+1):
    mutant_dir = os.path.join("./Metamorphic/mutant/Mutant_GF", f"mutant_GF_{i}")
    print(mutant_dir)
    os.makedirs(mutant_dir, exist_ok=True)


    sigma = 1.5  
    state_dict = data["act_net.model.2.weight"]  
    fuzzed_state_dict = fuzz_weights(state_dict, sigma=sigma)
    data["act_net.model.2.weight"] = fuzzed_state_dict

    model_path = os.path.join(mutant_dir, "TD3_actor.pht")

    torch.save(data, model_path)


print("Modified weights applied successfully!")


