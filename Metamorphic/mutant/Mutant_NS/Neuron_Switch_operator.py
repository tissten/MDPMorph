
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


data = torch.load("./Metamorphic/mutant/original_agent/TD3_actor.pht", map_location=torch.device('cpu'))

np.random.seed(0)



for i in range(1, num+1):
    mutant_dir = os.path.join("./Metamorphic/test", f"mutant_NS_{i}")

    os.makedirs(mutant_dir, exist_ok=True)


    layer_name = "act_net.model.2"  

    neuron_idx1, neuron_idx2 = np.random.choice(np.arange(0, 127), 2, replace=False)  


    weights = data[f"{layer_name}.weight"]  
    bias = data[f"{layer_name}.bias"]      


    weights[[neuron_idx1, neuron_idx2], :] = weights[[neuron_idx2, neuron_idx1], :]
    bias[[neuron_idx1, neuron_idx2]] = bias[[neuron_idx2, neuron_idx1]]

    data[f"{layer_name}.weight"] = weights
    data[f"{layer_name}.bias"] = bias




    model_path = os.path.join(mutant_dir, "TD3_actor.pht")

    torch.save(data, model_path)



print("Modified weights applied successfully!")
