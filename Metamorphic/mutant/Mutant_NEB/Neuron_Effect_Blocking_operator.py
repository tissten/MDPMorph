

import torch
import numpy as np
import os


import argparse
parser = argparse.ArgumentParser(description="Generate mutated models with inactive neurons.")
parser.add_argument("--num_mutants", type=int, default=20, help="Number of mutated models to generate.")
args = parser.parse_args()

num = args.num_mutants



data = torch.load("./Metamorphic/mutant/original_agent/TD3_actor.pht", map_location=torch.device('cpu'))




np.random.seed(0)


for i in range(1, num+1):
    mutant_dir = os.path.join("./Metamorphic/test", f"mutant_NEB_{i}")

    os.makedirs(mutant_dir, exist_ok=True)

    j = np.random.randint(0, 127)

    block_neuron_index = 128
    fc2_weights = data["act_net.model.2.weight"][j]

    fc2_weights[0:block_neuron_index] = 0

    data["act_net.model.2.weight"][j] = fc2_weights



    model_path = os.path.join(mutant_dir, "TD3_actor.pht")

    torch.save(data, model_path)

print("Modified weights applied successfully!")
