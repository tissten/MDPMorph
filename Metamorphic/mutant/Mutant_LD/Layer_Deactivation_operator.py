
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


class MyModel(nn.Module):
    def __init__(self, state_size=24, action_size=4, fc1_units=256, fc2_units=256):

        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)


    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    



data = torch.load(r"F:\UoA tool paper\MDPMorph_open_source\Metamorphic\mutant\original_agent\TD3_actor.pht", map_location=torch.device('cpu'))
model = MyModel()

new_state_dict = {}
for key, value in data.items():
    if key.startswith("act_net.model.0."):
        new_key = key.replace("act_net.model.0.", "fc1.")
    elif key.startswith("act_net.model.2."):
        new_key = key.replace("act_net.model.2.", "fc2.")
    elif key.startswith("act_net.model.4."):
        new_key = key.replace("act_net.model.4.", "fc3.")
    else:
        new_key = key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)





for i in range(1, num+1):
    mutant_dir = os.path.join(r"F:\UoA tool paper\MDPMorph_open_source\Metamorphic\test", f"mutant_LD_{i}")

    os.makedirs(mutant_dir, exist_ok=True)

    with torch.no_grad():

        model.fc2.weight[40:90] = 0

        model.fc2.bias[40:90] = 0

    data1 = model.state_dict()
    new_state_dict = {}
    for key, value in data1.items():

        if key.startswith("fc1."):
            new_key = key.replace("fc1.", "act_net.model.0.")
        elif key.startswith("fc2."):
            new_key = key.replace("fc2.", "act_net.model.2.")
        elif key.startswith("fc3."):
            new_key = key.replace("fc3.", "act_net.model.4.")
        else:
            new_key = key
        new_state_dict[new_key] = value
    


    model_path = os.path.join(mutant_dir, "TD3_actor.pht")

    torch.save(new_state_dict, model_path)


print("Modified weights applied successfully!")
