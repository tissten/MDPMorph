

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


class InvertedActivation(nn.Module):
    def __init__(self, activation, mutate=True):

        super(InvertedActivation, self).__init__()
        self.activation = activation
        self.mutate = mutate

    def forward(self, x):
        if self.mutate:
            x = -x  
        return self.activation(x)



def replace_activation_modules(module, mutate=True):

    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):

            setattr(module, name, InvertedActivation(child, mutate))
        else:
            replace_activation_modules(child, mutate)



data = torch.load("./Metamorphic/mutant/original_agent/TD3_actor.pht", map_location=torch.device('cpu'))
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
    mutant_dir = os.path.join("./Metamorphic/test", f"mutant_NAI_{i}")
    print(mutant_dir)
    os.makedirs(mutant_dir, exist_ok=True)

    with torch.no_grad():
        model.fc2.weight[int(a*10):int(a*10+40)] *= -1
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
