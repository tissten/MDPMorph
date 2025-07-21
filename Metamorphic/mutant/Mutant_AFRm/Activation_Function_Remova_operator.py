
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


parser = argparse.ArgumentParser(description="Generate mutated models with inactive neurons.")
parser.add_argument("--num_mutants", type=int, default=20, help="Number of mutated models to generate.")
args = parser.parse_args()

num = args.num_mutants

class CustomNet(nn.Module):
    def __init__(self, state_size=28, action_size=1, fc1_units=256, fc2_units=256, inactive_neurons=None):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)  
        self.fc2 = nn.Linear(fc1_units, fc2_units)  
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.activation = nn.ReLU()  
        self.inactive_neurons = inactive_neurons if inactive_neurons.size>0 else []

    def forward(self, x):
        x = self.fc1(x)  
        x = self.custom_activation(x)  

        x = self.fc2(x)  
        x = self.custom_activation(x) 
        
        x = self.fc3(x)  
        return x

    def custom_activation(self, x):

        mask = torch.ones_like(x)
        mask[:, self.inactive_neurons] = 0  
        activated = self.activation(x) * mask + x * (1 - mask)  
        return activated




data = torch.load("./Metamorphic/mutant/original_agent/TD3_critic.pht", map_location=torch.device('cpu'))




np.random.seed(0)

for i in range(1, num+1):
    mutant_dir = os.path.join(r"./Metamorphic/mutant/Mutant_AFRm", f"mutant_AFRm_{i}")

    os.makedirs(mutant_dir, exist_ok=True)


    inactive_neurons = np.random.randint(low=0, high=255, size=(50,))

    modified_model = CustomNet(inactive_neurons=inactive_neurons)
    
    new_state_dict = {}
    for key, value in data.items():
        if key.startswith("Q1.model.0."):
            new_key = key.replace("Q1.model.0.", "fc1.")
        elif key.startswith("Q1.model.2."):
            new_key = key.replace("Q1.model.2.", "fc2.")
        elif key.startswith("Q1.model.4."):
            new_key = key.replace("Q1.model.4.", "fc3.")
        else:
            continue
        new_state_dict[new_key] = value

    modified_model.load_state_dict(new_state_dict)  

    model_path = os.path.join(mutant_dir, "TD3_critic.pht")

    torch.save(data, model_path)



print("Modified weights applied successfully!")
