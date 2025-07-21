import torch

# data = torch.load(r"E:\UoA_BipedalWalker\Metamorphic\mutant\original_agent\TD3_actor.pht", map_location=torch.device('cpu'))

data = torch.load(r"E:\mutant_LD_1\TD3_actor.pht", map_location=torch.device('cpu'))

# 加载 .pth 文件
# file_path = r"E:\UoA_BipedalWalker\Metamorphic\mutant\original_agent\TD3_actor.pht"
# data = torch.load(file_path)

# 查看文件内容
print(data["fc2.weight"][30:90])  # 输出文件中包含的键，例如 state_dict、optimizer_state 等