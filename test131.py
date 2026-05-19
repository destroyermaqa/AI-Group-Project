import torch

# Load the entire model or a state dictionary
data = torch.load('unet_pet_segmentation.pth')
print(data)
