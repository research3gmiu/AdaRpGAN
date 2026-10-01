import os
import torch
from torchvision.utils import save_image

def create_dummy_data(root_dir, n_samples=200, img_size=32):
    os.makedirs(root_dir, exist_ok=True)
    for i in range(n_samples):
        img = torch.rand(3, img_size, img_size)
        save_image(img, os.path.join(root_dir, f"img_{i}.png"))
            
if __name__ == "__main__":
    create_dummy_data("./data/dummy_images")
