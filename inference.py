import torch
import torchvision.transforms as transforms
import numpy as np
import model
import data
import train
from utils import EasyDict, plot_dataset
from PIL import Image

env = train.env
transform = train.transform

IMAGE_PATH = "./dataset/cat/1200px-Cat03.jpg"

if __name__ == "__main__":
    model = model.FineTuningModel(env)

    img = Image.open(IMAGE_PATH)
    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img).squeeze(0)
    print(
        '\n'.join([f"{i}: {v}" for i, v in enumerate(output.detach().numpy())]))
