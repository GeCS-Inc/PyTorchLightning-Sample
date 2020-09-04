import model
import train
from PIL import Image

from pytorch_lightning import seed_everything

env = train.env
transform = train.transform

IMAGE_PATH = "./dataset/cat/1200px-Cat03.jpg"
PRETRAINED = "lightning_logs/version_10/checkpoints/epoch=1.ckpt"

if __name__ == "__main__":
    seed_everything(42)

    model = model.FineTuningModel.load_from_checkpoint(PRETRAINED)
    model.eval()

    img = Image.open(IMAGE_PATH)
    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img).squeeze(0)
    print(
        '\n'.join([f"{i}: {v}" for i, v in enumerate(output.detach().numpy())]))
