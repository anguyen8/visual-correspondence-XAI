import argparse
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append("../pretrained-models.pytorch")
import pretrainedmodels

model_names = sorted(
    name
    for name in pretrainedmodels.__dict__
    if not name.startswith("__") and callable(pretrainedmodels.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="fbresnet152",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: fbresnet152)",
)
args = parser.parse_args()

# Load Model
model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained="imagenet")
model.eval()

# Load One Input Image
path_img = "data/ILSVRC2012_val_00002147.JPEG"
with open(path_img, "rb") as f:
    with Image.open(f) as img:
        input_data = img.convert(model.input_space)

tf = transforms.Compose(
    [
        transforms.Scale(round(max(model.input_size) * 1.143)),
        transforms.CenterCrop(max(model.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model.mean, std=model.std),
    ]
)

input_data = tf(input_data)  # 3x400x225 -> 3x299x299
input_data = input_data.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_data)
print(input)
exit()
# Load Imagenet Synsets
with open("data/imagenet_synsets.txt", "r") as f:
    synsets = f.readlines()

# len(synsets)==1001
# sysnets[0] == background
synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

with open("data/imagenet_classes.txt", "r") as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Make predictions
output = model(input)  # size(1, 1000)
max, argmax = output.data.squeeze().max(0)
class_id = argmax[0]
class_key = class_id_to_key[class_id]
classname = key_to_classname[class_key]

print(path_img, "is a", classname)
