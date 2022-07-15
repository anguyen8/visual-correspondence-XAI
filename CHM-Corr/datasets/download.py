import gdown
import zipfile
import os

print("Downloading dataset...")
print("Adversarials")
gdown.cached_download(
    url="https://drive.google.com/uc?id=1oehcahqVokBH-nfGTEdEgdK7Uc7vmzRt",
    path="./Adversarials.zip",
    quiet=False,
    md5="e0ea2005f800f125ff6502b7193a44c4",
)

print("Extracting ...")
if not os.path.exists("./adversarial_patches"):
    with zipfile.ZipFile("Adversarials.zip", "r") as zip_ref:
        zip_ref.extractall(".")
