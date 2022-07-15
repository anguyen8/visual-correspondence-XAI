import gdown
import zipfile
import os

print("Downloading dataset...")
print("ImageNet 1K")
gdown.cached_download(
    url="https://drive.google.com/uc?id=1ULkC_6oJyFwPs9I_Kr1HgOSIIR4AHWmV",
    path="./ILSVRC2012_img_val.zip",
    quiet=False,
    md5="f091bb4ef14c012a7c329c203ea95419",
)

print("Extracting ...")
if not os.path.exists("./ILSVRC2012_img_val"):
    with zipfile.ZipFile("ILSVRC2012_img_val.zip", "r") as zip_ref:
        zip_ref.extractall(".")


print("ImageNet-Sketch")
gdown.cached_download(
    url="https://drive.google.com/uc?id=1QjD5GrDNvNJCY9WFFLJDeKzLXPI-FnGu",
    path="./ImageNet-Sketch.zip",
    quiet=False,
    md5="884e58a8742a8413e758419a489e919c",
)

print("Extracting ...")
if not os.path.exists("./imagenet-sketch"):
    with zipfile.ZipFile("ImageNet-Sketch.zip", "r") as zip_ref:
        zip_ref.extractall(".")


print("ImageNet-R")
gdown.cached_download(
    url="https://drive.google.com/uc?id=1zOmbF9sdB9tel0dIngWL4ZqmXfc85O4G",
    path="./ImageNet-R.zip",
    quiet=False,
    md5="47272996687ba8b8689904280e1fce2e",
)

print("Extracting ...")
if not os.path.exists("./imagenet-r"):
    with zipfile.ZipFile("ImageNet-R.zip", "r") as zip_ref:
        zip_ref.extractall(".")


print("DAmageNet")
gdown.cached_download(
    url="https://drive.google.com/uc?id=1kKAtPSjdaO7ZTfMwn5XKPI7gQkdDlLoX",
    path="./DAmageNet.zip",
    quiet=False,
    md5="7c00aec1912326ab123177984bf5b712",
)

print("Extracting ...")
if not os.path.exists("./DAmageNet"):
    with zipfile.ZipFile("DAmageNet.zip", "r") as zip_ref:
        zip_ref.extractall(".")


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
