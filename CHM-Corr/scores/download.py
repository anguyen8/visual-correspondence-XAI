import gdown

print("Downloading kNN Scores...")
print("ImageNet 1K")
gdown.download(
    url="https://drive.google.com/uc?id=1qTTL32RlsN6T9L4wtYM9EVNTZI2McYej",
    output="./ImageNet.pickle",
    quiet=False,
)

print("ImageNet-Sketch")
gdown.download(
    url="https://drive.google.com/uc?id=1AzGKZx1RwCDIouTJ1ij07ZWUbASXoAon",
    output="./ImageNet-Sketch.pickle",
    quiet=False,
)

print("ImageNet-R")
gdown.download(
    url="https://drive.google.com/uc?id=1AriTp3jcm88_lYJNJKLFMTEBScX0Z7D5",
    output="./ImageNet-R.pickle",
    quiet=False,
)

print("DAmageNet")
gdown.download(
    url="https://drive.google.com/uc?id=1H-P52tf4AG6Ick5A6MWuFQ0y2IT8qjSK",
    output="./DAmageNet.pickle",
    quiet=False,
)

print("Adversarials")
gdown.download(
    url="https://drive.google.com/uc?id=1e2T3bronoCd8fIVAg4mW-SPqZTjRGTmm",
    output="./Adversarials.pickle",
    quiet=False,
)

print("CUB iNaturalist")
gdown.download(
    url="https://drive.google.com/uc?id=1MTWkEvZAAj8fgni4sk-l96_dloXjh8zh",
    output="./CUB-iNaturalist.pickle",
    quiet=False,
)

print("CUB ResNet-50")
gdown.download(
    url="https://drive.google.com/uc?id=1zJXIWKFvwEYwac_5ubfWy2krtJA_P1Co",
    output="./CUB-ResNet-50.pickle",
    quiet=False,
)
