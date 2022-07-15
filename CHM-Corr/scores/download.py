import gdown

print("Downloading kNN Scores...")
print("ImageNet 1K")
gdown.download(
    url="https://drive.google.com/uc?id=1-KdYgbb3t6V2DNGuRUpFCM9gO3l6umoM",
    output="./ImageNet.pickle",
    quiet=False,
)

print("ImageNet-Sketch")
gdown.download(
    url="https://drive.google.com/uc?id=1-PSlCPsdpqCExBsfclKcaDyhozH2HGTt",
    output="./ImageNet-Sketch.pickle",
    quiet=False,
)

print("ImageNet-R")
gdown.download(
    url="https://drive.google.com/uc?id=1-SXRTvKyqEQxBmVPg2YJLU0VXQsAjoUt",
    output="./ImageNet-R.pickle",
    quiet=False,
)

print("DAmageNet")
gdown.download(
    url="https://drive.google.com/uc?id=1-WGAlU3WvVz34NUKkTboEq2wIKajrTDt",
    output="./DAmageNet.pickle",
    quiet=False,
)

print("Adversarials")
gdown.download(
    url="https://drive.google.com/uc?id=1-HhKJwGCffOMe-WLtECgsCdxG25zkkmm",
    output="./Adversarials.pickle",
    quiet=False,
)

print("CUB iNaturalist")
gdown.download(
    url="https://drive.google.com/uc?id=1EMNacNPonhg1eL2jeQsbyyiePbh8F5Oo",
    output="./CUB-iNaturalist.pickle",
    quiet=False,
)

print("CUB ResNet-50")
gdown.download(
    url="https://drive.google.com/uc?id=1FGOkOmRWBTUteltnydj0ViYJW1BqmfVX",
    output="./CUB-ResNet-50.pickle",
    quiet=False,
)
