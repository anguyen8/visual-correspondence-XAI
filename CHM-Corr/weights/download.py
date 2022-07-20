import gdown

print("Downloading kNN Scores...")
print("pas_psi")
gdown.download(
    url="https://drive.google.com/uc?id=1yM1zA0Ews2I8d9-BGc6Q0hIAl7LzYqr0",
    output="./pas_psi.pt",
    quiet=False,
)

print("pas_iso")
gdown.download(
    url="https://drive.google.com/uc?id=18eP9DRKUp2HnsoHQD0jYasNgLGF7I_Fa",
    output="./pas_iso.pt",
    quiet=False,
)

print("pas_full")
gdown.download(
    url="https://drive.google.com/uc?id=1ZKO29Ba9QRNenvFRGuIX1e9yUELKvW3S",
    output="./pas_full.pt",
    quiet=False,
)

print("iNaturalist ResNet-50")
gdown.download(
    url="https://drive.google.com/uc?id=1URkZd1qlFolDeS77lRTNXybG8a87aeAO",
    output="BBN.iNaturalist2017.res50.90epoch.best_model.pth.pt",
    quiet=False,
)
