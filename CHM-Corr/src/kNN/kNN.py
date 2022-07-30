import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
import argparse

# Helper functions
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to("cpu").numpy()


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.avgpool_output = None
        self.query = None
        self.cossim_value = {}

        def fw_hook(module, input, output):
            self.avgpool_output = output.squeeze()

        self.model.avgpool.register_forward_hook(fw_hook)

    def forward(self, input):
        _ = self.model(input)
        return self.avgpool_output

    def __repr__(self):
        return "Wrappper"


def run_knn(train_loader, test_loader):
    model = torchvision.models.resnet50(pretrained=True)
    model = model.cuda()
    model.eval()

    myw = Wrapper(model)
    list_for_all_val_emds = []

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data = data.cuda()
            embeddings = myw(data)
            list_for_all_val_emds.append(embeddings)

    val_concat = torch.concat(list_for_all_val_emds)

    del test_loader

    query = F.normalize(val_concat, dim=1)
    saved_results = []

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(train_loader)):
            data = data.cuda()
            embeddings = myw(data)
            embeddings = F.normalize(embeddings, dim=1)
            q_results = torch.einsum("id,jd->ij", query, embeddings)
            saved_results.append(q_results)

    del train_loader

    # Sort!
    knn_scores = {}
    for i in tqdm(range(len(val_concat))):
        knn_scores[i] = to_np(
            torch.argsort(torch.concat([X[i] for X in saved_results]))[-100:]
        )

    print("Saving Results on Disk")
    return knn_scores


def prepare_and_run(
    dataset_folder_path,
    dataset_name,
    imagenet_training_path,
    output_folder,
    transform,
    split,
    bs=512,
):
    train_dataset_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_size = 256
    if transform == "multi":
        val_size = 224

    val_dataset_transform = transforms.Compose(
        [
            transforms.Resize(val_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    imagenet_train_data = ImageFolder(
        root=imagenet_training_path, transform=train_dataset_transform
    )

    train_loader = torch.utils.data.DataLoader(
        imagenet_train_data,
        batch_size=bs,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    print(f"# of Training Folder Samples: {len(imagenet_train_data)}")

    val_dataset = ImageFolder(root=dataset_folder_path, transform=val_dataset_transform)

    dataset_size = len(val_dataset)
    split_size = dataset_size // split

    for l, start_index in enumerate(range(0, dataset_size, split_size)):
        end_index = start_index + split_size

        # Make a subset of validation set and run the kNN
        subset_indices = list(range(start_index, end_index))
        val_subset = torch.utils.data.Subset(val_dataset, subset_indices)

        test_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True
        )

        print(f"# of Test Folder Samples: {len(val_dataset)}")
        print(
            f"# of Samples in subset : {len(val_subset)} - Starting from {start_index} to {end_index}"
        )

        results = run_knn(
            train_loader,
            test_loader,
        )
        print("saving on the disk ...")
        with open(f"{output_folder}/{dataset_name}__{l}.pickle", "wb") as f:
            pickle.dump(results, f)

    # Merge results
    global_counter = 0
    merged_scores = {}

    for l, start_index in enumerate(range(0, dataset_size, split_size)):
        with open(f"{output_folder}/{dataset_name}__{l}.pickle", "rb") as f:
            slided_results = pickle.load(f)

        for i in range(len(slided_results)):
            merged_scores[global_counter] = slided_results[i]
            global_counter += 1

    with open(f"{output_folder}/{dataset_name}.pickle", "wb") as f:
        pickle.dump(merged_scores, f)

    # Calculate and Print Accuracy
    train_labels = np.asarray([x[1] for x in imagenet_train_data.imgs])
    val_labels = np.asarray([x[1] for x in val_dataset.imgs])

    for k in [3, 5, 10, 20, 25, 50, 100]:
        predictions = [
            np.argmax(np.bincount(train_labels[merged_scores[I][-k:][::-1]]))
            for I in range(len(merged_scores))
        ]
        final_accuracy = (
            100 * np.sum((np.asarray(predictions) == val_labels)) / len(val_labels)
        )
        print(f"K={k} -> Accuracy: {final_accuracy}")


def merge_results():
    pass


def main():
    parser = argparse.ArgumentParser(description="Doing kNN on Custom Dataset")
    parser.add_argument("--name", help="Name of Dataset", type=str, required=True)
    parser.add_argument("--val", help="Path to the Dataset", type=str, required=True)
    parser.add_argument("--train", help="Training folder Path", type=str, required=True)
    parser.add_argument(
        "--transform",
        help="Test transform for the validation set",
        type=str,
        default="single",
    )
    parser.add_argument("--out", help="Output folder Path", type=str, required=True)
    parser.add_argument(
        "--split",
        help="Number of splits to fit everything on GPU memory",
        type=int,
        required=True,
    )
    parser.add_argument("--bs", help="Batch size", type=int, default=512)

    args = parser.parse_args()
    prepare_and_run(
        args.val,
        args.name,
        args.train,
        args.out,
        args.transform,
        args.split,
        bs=args.bs,
    )


if __name__ == "__main__":
    main()
