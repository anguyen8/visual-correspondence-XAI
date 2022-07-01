import collections
import json
import os
import random
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from datasets import Dataset
from helper import HelperFunctions
from params import RunningParams
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class GPUParams(object):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(device=0))

        self.device = torch.device("cuda:0")


RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()

layer = 4

model = torchvision.models.resnet50(pretrained=True).eval()
feature_extractor = nn.Sequential(*list(model.children())[: layer - 6]).cuda()

# ---------- Parse argv -------------
val_datasets = sys.argv[1]
inat = (sys.argv[2] == 'True')
RunningParams.INAT = inat
if val_datasets == Dataset.CUB200 and RunningParams.INAT is True:
    RunningParams.Deformable_ProtoPNet = True

if RunningParams.Deformable_ProtoPNet:
    from cub200_features import get_resnet50_features

    model = get_resnet50_features(inat=RunningParams.INAT, pretrained=True)

    model = model.cuda()
    model.eval()
    model = nn.DataParallel(model)
    feature_extractor = model
else:
    if RunningParams.AP_FEATURE:
        feature_extractor = nn.Sequential(*list(model.children())[:-1]).cuda()
    model.cuda()

    feature_extractor = nn.DataParallel(feature_extractor)
    model = nn.DataParallel(model)

imagenet_train_data = ImageFolder(
    # ImageNet train folder
    root="/home/giang/Downloads/train/", transform=Dataset.imagenet_transform
)

print(RunningParams.__dict__)

for val_dataset in [val_datasets]:
    if val_dataset == Dataset.CUB200:
        imagenet_train_data = ImageFolder(
            # CUB train folder
            root="/home/giang/Downloads/cub200/train/",
            transform=Dataset.imagenet_transform,
        )
        RunningParams.DEEP_NN_TEST = False

    random.seed(42)
    np.random.seed(42)
    print(val_dataset)

    if (
        RunningParams.IMAGENET_REAL is True
    ):
        real_json = open("reassessed-imagenet/real.json")
        real_ids = json.load(real_json)
        real_labels = {
            f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels
            for i, labels in enumerate(real_ids)
        }

    # Load validation datasets
    if val_dataset in Dataset.IMAGENET_C_NOISE:
        imagenet_val_data = ImageFolder(
            root="/home/giang/Downloads/shared_datasets/imagenet-c/{}/".format(
                val_dataset
            ),
            transform=Dataset.imagenet_transform,
        )
    else:
        if (
            val_dataset in Dataset.ADVERSARIAL_PATCH
            or val_dataset == Dataset.DAMAGE_NET
        ):
            imagenet_val_data = ImageFolder(
                root="/home/giang/Downloads/{}/".format(val_dataset),
                transform=Dataset.imagenet_transform_crop_patch,
            )
        else:
            if val_dataset == Dataset.CUB200:
                imagenet_val_data = ImageFolder(
                    root="/home/giang/Downloads/cub200/{}/".format(val_dataset),
                    transform=Dataset.imagenet_transform,
                )
            else:
                if val_dataset in [Dataset.IMAGENET_1K_50K_CLEAN, Dataset.OBJECTNET_5K, Dataset.IMAGENET_A]:
                    imagenet_val_data = ImageFolder(
                        root="/home/giang/Downloads/shared_datasets/{}/".format(val_dataset),
                        transform=Dataset.imagenet_transform,
                    )
                else:
                    imagenet_val_data = ImageFolder(
                        root="/home/giang/Downloads/{}/".format(val_dataset),
                        transform=Dataset.imagenet_transform,
                    )

    if val_dataset in (
        [
            Dataset.IMAGENET_A,
            Dataset.IMAGENET_R,
            Dataset.OBJECTNET_5K,
            Dataset.IMAGENET_HARD,
            Dataset.IMAGENET_MULTI_OBJECT,
            Dataset.IMAGENET_PILOT_VIS,
        ]
        + Dataset.IMAGENET_C_NOISE
    ):  # Subset the train dataset
        imagenet_train_indices = [
            i
            for i in range(len(imagenet_train_data))
            if HelperFunctions.train_extract_wnid(imagenet_train_data.imgs[i][0])
            in imagenet_val_data.classes
        ]
        imagenet_train_set = torch.utils.data.Subset(
            imagenet_train_data, imagenet_train_indices
        )

    if len(imagenet_val_data) < 5000 or val_dataset in [
        Dataset.IMAGENET_MULTI_OBJECT,
        Dataset.IMAGENET_PILOT_VIS,
    ]:
        N_test = len(imagenet_val_data)
    elif val_dataset == Dataset.IMAGENET_1K_50K:
        N_test = 50000
    elif val_dataset == Dataset.IMAGENET_1K_50K_CLEAN:
        N_test = 46043
    elif val_dataset == Dataset.CUB200:
        N_test = 5794
    else:
        N_test = len(imagenet_val_data)

    # N_test = 4 # Just for test, remove to get the paper numbers

    # Subset the validation dataset
    random_indices = random.sample(range(0, len(imagenet_val_data)), N_test)
    val_set_tenth = torch.utils.data.Subset(imagenet_val_data, random_indices)

    bs = 512  # Batch Size

    # Load only the training samples of the targeted classes (e.g. 200 classes for ImageNet-A).
    if val_dataset in (
        [
            Dataset.IMAGENET_A,
            Dataset.IMAGENET_R,
            Dataset.OBJECTNET_5K,
            Dataset.IMAGENET_HARD,
            Dataset.IMAGENET_MULTI_OBJECT,
            Dataset.IMAGENET_PILOT_VIS,
        ]
        + Dataset.IMAGENET_C_NOISE
    ):
        train_loader = torch.utils.data.DataLoader(
            imagenet_train_set,
            batch_size=bs,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            imagenet_train_data,
            batch_size=bs,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    test_loader = torch.utils.data.DataLoader(
        val_set_tenth, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True
    )

    print(f"Total Images: {len(train_loader.dataset)} in Training folder")

    print(f"Total Images: {len(test_loader.dataset)} in Validation folder")
    print(f"Number of Classes: {len(test_loader.dataset.dataset.classes)}")

    # This class makes a dataset compatible with the original ImageNet1K dataset e.g. the indexing of classes
    class DatasetCompat:
        def __init__(self, train_loader, val_loader):
            if len(train_loader.dataset.classes) < len(val_loader.dataset.classes):
                raise ValueError("Validation set is not a subset of train set.")
            if not set(val_loader.dataset.classes) <= set(train_loader.dataset.classes):
                raise ValueError("Validation set is not a subset of train set.")

            self.train_loader = train_loader
            self.val_loader = val_loader

            self.train_indices = self.train_loader.dataset.class_to_idx
            self.train_num_class = len(train_loader.dataset.classes)

            self.val_indices = self.val_loader.dataset.class_to_idx
            self.val_num_class = len(val_loader.dataset.classes)

        # Convert ImageNet1K ID to the target dataset ID (e.g. 986 to 89)
        def convert_train_to_val_idx(self, train_idx):
            wnid = [k for (k, v) in self.train_indices.items() if v == train_idx][0]
            if wnid in self.val_indices:
                val_idx = self.val_indices[wnid]
            else:
                val_idx = -1
            return val_idx

        # Convert the target dataset ID to ImageNet1K ID (e.g. 89 to 986)
        def convert_val_to_train_idx(self, val_idx):
            wnid = [k for (k, v) in self.val_indices.items() if v == val_idx][0]
            train_idx = self.train_indices[wnid]
            return train_idx

    if val_dataset in (
        [
            Dataset.IMAGENET_A,
            Dataset.IMAGENET_R,
            Dataset.OBJECTNET_5K,
            Dataset.IMAGENET_HARD,
            Dataset.IMAGENET_MULTI_OBJECT,
            Dataset.IMAGENET_PILOT_VIS,
        ]
        + Dataset.IMAGENET_C_NOISE
    ):
        data_compat = DatasetCompat(imagenet_train_set, val_set_tenth)
    else:
        data_compat = DatasetCompat(train_loader, val_set_tenth)

    if RunningParams.DEEP_NN_TEST:
        dataset_wnids = val_set_tenth.dataset.classes
        mask = [wnid in dataset_wnids for wnid in Dataset.all_wnids]

        correct_ones = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                data = data.cuda()
                target_c = target.cuda()

                labels = HelperFunctions.to_np(target)
                out = model(data)
                model_output = out[:, mask]
                pred = model_output.data.max(1)[1]
                correct_ones += pred.eq(target_c.data).sum().item()

        acc = 100 * correct_ones / N_test
        print("{} Accuracy (%):".format(val_dataset), round(acc, 4))
        print("############################################################")

    all_val_embds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data = data.cuda()
            if RunningParams.Deformable_ProtoPNet:
                embeddings = HelperFunctions.to_np(
                    F.avg_pool2d(feature_extractor(data), (7, 7))
                )
            else:
                embeddings = HelperFunctions.to_np(feature_extractor(data))
            labels = HelperFunctions.to_np(target)

            # For datasets having < 1000 classes, we need to convert its label to ImageNet1K labels
            if val_dataset in (
                [
                    Dataset.IMAGENET_A,
                    Dataset.IMAGENET_R,
                    Dataset.OBJECTNET_5K,
                    Dataset.IMAGENET_HARD,
                    Dataset.IMAGENET_MULTI_OBJECT,
                    Dataset.IMAGENET_PILOT_VIS,
                ]
                + Dataset.IMAGENET_C_NOISE
            ):
                for i in range(len(labels)):
                    labels[i] = data_compat.convert_val_to_train_idx(labels[i])

            all_val_embds.append(embeddings)
            all_val_labels.append(labels)

    all_val_concatted = HelperFunctions.concat(all_val_embds)
    all_val_labels_concatted = HelperFunctions.concat(all_val_labels)

    #
    if RunningParams.AP_FEATURE:
        all_val_concatted = all_val_concatted.reshape(-1, 2048)
    else:
        all_val_concatted = all_val_concatted.reshape(-1, 2048 * 7 * 7)

    print(all_val_concatted.shape)
    print(all_val_labels_concatted.shape)

    Query = torch.from_numpy(all_val_concatted)
    Query = Query.cuda()
    Query = F.normalize(Query, dim=1)

    # Query must be shape (BatchSize, 100352) 100352=2048*7*7
    # Query must be normalized to have unit norm

    saved_results = []
    target_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # if batch_idx == 2:
            #     break
            data = data.cuda()
            labels = HelperFunctions.to_np(target)

            if RunningParams.Deformable_ProtoPNet:
                embeddings = F.avg_pool2d(feature_extractor(data), (7, 7))
            else:
                embeddings = feature_extractor(data)

            if RunningParams.AP_FEATURE:
                embeddings = embeddings.view(-1, 2048)
            else:
                embeddings = embeddings.view(-1, 2048 * 7 * 7)
            embeddings = F.normalize(embeddings, dim=1)
            q_results = torch.einsum("id,jd->ij", Query, embeddings).to("cpu")

            saved_results.append(q_results)
            target_labels.append(target)

    # Convert to numpy arrays
    labels_np = torch.cat(target_labels, -1)
    val_labels_np = np.concatenate(all_val_labels)

    saved_results = torch.cat(saved_results, 1)

    # Compute the top-1 accuracy of KNNs, save the KNN dictionary
    scores = {}
    import torch

    # K_values = [1,3,5,10,20,25,50,100,200]
    K_values = [20]

    for K in K_values:
        print(val_dataset)
        KNN_dict = {}
        KNN_dict["K_value"] = K
        KNN_dict["Test_labels"] = labels_np
        correct_cnt = 0
        duplicate_cnt = 0
        for i in tqdm(range(N_test)):
            KNN_dict[i] = {}
            X_ts_list = []
            concat_ts = saved_results[i].cuda()
            sorted_ts = torch.argsort(concat_ts).cuda()
            sorted_1k = sorted_ts[-50:].cuda()
            sorted_topk = sorted_ts[-K:]
            scores[i] = torch.flip(
                sorted_topk, dims=[0]
            )  # Move the closest to the head

            gt_id = val_labels_np[i]

            prediction = torch.argmax(torch.bincount(labels_np[scores[i]]))

            KNN_dict[i]["1000_NNs"] = sorted_1k.to("cpu")
            KNN_dict[i]["NNs_id_count_dict"] = collections.Counter(
                labels_np[scores[i]].numpy()
            )

            KNN_dict[i]["Predicted_id"] = prediction.to("cpu").numpy()
            KNN_dict[i]["Predicted_wnid"] = [
                k for (k, v) in data_compat.train_indices.items() if v == prediction
            ][0]

            # As we are subsetting train/val dataset, we need to convert back to the original ID of train/val dataset
            Query = [
                imagenet_val_data.samples[test_loader.dataset.indices[i]][0],
                HelperFunctions.val_extract_wnid(
                    imagenet_val_data.samples[test_loader.dataset.indices[i]][0]
                ),
            ]
            if val_dataset in [
                Dataset.IMAGENET_A,
                Dataset.IMAGENET_R,
                Dataset.OBJECTNET_5K,
                Dataset.IMAGENET_HARD,
                Dataset.IMAGENET_MULTI_OBJECT,
                Dataset.IMAGENET_PILOT_VIS,
            ]:
                NNs = [
                    [
                        imagenet_train_data.samples[
                            train_loader.dataset.indices[nn_idx]
                        ][0],
                        HelperFunctions.train_extract_wnid(
                            imagenet_train_data.samples[
                                train_loader.dataset.indices[nn_idx]
                            ][0]
                        ),
                    ]
                    for nn_idx in sorted_1k
                ]
            else:
                # Save 1K NNs
                NNs = [
                    [
                        imagenet_train_data.samples[nn_idx][0],
                        HelperFunctions.train_extract_wnid(
                            imagenet_train_data.samples[nn_idx][0]
                        ),
                    ]
                    for nn_idx in sorted_1k
                ]

            KNN_dict[i]["Query"] = Query
            KNN_dict[i]["NNs"] = NNs

            KNN_dict[i]["wnid"] = [NN[1] for NN in NNs]
            KNN_dict[i]["Unique_wnid_count"] = len(set(KNN_dict[i]["wnid"]))

            img_name = os.path.basename(Query[0])
            if RunningParams.IMAGENET_REAL is True:
                KNN_dict[i]["GT_id"] = real_labels[img_name]
                if prediction in real_labels[img_name]:
                    correctness = True
                else:
                    correctness = False
            else:
                KNN_dict[i]["GT_id"] = gt_id
                if prediction == gt_id:
                    correctness = True
                else:
                    correctness = False

            if correctness:
                correct_cnt += 1
                KNN_dict[i]["Output"] = True
            else:
                KNN_dict[i]["Output"] = False

        acc = 100 * correct_cnt / N_test

        print("The accuracy at K = {} is {}".format(K, acc))
        print("The duplicate count at K = {} is {}".format(K, duplicate_cnt))

        if RunningParams.KNN_RESULT_SAVE and KNN_dict and K == 20:
            if RunningParams.AP_FEATURE:
                # Save and read the KNN dict
                if RunningParams.Deformable_ProtoPNet is True:
                    HelperFunctions.check_and_mkdir("KNN_dict_Deform_ProtoP_AP")
                    np.save(
                        "KNN_dict_Deform_ProtoP_AP/KNN_dict_{}.npy".format(val_dataset),
                        KNN_dict,
                    )
                else:
                    HelperFunctions.check_and_mkdir("KNN_dict_AP")
                    np.save("KNN_dict_AP/KNN_dict_{}.npy".format(val_dataset), KNN_dict)
            else:
                # Save and read the KNN dict
                if RunningParams.Deformable_ProtoPNet is True:
                    HelperFunctions.check_and_mkdir("KNN_dict_Deform_ProtoP")
                    np.save(
                        "KNN_dict_Deform_ProtoP/KNN_dict_{}.npy".format(val_dataset),
                        KNN_dict,
                    )
                else:
                    HelperFunctions.check_and_mkdir("KNN_dict")
                    np.save("KNN_dict/KNN_dict_{}.npy".format(val_dataset), KNN_dict)
