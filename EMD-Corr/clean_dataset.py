import collections
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
from datasets import Dataset
from helper import HelperFunctions
from params import RunningParams
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()

Dataset.datasets = [Dataset.IMAGENET_1K_50K]

print(RunningParams.__dict__)

layer = 4
model = torchvision.models.resnet50(pretrained=True).eval()
feature_extractor = nn.Sequential(*list(model.children())[: layer - 6]).cuda()
feature_extractor = nn.DataParallel(feature_extractor)

imagenet_train_data = ImageFolder(
    root="/home/giang/Downloads/train/", transform=Dataset.imagenet_transform
)

for val_dataset in Dataset.datasets:
    random.seed(42)
    np.random.seed(42)
    print(val_dataset)

    real_json = open("reassessed-imagenet/real.json")
    real_ids = json.load(real_json)
    real_labels = {
        f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels for i, labels in enumerate(real_ids)
    }

    imagenet_val_data = ImageFolder(
        root="/home/giang/Downloads/shared_datasets/{}/".format(val_dataset),
        transform=Dataset.imagenet_transform,
    )

    N_test = 50000
    N_step = 5000
    bs = 512  # Batch Size

    train_loader = torch.utils.data.DataLoader(
        imagenet_train_data,
        batch_size=bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Subset the validation dataset
    # random_indices = random.sample(range(0, len(imagenet_val_data)), N_test)
    KNN_dict = {}
    correct_cnt = 0
    duplicate_cnt = 0
    no_labels = 0

    for start_idx in range(0, N_test, N_step):
        print(start_idx)
        random_indices = list(range(start_idx, start_idx + N_step))

        val_set_tenth = torch.utils.data.Subset(imagenet_val_data, random_indices)
        test_loader = torch.utils.data.DataLoader(
            val_set_tenth, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True
        )

        if RunningParams.DEEP_NN_TEST and False:
            correct_ones = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                    data = data.cuda()
                    target_c = target.cuda()

                    labels = HelperFunctions.to_np(target)
                    out = model(data)
                    pred = out.data.max(1)[1]
                    correct_ones += pred.eq(target_c.data).sum().item()

            acc = 100 * correct_ones / N_test
            print("{} Accuracy (%):".format(val_dataset), round(acc, 4))
            print("############################################################")

        all_val_embds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                data = data.cuda()
                embeddings = HelperFunctions.to_np(feature_extractor(data))
                labels = HelperFunctions.to_np(target)
                all_val_embds.append(embeddings)
                all_val_labels.append(labels)

        all_val_concatted = HelperFunctions.concat(all_val_embds)
        all_val_labels_concatted = HelperFunctions.concat(all_val_labels)

        #
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
                data = data.cuda()
                labels = HelperFunctions.to_np(target)

                embeddings = feature_extractor(data)
                embeddings = embeddings.view(-1, 2048 * 7 * 7)
                embeddings = F.normalize(embeddings, dim=1)
                q_results = HelperFunctions.to_np(
                    torch.einsum("id,jd->ij", Query, embeddings)
                )

                saved_results.append(q_results)
                target_labels.append(labels)

        # convert to numpy arrays
        labels_np = np.concatenate(target_labels)
        val_labels_np = np.concatenate(all_val_labels)

        # Compute the top-1 accuracy of KNNs, save the KNN dictionary
        scores = {}
        K = 20

        for i in tqdm(range(N_step)):
            Query = [
                imagenet_val_data.samples[test_loader.dataset.indices[i]][0],
                HelperFunctions.val_extract_wnid(
                    imagenet_val_data.samples[test_loader.dataset.indices[i]][0]
                ),
            ]
            img_name = os.path.basename(Query[0])
            gt_ids = real_labels[img_name]

            if len(gt_ids) != 1:
                no_labels += 1
                continue

            X_ts_list = []
            for X in saved_results:
                X_ts = HelperFunctions.to_ts(X[i])
                X_ts_list.append(X_ts)

            concat_ts = torch.cat(X_ts_list)
            sorted_ts = torch.argsort(concat_ts).to("cpu")
            sorted_topk = sorted_ts[-K:]
            scores[i] = torch.flip(
                sorted_topk.to("cpu"), dims=[0]
            )  # Move the closest to the head

            cosine_sim_topk = concat_ts[sorted_topk]
            max_cosine_sim = max(cosine_sim_topk)

            if max_cosine_sim > RunningParams.DUPLICATE_THRESHOLD:
                duplicate_cnt += 1
                continue

            KNN_dict[img_name] = {}
            NNs_id_count_dict = collections.Counter(labels_np[scores[i]])

            prediction = np.argmax(np.bincount(labels_np[scores[i]]))
            KNN_dict[img_name]["image-path"] = Query[0]
            KNN_dict[img_name]["imagenet-real-gt-ids"] = real_labels[img_name]
            KNN_dict[img_name]["predicted-id"] = prediction
            KNN_dict[img_name]["knn_confidence"] = NNs_id_count_dict[prediction]
            KNN_dict[img_name]["feature"] = "conv4_no_avg_pooling"

            if prediction in real_labels[img_name]:
                correctness = True
            else:
                correctness = False

            if correctness:
                correct_cnt += 1
                KNN_dict[img_name]["Correctness"] = True
            else:
                KNN_dict[img_name]["Correctness"] = False

        print(
            "The duplicate count at K = {} is {} and no label count = {}".format(
                K, duplicate_cnt, no_labels
            )
        )

    acc = 100 * correct_cnt / N_test

    print("The accuracy at K = {} is {}".format(K, acc))

    if RunningParams.KNN_RESULT_SAVE and KNN_dict and K == 20:
        HelperFunctions.check_and_mkdir("KNN_dict")
        np.save("KNN_dict/{}_clean.npy".format(val_dataset), KNN_dict)
