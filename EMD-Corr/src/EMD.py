import collections
import multiprocessing
import os
import pickle
import random
import json
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb
import sys
from datasets import Dataset
from emd_utils import EMDFunctions
from helper import HelperFunctions
from params import RunningParams
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from visualize import Visualization


class GPUParams(object):
    def __init__(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(device=0))

        self.device = torch.device("cuda:4")

# The purpose of instance methods is to set or get details about instances (objects).
# The purpose of the class methods is to set or get the details (status) of the class.
# Static methods cannot access the class data.
# Since they are not attached to any class attribute, they cannot get or set the instance state or class state.


RunningParams = RunningParams()
Visualization = Visualization()
Dataset = Dataset()
HelperFunctions = HelperFunctions()

wandb.init(
    project="my-test-project",
    entity="luulinh90s",
    config={"datasets": Dataset.datasets, "Running Params": RunningParams},
)

if RunningParams.Deformable_ProtoPNet is True:
    pass
else:
    model = torchvision.models.resnet50(pretrained=True).eval().cuda()
    feature_extractor = nn.Sequential(*list(model.children())[: 4 - 6]).cuda()

# ---------- Parse argv -------------
val_datasets = sys.argv[1]
vis_flag = (sys.argv[2] == 'True')
inat = (sys.argv[3] == 'True')
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
    feature_extractor = nn.DataParallel(feature_extractor)
    model = nn.DataParallel(model)

imagenet_train_data = ImageFolder(
    # Your path to ImageNet train dataset
    root="/home/giang/Downloads/train/", transform=Dataset.imagenet_transform
)

exported_results = {}
RunningParams.VISUALIZATION = vis_flag

print(RunningParams.__dict__)

with torch.no_grad():
    for val_dataset in [val_datasets]:
        if val_dataset == Dataset.CUB200:
            imagenet_train_data = ImageFolder(
                # Your path to CUB train dataset
                root="/home/giang/Downloads/cub200/train/",
                transform=Dataset.imagenet_transform,
            )
            RunningParams.DEEP_NN_TEST = False
            RunningParams.IMAGENET_REAL = False

        random.seed(42)
        np.random.seed(42)
        print("Running {} ...".format(val_dataset))

        exported_results["dataset_name"] = val_dataset
        exported_results["prediction_info"] = {}
        exported_results["N"] = RunningParams.K_value
        exported_results["K"] = RunningParams.MajorityVotes_K

        if RunningParams.UNIFORM:
            att = "UNIFORM"
        else:
            att = "CROSS"

        print(vars(RunningParams))

        N_test = RunningParams.N_test

        if RunningParams.IMAGENET_REAL is True:
            real_json = open("reassessed-imagenet/real.json")
            real_ids = json.load(real_json)
            real_labels = {
                f"ILSVRC2012_val_{i + 1:08d}.JPEG": labels
                for i, labels in enumerate(real_ids)
            }

        if val_dataset == Dataset.IMAGENET_HARD:
            N_test = 2894
        elif val_dataset == Dataset.IMAGENET_MULTI_OBJECT:
            N_test = 6786
        elif val_dataset == Dataset.IMAGENET_PILOT_VIS:
            N_test = 6000
        elif val_dataset == Dataset.IMAGENET_1K_50K:
            N_test = 50000
        elif val_dataset == Dataset.IMAGENET_1K_50K_CLEAN:
            N_test = 46043
        elif val_dataset == Dataset.CUB200:
            N_test = 5794

        if RunningParams.AP_FEATURE:
            if RunningParams.Deformable_ProtoPNet is True:
                KNN_dict = np.load(
                    "KNN_dict_Deform_ProtoP_AP/KNN_dict_{}.npy".format(val_dataset),
                    allow_pickle="False",
                ).item()
            else:
                print("AP with ResNet")
                KNN_dict = np.load(
                    "KNN_dict_AP/KNN_dict_{}.npy".format(val_dataset),
                    allow_pickle="False",
                ).item()
        else:
            if RunningParams.Deformable_ProtoPNet is True:
                KNN_dict = np.load(
                    "KNN_dict_Deform_ProtoP/KNN_dict_{}.npy".format(val_dataset),
                    allow_pickle="False",
                ).item()
            else:
                KNN_dict = np.load(
                    "KNN_dict/KNN_dict_{}.npy".format(val_dataset), allow_pickle="False"
                ).item()

        K_value = RunningParams.K_value
        if RunningParams.CLOSEST_PATCH_COMPARISON:
            num_patches = [5] # Using only top-5 most similar patch pairs
        else:
            num_patches = [49 * 49] # Using all possible pairs

        train_id_to_wnid = dict(
            (v, k) for k, v in imagenet_train_data.class_to_idx.items()
        )
        
        N_test = len(KNN_dict) - 2
        
        for num_patch in num_patches:
            print("K is {}| Num patch is {}".format(K_value, num_patch))
            correct_count = 0
            correct_count_KNN = 0
            correct_count_DNN = 0

            wrong2crt_cnt = 0
            crt2wrong_cnt = 0

            vis_cnt = 0
            vis_cnt_dict = {
                "EMD_correct": 0,
                "KNN_correct": 0,
                "Both_correct": 0,
                "Both_wrong": 0,
            }

            correct_predictions = {}
            wrong_predictions = {}
            scores = {}
            no_candidate = 0
            patch_0_num_total = 0

            for i in tqdm(range(N_test)):
                vis_cnt_total = sum([i for i in vis_cnt_dict.values()])

                scores[i] = KNN_dict[i]["1000_NNs"][-K_value:].to("cpu")
                Query = KNN_dict[i]["Query"]
                base_path_name = os.path.basename(Query[0])

                NNs = KNN_dict[i]['NNs'][-K_value:]

                if RunningParams.IMAGENET_REAL is True:
                    gt_ids = real_labels[base_path_name]
                    reaL_gt_wnids = []
                    if len(gt_ids) == 0:
                        continue
                    else:
                        for gt_id in gt_ids:
                            gt_wnid = train_id_to_wnid[gt_id]
                            reaL_gt_wnids.append(gt_wnid)
                else:
                    reaL_gt_wnids = []

                images = []
                img_paths = [Query[0]] + [NN[0] for NN in NNs]
                for path in img_paths:
                    image = PIL.Image.open(path)
                    if HelperFunctions.is_grey_scale(path) and image.mode not in [
                        "RGBA",
                        "CMYK",
                    ]:
                        if (
                            val_dataset == Dataset.ADVERSARIAL_PATCH_NEW
                            or val_dataset == Dataset.DAMAGE_NET
                            and path == Query[0]
                        ):
                            transform_img = Dataset.gray_transform_crop_patch(
                                image
                            ).unsqueeze(0)
                        else:
                            transform_img = Dataset.gray_transform(image).unsqueeze(0)

                        if (
                            transform_img.shape[1] == 1
                        ):  # Grey images have first dim is 1
                            # Stacking keeps the content the same
                            transform_img = torch.cat(
                                [transform_img, transform_img, transform_img], dim=1
                            )
                        else:
                            transform_img = transform_img
                    else:
                        if image.mode in ["RGBA", "CMYK"]:
                            image = image.convert("RGB")

                        if (
                            val_dataset == Dataset.ADVERSARIAL_PATCH_NEW
                            or val_dataset == Dataset.DAMAGE_NET
                            and path == Query[0]
                        ):
                            transform_img = Dataset.imagenet_transform_crop_patch(
                                image
                            ).unsqueeze(0)
                        else:
                            transform_img = Dataset.imagenet_transform(image).unsqueeze(
                                0
                            )
                    images.append(transform_img)

                data = torch.stack(images, dim=0)
                data = torch.squeeze(data).cuda()

                fb = feature_extractor(data)

                if RunningParams.DIML_FEAT:
                    fb_center = fb

                (
                    emd_distance,
                    q2g_att,
                    g2q_att,
                    opt_plan,
                ) = EMDFunctions.compute_emd_distance(
                    K_value, fb_center, fb, RunningParams.UNIFORM, num_patch
                )

                emd_distance = emd_distance[
                    1:
                ]  # Remove the Query itself from distance array
                sorted_ts = torch.argsort(emd_distance)
                sorted_top20_dist = sorted_ts[-RunningParams.MajorityVotes_K :].to(
                    "cpu"
                )  # ID in 50 images
                sorted_top20_dist = torch.flip(
                    sorted_top20_dist, dims=[0]
                )  # Move the closest to the head
                scores_top20 = scores[i][
                    sorted_top20_dist
                ]  # Convert to ID in training set

                labels_np = KNN_dict["Test_labels"]
                prediction = np.argmax(np.bincount(labels_np[scores_top20]))
                gt_id = KNN_dict[i]["GT_id"]

                if RunningParams.IMAGENET_REAL is True:
                    gt_id = gt_ids
                    if prediction in gt_id:
                        correct_count += 1
                        correctness = True
                    else:
                        correctness = False
                    if prediction in gt_id and KNN_dict[i]["Output"] is False:
                        wrong2crt_cnt += 1
                    elif prediction not in gt_id and KNN_dict[i]["Output"] is True:
                        crt2wrong_cnt += 1
                else:
                    if prediction == gt_id:
                        correct_count += 1
                    if prediction == gt_id and KNN_dict[i]["Output"] is False:
                        wrong2crt_cnt += 1
                    elif prediction != gt_id and KNN_dict[i]["Output"] is True:
                        crt2wrong_cnt += 1

                wandb.log({"EMD Correct Count": correct_count})

                if KNN_dict[i]["Output"] is True:
                    correct_count_KNN += 1

                NNs_id_count_dict = collections.Counter(
                    list(labels_np[scores_top20].numpy())
                )
                predicted_wnid = [
                    k
                    for (k, v) in imagenet_train_data.class_to_idx.items()
                    if v == prediction
                ][0]
                top1_count = NNs_id_count_dict[prediction.item()]

                if RunningParams.VISUALIZATION:
                    vis_cnt += 1

                    if Visualization.VISUALIZE_TOP1:
                        top1_nns = []
                        for nn_idx in sorted_top20_dist:
                            if NNs[nn_idx][1] == predicted_wnid:
                                top1_nns.append(nn_idx.item())

                        sorted_top20_dist = top1_nns

                    knn_top_predicted_wnids = [
                        train_id_to_wnid[top_id]
                        for top_id in list(KNN_dict[i]["NNs_id_count_dict"].keys())
                    ]
                    knn_predicted_wnid = KNN_dict[i]["Predicted_wnid"]
                    knn_candidates = HelperFunctions.get_candidate_wnids(
                        knn_predicted_wnid, knn_top_predicted_wnids, reaL_gt_wnids, 3
                    )

                    emd_top_predicted_wnids = [
                        train_id_to_wnid[top_id.item()]
                        for top_id in list(NNs_id_count_dict.keys())
                    ]
                    emd_predicted_wnid = predicted_wnid
                    emd_candidates = HelperFunctions.get_candidate_wnids(
                        emd_predicted_wnid, emd_top_predicted_wnids, reaL_gt_wnids, 3
                    )

                    if knn_candidates is None or emd_candidates is None:
                        no_candidate += 1
                        Task2_candidate = False

                    Visualization.visualize_correlation_maps_with_box(
                        Query,
                        NNs,
                        i,
                        q2g_att,
                        g2q_att,
                        Visualization.NN_vis_num,
                        Visualization.upsample_size,
                        val_dataset,
                        sorted_top20_dist,
                    )

                    Visualization.visualize_optimal_plans(
                        NNs, opt_plan, 1, val_dataset, sorted_top20_dist
                    )

                    Visualization.visualize_correspondence_with_boxes(
                        Query,
                        NNs,
                        i,
                        Visualization.NN_vis_num,
                        opt_plan,
                        val_dataset,
                        RunningParams.layer4_fm_size,
                        sorted_top20_dist,
                    )
                    if (
                        RunningParams.VISUALIZATION is True
                        and val_dataset == Dataset.CUB200
                    ):
                        gt_wnid = Query[0].split("/")[-2]
                        if (
                            prediction == gt_id
                            and KNN_dict[i]["Predicted_wnid"] != gt_wnid
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "EMD_correct",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["EMD_correct"] += 1
                        elif (
                            prediction != gt_id
                            and KNN_dict[i]["Predicted_wnid"] == gt_wnid
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "KNN_correct",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["KNN_correct"] += 1

                        elif (
                            prediction == gt_id
                            and KNN_dict[i]["Predicted_wnid"] == gt_wnid
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "Both_correct",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["Both_correct"] += 1

                        elif (
                            prediction != gt_id
                            and KNN_dict[i]["Predicted_wnid"] != gt_wnid
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "Both_wrong",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["Both_wrong"] += 1
                    else:
                        if (
                            predicted_wnid in reaL_gt_wnids
                            and KNN_dict[i]["Predicted_wnid"] not in reaL_gt_wnids
                            and vis_cnt_dict["EMD_correct"] < Visualization.vis_cnt
                            or (val_dataset == Dataset.CUB200)
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "EMD_correct",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["EMD_correct"] += 1
                        elif (
                            predicted_wnid not in reaL_gt_wnids
                            and KNN_dict[i]["Predicted_wnid"] in reaL_gt_wnids
                            and vis_cnt_dict["KNN_correct"] < Visualization.vis_cnt
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "KNN_correct",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["KNN_correct"] += 1

                        elif (
                            predicted_wnid in reaL_gt_wnids
                            and KNN_dict[i]["Predicted_wnid"] in reaL_gt_wnids
                            and vis_cnt_dict["Both_correct"] < Visualization.vis_cnt
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "Both_correct",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["Both_correct"] += 1

                        elif (
                            predicted_wnid not in reaL_gt_wnids
                            and KNN_dict[i]["Predicted_wnid"] not in reaL_gt_wnids
                            and vis_cnt_dict["Both_wrong"] < Visualization.vis_cnt
                        ):
                            Visualization.imagenet_real_visualize_neighbor_explanation(
                                i,
                                Query,
                                NNs,
                                top1_count,
                                predicted_wnid,
                                val_dataset,
                                "Both_wrong",
                                sorted_top20_dist,
                                KNN_dict[i],
                                reaL_gt_wnids,
                            )
                            vis_cnt_dict["Both_wrong"] += 1

                    exported_results["prediction_info"][
                        i
                    ] = {}  # e.g. i value runs from 0 -> 6785 (6786 images)
                    if val_dataset != Dataset.IMAGENET_1K_50K_CLEAN:
                        exported_results["prediction_info"][i]["EMD-Output"] = (
                            prediction == gt_id
                        )
                    else:
                        exported_results["prediction_info"][i]["EMD-Output"] = (
                            prediction in gt_id
                        )
                        exported_results["prediction_info"][i][
                            "real-gts"
                        ] = gt_ids  # e.g. a list of ints [100, 123, 131]
                        exported_results["prediction_info"][i][
                            "real-gts-winds"
                        ] = reaL_gt_wnids  # e.g. ['n02509815','n02417914', 'n02077923']
                        exported_results["prediction_info"][i]["Task_1"] = Task1
                        exported_results["prediction_info"][i]["Task_2"] = Task2

                    exported_results["prediction_info"][i]["KNN-Output"] = KNN_dict[i][
                        "Output"
                    ]
                    exported_results["prediction_info"][i]["query-path"] = Query[
                        0
                    ]  # Absolute path to the query
                    exported_results["prediction_info"][i][
                        "emd-predictions"
                    ] = predicted_wnid  # e.g. string of n02110185
                    exported_results["prediction_info"][i][
                        "emd-confidence"
                    ] = top1_count  # e.g. integer of 10
                    exported_results["prediction_info"][i][
                        "emd-candidates"
                    ] = emd_candidates  # e.g. ['n02509815','n02417914', 'n02077923']
                    exported_results["prediction_info"][i][
                        "knn-predictions"
                    ] = KNN_dict[i][
                        "Predicted_wnid"
                    ]  # e.g. string of n02110185
                    exported_results["prediction_info"][i]["knn-confidence"] = KNN_dict[
                        i
                    ]["NNs_id_count_dict"][
                        KNN_dict[i]["Predicted_id"].item()
                    ]  # e.g. integer of 10
                    exported_results["prediction_info"][i][
                        "knn-candidates"
                    ] = knn_candidates  # e.g. ['n02509815','n02417914', 'n02077923']
                    exported_results["prediction_info"][i]["NNs"] = NNs
                    exported_results["prediction_info"][i][
                        "emd-reranked"
                    ] = sorted_top20_dist

                else:
                    exported_results["prediction_info"][
                        i
                    ] = {}
                    exported_results["prediction_info"][i]["EMD-Output"] = (
                        prediction == gt_id
                    )
                    exported_results["prediction_info"][i]["query-path"] = Query[
                        0
                    ]  # Absolute path to the query
                    exported_results["prediction_info"][i][
                        "emd-predictions"
                    ] = predicted_wnid  # e.g. string of n02110185
                    exported_results["prediction_info"][i][
                        "emd-confidence"
                    ] = top1_count  # e.g. integer of 10
                    exported_results["prediction_info"][i][
                        "knn-predictions"
                    ] = KNN_dict[i][
                        "Predicted_wnid"
                    ]  # e.g. string of n02110185
                    exported_results["prediction_info"][i]["gt-id"] = gt_id
                    exported_results["prediction_info"][i][
                        "emd-predicted-id"
                    ] = prediction
                    exported_results["prediction_info"][i]["NNs"] = NNs
                    exported_results["prediction_info"][i][
                        "emd-reranked"
                    ] = sorted_top20_dist

            HelperFunctions.check_and_mkdir("pickle_files")
            with open(
                "pickle_files/{}-emd_results_{}.pickle".format(
                    val_dataset, wandb.run.name
                ),
                "wb",
            ) as f:
                pickle.dump(exported_results, f)

            acc_of_K_EMD = round((correct_count / N_test) * 100, 2)
            acc_of_K_KNN = round((correct_count_KNN / N_test) * 100, 2)

            print(
                "Patch 0 ratio: Over {} query-gallery pairs, there are {}% having highly similar patch 0 ({})".format(
                    50 * RunningParams.N_test,
                    (patch_0_num_total * 100) / (50 * RunningParams.N_test),
                    patch_0_num_total,
                )
            )

            print(
                "{} changed from wrong to correct and {} changed from correct to wrong".format(
                    wrong2crt_cnt, crt2wrong_cnt
                )
            )

            print(
                "KNN | N: {} - Dataset: {} - K: {} - Accuracy: {}%".format(
                    N_test, val_dataset, K_value, acc_of_K_KNN
                )
            )
            print(
                "EMD | N: {} - Dataset: {} - K: {} - Accuracy: {}%".format(
                    N_test, val_dataset, K_value, acc_of_K_EMD
                )
            )

            correct_count_KNN = 0
            if val_dataset == Dataset.IMAGENET_HARD:
                N_test_all = 2894
            elif val_dataset == Dataset.IMAGENET_MULTI_OBJECT:
                N_test_all = 6786
            elif val_dataset == Dataset.IMAGENET_PILOT_VIS:
                N_test_all = 6000
            elif val_dataset == Dataset.IMAGENET_1K_50K:
                N_test_all = 50000
            elif val_dataset == Dataset.IMAGENET_1K_50K_CLEAN:
                N_test_all = 46043
                N_test_all = N_test
            else:
                N_test_all = N_test

            for i in tqdm(range(N_test_all)):
                if KNN_dict[i]["Output"] is True:
                    correct_count_KNN += 1

            acc_of_K_KNN_all = round((correct_count_KNN / N_test_all) * 100, 2)
            print(
                "KNN | N: {} - Dataset: {} - K: {} - Accuracy: {}%".format(
                    N_test_all, val_dataset, K_value, acc_of_K_KNN_all
                )
            )

            if RunningParams.RANDOM_SHUFFLE:
                rd_correct_count = 0
                for i in tqdm(range(N_test)):
                    scores[i] = KNN_dict[i]["1000_NNs"][-K_value:].to("cpu")
                    Query = KNN_dict[i]["Query"]

                    torch.manual_seed(42)
                    rd_idx = torch.randperm(K_value)
                    rd_topk = scores[i][rd_idx][:20]

                    labels_np = KNN_dict["Test_labels"]

                    prediction = np.argmax(np.bincount(labels_np[rd_topk]))
                    gt_id = KNN_dict[i]["GT_id"]

                    if RunningParams.IMAGENET_REAL:
                        base_path_name = os.path.basename(Query[0])
                        gt_ids = real_labels[base_path_name]
                        gt_id = gt_ids
                        if prediction in gt_id:
                            rd_correct_count += 1
                    else:
                        if prediction == gt_id:
                            rd_correct_count += 1

                rd_acc_of_K_KNN = round((rd_correct_count / N_test) * 100, 2)
                print(
                    "Random | N: {} - Dataset: {} - K: {} - Accuracy: {}%".format(
                        N_test, val_dataset, K_value, rd_acc_of_K_KNN
                    )
                )

            print("##################################")
            wandb.log(
                {
                    "acc_of_K_EMD": acc_of_K_EMD,
                    "acc_of_K_KNN": acc_of_K_KNN,
                    "acc_of_K_KNN_all": acc_of_K_KNN_all,
                    "rd_acc_of_K_KNN": rd_acc_of_K_KNN,
                }
            )

wandb.finish()
