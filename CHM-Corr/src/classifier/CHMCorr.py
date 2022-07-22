# CHM-Corr Classifier
import os
import sys
import argparse
import json
import pickle
import random
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], "chmnet"))

from common.evaluation import Evaluator
from model import chmnet
from model.base.geometry import Geometry

from Utils import (
    CosineCustomDataset,
    KNNSupportSet,
    PairedLayer4Extractor,
    compute_spatial_similarity,
    generate_mask,
    normalize_array,
    get_transforms,
    arg_topK,
)

# Setting the random seed
random.seed(42)

# Helper Function
to_np = lambda x: x.data.to("cpu").numpy()

# ImageNet Label Names
with open("imagenet-labels.json", "rb") as f:
    folder_to_name = json.load(f)

# CHMNet Config
chm_args = dict(
    {
        "alpha": [0.05, 0.1],
        "img_size": 240,
        "ktype": "full",
        "load": os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../weights/pas_psi.pt"
        ),
    }
)

# Transform for displaying images
display_transform = transforms.Compose(
    [
        transforms.Resize(chm_args["img_size"]),
        transforms.CenterCrop((chm_args["img_size"], chm_args["img_size"])),
    ]
)


class CHMGridTransfer:
    def __init__(
        self,
        query_id,
        train_folder,
        val_folder,
        knn_scores,
        top_N,
        top_K,
        binarization_threshold,
        chm_source_transform,
        chm_target_transform,
        cosine_source_transform,
        cosine_target_transform,
        batch_size=64,
    ):
        self.knn_results = KNNSupportSet(
            train_folder=train_folder, val_folder=val_folder, knn_scores=knn_scores
        )
        self.N = top_N
        self.K = top_K
        self.BS = batch_size

        self.chm_source_transform = chm_source_transform
        self.chm_target_transform = chm_target_transform
        self.cosine_source_transform = cosine_source_transform
        self.cosine_target_transform = cosine_target_transform

        self.source_embeddings = None
        self.target_embeddings = None
        self.correspondence_map = None
        self.similarity_maps = None
        self.reverse_similarity_maps = None
        self.transferred_points = None

        self.binarization_threshold = binarization_threshold
        self.query_id = query_id

        self.q, self.ql = self.knn_results.get_image_and_label_by_id(self.query_id)
        # Folder Name as Ground Truth
        self.GT = self.knn_results.get_folder_name(self.query_id)
        # Get support set (nearest neighbor) for given query
        self.support_set = self.knn_results.get_support_set(self.query_id, top_N=top_N)
        # Get labels for support set for given query
        self.labels_ss = self.knn_results.get_support_set_labels(
            self.query_id, top_N=top_N
        )

    def build(self):
        # C.M.H
        test_ds = CosineCustomDataset(
            query_image=self.q,
            supporting_set=self.support_set,
            source_transform=self.chm_source_transform,
            target_transform=self.chm_target_transform,
        )
        test_dl = DataLoader(test_ds, batch_size=self.BS, shuffle=False)
        self.find_correspondences(test_dl)

        # LAYER 4s
        test_ds = CosineCustomDataset(
            query_image=self.q,
            supporting_set=self.support_set,
            source_transform=self.cosine_source_transform,
            target_transform=self.cosine_target_transform,
        )
        test_dl = DataLoader(test_ds, batch_size=self.BS, shuffle=False)
        self.compute_embeddings(test_dl)
        self.compute_similarity_map()

    def find_correspondences(self, test_dl):
        # Model initialization
        model = chmnet.CHMNet(chm_args["ktype"]).cuda()
        model.load_state_dict(torch.load(chm_args["load"]))
        Evaluator.initialize(chm_args["alpha"])
        Geometry.initialize(img_size=chm_args["img_size"])

        grid_results = []
        transferred_points = []

        # FIXED GRID HARD CODED
        fixed_src_grid_points = list(
            product(
                np.linspace(1 + 17, 240 - 17 - 1, 7),
                np.linspace(1 + 17, 240 - 17 - 1, 7),
            )
        )
        fixed_src_grid_points = np.asarray(fixed_src_grid_points, dtype=np.float64).T

        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(test_dl):

                keypoints = (
                    torch.tensor(fixed_src_grid_points)
                    .unsqueeze(0)
                    .repeat(batch["src_img"].shape[0], 1, 1)
                )
                n_pts = torch.tensor(
                    np.asarray(batch["src_img"].shape[0] * [49]), dtype=torch.long
                )

                corr_matrix = model(batch["src_img"].cuda(), batch["trg_img"].cuda())
                prd_kps = Geometry.transfer_kps(
                    corr_matrix, keypoints.cuda(), n_pts.cuda(), normalized=False
                )
                transferred_points.append(prd_kps.cpu().numpy())
                for tgt_points in prd_kps:
                    tgt_grid = []
                    for x, y in zip(tgt_points[0], tgt_points[1]):
                        tgt_grid.append(
                            [int(((x + 1) / 2.0) * 7), int(((y + 1) / 2.0) * 7)]
                        )
                    grid_results.append(tgt_grid)

        self.correspondence_map = grid_results
        self.transferred_points = np.vstack(transferred_points)

    def compute_embeddings(self, test_dl):
        paired_extractor = PairedLayer4Extractor()

        source_embeddings = []
        target_embeddings = []

        with torch.no_grad():
            for idx, batch in enumerate(test_dl):
                s_e, t_e = paired_extractor(
                    (batch["src_img"].cuda(), batch["trg_img"].cuda())
                )

                source_embeddings.append(s_e)
                target_embeddings.append(t_e)

        # EMBEDDINGS
        self.source_embeddings = torch.cat(source_embeddings, axis=0)
        self.target_embeddings = torch.cat(target_embeddings, axis=0)

    def compute_similarity_map(self):
        CosSim = nn.CosineSimilarity(dim=0, eps=1e-6)

        similarity_maps = []
        rsimilarity_maps = []

        grid = []
        for i in range(7):
            for j in range(7):
                grid.append([i, j])

        # Compute for all image pairs
        for i in range(len(self.correspondence_map)):
            cosine_map = np.zeros((7, 7))
            reverse_cosine_map = np.zeros((7, 7))

            # calculate cosine based on the chm corr. map
            for S, T in zip(grid, self.correspondence_map[i]):
                v1 = self.source_embeddings[i][:, S[0], S[1]]
                v2 = self.target_embeddings[i][:, T[0], T[1]]
                covalue = CosSim(v1, v2)
                cosine_map[S[0], S[1]] = covalue
                reverse_cosine_map[T[0], T[1]] = covalue

            similarity_maps.append(cosine_map)
            rsimilarity_maps.append(reverse_cosine_map)

        self.similarity_maps = similarity_maps
        self.reverse_similarity_maps = rsimilarity_maps

    def compute_score_using_cc(self):
        # CC MAPS
        SIMS_source, SIMS_target = [], []
        for i in range(len(self.source_embeddings)):
            simA, simB = compute_spatial_similarity(
                to_np(self.source_embeddings[i]), to_np(self.target_embeddings[i])
            )

            SIMS_source.append(simA)
            SIMS_target.append(simB)

        SIMS_source = np.stack(SIMS_source, axis=0)
        # SIMS_target = np.stack(SIMS_target, axis=0)

        top_cos_values = []

        for i in range(len(self.similarity_maps)):
            cosine_value = np.multiply(
                self.similarity_maps[i],
                generate_mask(
                    normalize_array(SIMS_source[i]), t=self.binarization_threshold
                ),
            )
            top_5_indicies = np.argsort(cosine_value.T.reshape(-1))[::-1][:5]
            mean_of_top_5 = np.mean(
                [cosine_value.T.reshape(-1)[x] for x in top_5_indicies]
            )
            top_cos_values.append(np.mean(mean_of_top_5))

        return top_cos_values

    def compute_score_using_custom_points(self, selected_keypoint_masks):
        top_cos_values = []

        for i in range(len(self.similarity_maps)):
            cosine_value = np.multiply(self.similarity_maps[i], selected_keypoint_masks)
            top_indicies = np.argsort(cosine_value.T.reshape(-1))[::-1]
            mean_of_tops = np.mean(
                [cosine_value.T.reshape(-1)[x] for x in top_indicies]
            )
            top_cos_values.append(np.mean(mean_of_tops))

        return top_cos_values

    #
    # def predict_using_cc(self):
    #     # COSINE
    #     top_cos_values = self.compute_score_using_cc()
    #
    #     # Predict
    #     prediction = np.argmax(
    #         np.bincount([self.l[x] for x in np.argsort(top_cos_values)[::-1][: self.K]])
    #     )
    #     prediction_weight = np.max(
    #         np.bincount([self.l[x] for x in np.argsort(top_cos_values)[::-1][: self.K]])
    #     )
    #     relevant_idx = [
    #         x for x in np.argsort(top_cos_values)[::-1] if self.l[x] == prediction
    #     ]
    #     relevant_files = [self.s[x] for x in relevant_idx]
    #     relevant_scores = [top_cos_values[x] for x in relevant_idx]
    #     predicted_folder_name = relevant_files[0].split("/")[-2]
    #     return prediction, prediction_weight, relevant_files[:5]
    #
    # def predict_custom_pairs(self, selected_keypoint_masks):
    #     # COSINE
    #     top_cos_values = self.compute_score_using_custom_points(selected_keypoint_masks)
    #
    #     # Predict
    #     prediction = np.argmax(
    #         np.bincount([self.l[x] for x in np.argsort(top_cos_values)[::-1][: self.K]])
    #     )
    #     prediction_weight = np.max(
    #         np.bincount([self.l[x] for x in np.argsort(top_cos_values)[::-1][: self.K]])
    #     )
    #     relevant_idx = [
    #         x for x in np.argsort(top_cos_values)[::-1] if self.l[x] == prediction
    #     ]
    #     relevant_files = [self.s[x] for x in relevant_idx]
    #     return prediction, prediction_weight, relevant_files[:5]

    def export(self):
        storage = {
            "knnresults": self.knn_results,
            "N": self.N,
            "K": self.K,
            "source_embeddings": self.source_embeddings,
            "target_embeddings": self.target_embeddings,
            "correspondence_map": self.correspondence_map,
            "similarity_maps": self.similarity_maps,
            "T": self.binarization_threshold,
            "query_id": self.query_id,
            "query": self.q,
            "query_label": self.ql,
            "GT": self.GT,
            "support_set": self.support_set,
            "labels_for_support_set": self.labels_ss,
            "rsimilarity_maps": self.reverse_similarity_maps,
            "transferred_points": self.transferred_points,
        }

        return ModifiableCHMResults(storage)


class ModifiableCHMResults:
    def __init__(self, storage):
        self.knn_results = storage["knnresults"]
        self.N = storage["N"]
        self.K = storage["K"]
        self.source_embeddings = storage["source_embeddings"]
        self.target_embeddings = storage["target_embeddings"]
        self.correspondence_map = storage["correspondence_map"]
        self.similarity_maps = storage["similarity_maps"]
        self.T = storage["T"]
        self.query_id = storage["query_id"]
        self.q = storage["query"]
        self.ql = storage["query_label"]
        self.GT = storage["GT"]
        self.support_set = storage["support_set"]
        self.labels_ss = storage["labels_for_support_set"]
        self.rsimilarity_maps = storage["rsimilarity_maps"]
        self.transferred_points = storage["transferred_points"]

        self.similarity_maps_masked = None
        self.SIMS_source = None
        self.SIMS_target = None
        self.masked_sim_values = []
        self.top_cos_values = []

    def compute_score_using_cc(self):
        # CC MAPS
        SIMS_source, SIMS_target = [], []
        for i in range(len(self.source_embeddings)):
            simA, simB = compute_spatial_similarity(
                to_np(self.source_embeddings[i]), to_np(self.target_embeddings[i])
            )

            SIMS_source.append(simA)
            SIMS_target.append(simB)

        SIMS_source = np.stack(SIMS_source, axis=0)
        SIMS_target = np.stack(SIMS_target, axis=0)

        self.SIMS_source = SIMS_source
        self.SIMS_target = SIMS_target

        top_cos_values = []

        for i in range(len(self.similarity_maps)):
            masked_sim_values = np.multiply(
                self.similarity_maps[i],
                generate_mask(normalize_array(SIMS_source[i]), t=self.T),
            )
            self.masked_sim_values.append(masked_sim_values)
            top_5_indicies = np.argsort(masked_sim_values.T.reshape(-1))[::-1][:5]
            mean_of_top_5 = np.mean(
                [masked_sim_values.T.reshape(-1)[x] for x in top_5_indicies]
            )
            top_cos_values.append(np.mean(mean_of_top_5))

        self.top_cos_values = top_cos_values

        return top_cos_values

    def compute_score_using_custom_points(self, selected_keypoint_masks):
        top_cos_values = []
        similarity_maps_masked = []

        for i in range(len(self.similarity_maps)):
            cosine_value = np.multiply(self.similarity_maps[i], selected_keypoint_masks)
            similarity_maps_masked.append(cosine_value)
            top_indicies = np.argsort(cosine_value.T.reshape(-1))[::-1]
            mean_of_tops = np.mean(
                [cosine_value.T.reshape(-1)[x] for x in top_indicies]
            )
            top_cos_values.append(np.mean(mean_of_tops))

        self.similarity_maps_masked = similarity_maps_masked
        return top_cos_values

    def predict_using_cc(self):
        top_cos_values = self.compute_score_using_cc()
        # Predict
        prediction = np.argmax(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: self.K]]
            )
        )
        prediction_weight = np.max(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: self.K]]
            )
        )

        reranked_nns_idx = [x for x in np.argsort(top_cos_values)[::-1]]
        reranked_nns_files = [self.support_set[x] for x in reranked_nns_idx]

        topK_idx = [
            x
            for x in np.argsort(top_cos_values)[::-1]
            if self.labels_ss[x] == prediction
        ]
        topK_files = [self.support_set[x] for x in topK_idx]
        topK_cmaps = [self.correspondence_map[x] for x in topK_idx]
        topK_similarity_maps = [self.similarity_maps[x] for x in topK_idx]
        topK_rsimilarity_maps = [self.rsimilarity_maps[x] for x in topK_idx]
        topK_transfered_points = [self.transferred_points[x] for x in topK_idx]
        # topK_scores = [top_cos_values[x] for x in topK_idx]
        predicted_folder_name = topK_files[0].split("/")[-2]

        return (
            topK_idx,
            prediction,
            predicted_folder_name,
            prediction_weight,
            topK_files[: self.K],
            reranked_nns_files[: self.K],
            topK_cmaps[: self.K],
            topK_similarity_maps[: self.K],
            topK_rsimilarity_maps[: self.K],
            topK_transfered_points[: self.K],
        )

    def predict_custom_pairs(self, selected_keypoint_masks):
        top_cos_values = self.compute_score_using_custom_points(selected_keypoint_masks)

        # Predict
        prediction = np.argmax(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: self.K]]
            )
        )
        prediction_weight = np.max(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: self.K]]
            )
        )

        reranked_nns_idx = [x for x in np.argsort(top_cos_values)[::-1]]
        reranked_nns_files = [self.support_set[x] for x in reranked_nns_idx]

        topK_idx = [
            x
            for x in np.argsort(top_cos_values)[::-1]
            if self.labels_ss[x] == prediction
        ]
        topK_files = [self.support_set[x] for x in topK_idx]
        topK_cmaps = [self.correspondence_map[x] for x in topK_idx]
        topK_similarity_maps = [self.similarity_maps[x] for x in topK_idx]
        topK_rsimilarity_maps = [self.rsimilarity_maps[x] for x in topK_idx]
        topK_transferred_points = [self.transferred_points[x] for x in topK_idx]
        # topK_scores = [top_cos_values[x] for x in topK_idx]
        topK_masked_sims = [self.similarity_maps_masked[x] for x in topK_idx]
        predicted_folder_name = topK_files[0].split("/")[-2]

        non_zero_mask = np.count_nonzero(selected_keypoint_masks)

        return (
            topK_idx,
            prediction,
            predicted_folder_name,
            prediction_weight,
            topK_files[: self.K],
            reranked_nns_files[: self.K],
            topK_cmaps[: self.K],
            topK_similarity_maps[: self.K],
            topK_rsimilarity_maps[: self.K],
            topK_transferred_points[: self.K],
            topK_masked_sims[: self.K],
            non_zero_mask,
        )


def export_visualizations_results(
    reranker_output,
    knn_results,
    mask=None,
    folder_to_label=folder_to_name,
    K=20,
    N=50,
    T=0.55,
):
    """
    Export all details for visualization and analysis
    """

    QID = reranker_output.query_id
    # GT = my_reranker.GT
    GTF = reranker_output.q.split("/")[-2]  # GT WNID/FOLDER NAME

    # KNN RESULTS
    topK_knns = knn_results.get_topK_knn(QID, k=K)
    knn_predictions, knn_acc = knn_results.get_knn_predictions(k=K)
    knn_predicted_label = folder_to_label[
        knn_results.get_foldername_for_label(knn_predictions[QID])
    ]

    # Make a prediction using CHM ranking
    if mask is not None:
        (
            topK_idx,
            p,
            pfn,
            pr,
            rfiles,
            reranked_nns,
            cmaps,
            sims,
            rsims,
            trns_kpts,
            masked_sims,
            non_zero_mask,
        ) = reranker_output.predict_custom_pairs(mask)
    else:
        non_zero_mask = 5  # default value
        (
            topK_idx,
            p,
            pfn,
            pr,
            rfiles,
            reranked_nns,
            cmaps,
            sims,
            rsims,
            trns_kpts,
        ) = reranker_output.predict_using_cc()

    if mask is not None:
        # When mask is provided, we zero out the cosine values by given mask
        MASKED_COSINE_VALUES = [np.multiply(sims[X], mask) for X in range(len(sims))]
    else:
        # When the mask is not provided, by default we are looking for top-5
        MASKED_COSINE_VALUES = [
            np.multiply(
                sims[X],
                generate_mask(
                    normalize_array(reranker_output.SIMS_source[topK_idx[X]]), t=T
                ),
            )
            for X in range(len(sims))
        ]

    list_of_source_points = []
    list_of_target_points = []

    for CK in range(len(sims)):
        target_keypoints = []
        topk_index = arg_topK(MASKED_COSINE_VALUES[CK], topK=non_zero_mask)

        for i in range(non_zero_mask):  # Number of Connections
            # Psource = point_list[topk_index[i]]
            x, y = trns_kpts[CK].T[topk_index[i]]
            Ptarget = int(((x + 1) / 2.0) * 240), int(((y + 1) / 2.0) * 240)
            target_keypoints.append(Ptarget)

        # Uniform Grid of points
        a = np.linspace(1 + 17, 240 - 17 - 1, 7)
        b = np.linspace(1 + 17, 240 - 17 - 1, 7)
        point_list = list(product(a, b))

        list_of_source_points.append(np.asarray([point_list[x] for x in topk_index]))
        list_of_target_points.append(np.asarray(target_keypoints))

    # EXPORT OUTPUT
    detailed_output = {
        "id": QID,
        "q": reranker_output.q,
        "gt_name": folder_to_label[GTF],
        "gt_wnid": GTF,
        "K": K,
        "N": N,
        "knn-prediction": knn_predicted_label,
        "knn-prediction-confidence": knn_results.get_knn_confidence(QID),
        "knn-nearest-neighbors": topK_knns,
        "knn-nearest-neighbors-all": knn_results.get_support_set(QID),
        "chm-prediction": folder_to_label[pfn],
        "chm-prediction-confidence": pr,
        "chm-nearest-neighbors": rfiles,
        "chm-nearest-neighbors-all": reranked_nns,
        "correspondance_map": cmaps,
        "masked_cos_values": MASKED_COSINE_VALUES,
        "src-keypoints": list_of_source_points,
        "tgt-keypoints": list_of_target_points,
        "non_zero_mask": non_zero_mask,
        "transferred_kpoints": trns_kpts,
    }

    return detailed_output


def rerank_and_save(
    QID,
    TRAIN_SET,
    VAL_SET,
    knn_results,
    knn_support_set,
    N,
    K,
    T,
    BS,
    chm_src_t,
    chm_tgt_t,
    cos_src_t,
    cos_tgt_t,
    output_dir,
    mask=None,
):
    """
    Creates a reranker, save the results inside a pickle file and returns nothing
    Args:
    QID: query id
    K: number of nearest neighbors
    N: number of nearest neighbors to be reranked
    BS: batch size
    T: Threshold
    chm_src_t: CHM Source Transform
    chm_tgt_t: CHM Target Transform
    cos_src_t: Cosine Source Transform
    cos_tgt_t: Cosine Target Transform
    output_dir: output directory
    """
    reranker = CHMGridTransfer(
        query_id=QID,
        train_folder=TRAIN_SET,
        val_folder=VAL_SET,
        knn_scores=knn_results,
        top_N=N,
        top_K=K,
        binarization_threshold=T,
        chm_source_transform=chm_src_t,
        chm_target_transform=chm_tgt_t,
        cosine_source_transform=cos_src_t,
        cosine_target_transform=cos_tgt_t,
        batch_size=BS,
    )

    # Building the reranker
    reranker.build()
    # Make a ModifiableCHMResults
    exported_reranker = reranker.export()
    # Export A details for visualizations
    output = export_visualizations_results(
        exported_reranker, knn_support_set, mask=mask, K=K, N=N, T=T
    )

    with open(f"{output_dir}/reranker_{QID}.pkl", "wb") as f:
        pickle.dump(output, f)


def report_accuracy(files):
    chm_counter = 0
    knn_counter = 0

    for r_file in tqdm(files):
        with open(r_file, "rb") as f:
            r = pickle.load(f)
        chm_wnid = r["chm-nearest-neighbors"][0].split("/")[-2]
        knn_wnid = r["knn-nearest-neighbors"][0].split("/")[-2]

        if chm_wnid == r["gt_wnid"]:
            chm_counter += 1

        if knn_wnid == r["gt_wnid"]:
            knn_counter += 1

    print(
        "CHM-Corr: ",
        chm_counter,
        "Corrects",
        "Accuracy: ",
        100 * chm_counter / len(files),
    )
    print(
        "kNN: ", knn_counter, "Corrects", "Accuracy: ", 100 * knn_counter / len(files)
    )


def main():
    parser = argparse.ArgumentParser(description="CHM Classifier")
    parser.add_argument(
        "--transform", help="Type of Transform", type=str, default="single"
    )
    parser.add_argument("--knn", help="Path to kNN Scores", type=str, required=True)
    parser.add_argument("--train", help="Path to Train folder", type=str, required=True)
    parser.add_argument(
        "--val", help="Path to Validation folder", type=str, required=True
    )
    parser.add_argument(
        "--out", help="Path to save output file", type=str, required=True
    )
    parser.add_argument("--N", help="Value for N", type=int, default=200)
    parser.add_argument("--K", help="Value for K", type=int, default=20)
    parser.add_argument("--T", help="Value for threshold", type=float, default=0.55)
    parser.add_argument("--bs", help="Value for batch size", type=int, default=128)

    args = parser.parse_args()

    # Open kNN results
    with open(args.knn, "rb") as f:
        knn_results = pickle.load(f)

    knn_support_set = KNNSupportSet(args.train, args.val, knn_results)

    # Get size of the validation set
    validation_set_size = len(ImageFolder(args.val))
    print(f"Images in the validation set: {validation_set_size}")

    # Get Transforms
    chm_src_t, chm_tgt_t, cos_src_t, cos_tgt_t = get_transforms(
        args.transform, chm_args
    )

    outputs = []

    for i in tqdm(range(validation_set_size)):
        if os.path.exists(f"{args.out}/reranker_{i}.pkl"):
            continue
        rerank_and_save(
            i,
            args.train,
            args.val,
            knn_results,
            knn_support_set,
            args.N,
            args.K,
            args.T,
            args.bs,
            chm_src_t,
            chm_tgt_t,
            cos_src_t,
            cos_tgt_t,
            args.out,
            mask=None,
        )
        outputs.append(f"{args.out}/reranker_{i}.pkl")

    print("Calculating final accuracy")
    report_accuracy(outputs)


if __name__ == "__main__":
    main()
