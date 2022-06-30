import os.path

import cv2
import cv2 as cv
import image_slicer
import matplotlib
import matplotlib.patches as patches
import torch.nn.functional as F
from datasets import *
from helper import *
from image_slicer import join
from IPython.display import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from params import *

RunningParams = RunningParams()
Dataset = Dataset()
HelperFunctions = HelperFunctions()


class Visualization(object):
    def __init__(self):
        self.cMap = cm.get_cmap("turbo", 96)

        # The images to AIs should be the same with the ones to humans
        self.display_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        color_map = matplotlib.cm.get_cmap("gist_rainbow")
        self.colors = []
        for k in range(5):
            self.colors.append(color_map(k / 5.0))

        self.NN_vis_num = 5
        self.upsample_size = 224
        self.num_line = 5
        self.vis_cnt = 50000  # Limit the numbers of plots for each category (e.g. Both_correct category)

        # Visualize the NNs having the predicted label only (may not the nearest based on distance)
        self.VISUALIZE_TOP1 = True

    def imagenet_real_visualize_neighbor_explanation(
        self,
        idx,
        Query,
        NNs,
        top1_count,
        predicted_wnid,
        val_dataset,
        subfol,
        sorted_top20_dist,
        KNN_dict_entry,
        reaL_gt_wnids,
    ):
        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/{}/".format(val_dataset))
        HelperFunctions.check_and_mkdir("tmp/{}/{}/".format(val_dataset, subfol))

        save_dir = "tmp/{}".format(val_dataset)
        vis_k = self.NN_vis_num

        input_wnid = Query[1]

        if val_dataset == Dataset.CUB200:
            input_label = Query[0].split("/")[-2]
        else:
            input_label = HelperFunctions.id_map.get(Query[1]).split(",")[0]
            input_label = input_label[0].lower() + input_label[1:]

        input_labels = []
        for reaL_gt_wnid in reaL_gt_wnids:
            input_labels.append(
                HelperFunctions.id_map.get(reaL_gt_wnid).split(",")[0][0].lower()
                + HelperFunctions.id_map.get(reaL_gt_wnid).split(",")[0][1:]
            )

        if val_dataset == Dataset.CUB200:
            input_labels = [input_label]

        # -------------------------------------------------------
        # Visualize EMD
        if val_dataset == Dataset.CUB200:
            predicted_label = predicted_wnid
        else:
            predicted_label = HelperFunctions.id_map.get(predicted_wnid).split(",")[0]
            predicted_label = predicted_label[0].lower() + predicted_label[1:]

        EMD_predicted_label = predicted_label
        EMD_wnid = predicted_wnid

        if val_dataset == Dataset.CUB200:
            if input_label == predicted_label:
                color = "green"
            else:
                color = "red"
        else:
            if predicted_wnid in reaL_gt_wnids:
                color = "green"
            else:
                color = "red"

        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            Query[0], save_dir
        )
        os.system(cmd)
        cmd = "convert {}/query.jpeg -resize 400x400\! {}/query.jpeg".format(
            save_dir, save_dir
        )
        os.system(cmd)

        EMD_NNs = [NNs[top_nn_idx] for top_nn_idx in sorted_top20_dist]
        NN_merge_cmd = "montage "
        for index, NN in enumerate(EMD_NNs):
            if index == vis_k:
                break
            else:
                if val_dataset == Dataset.CUB200:
                    label = NN[1]
                else:
                    label = HelperFunctions.id_map.get(NN[1]).split(",")[0]
                    label = label[0].lower() + label[1:]

            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                NN[0], save_dir, index
            )
            os.system(cmd)
            cmd = "convert {}/{}.jpeg -resize 400x400\! {}/{}.jpeg".format(
                save_dir, index, save_dir, index
            )
            os.system(cmd)

            NN_merge_cmd += "{}/{}.jpeg ".format(save_dir, index)

        NN_merge_cmd += "-tile {}x1 -geometry +0+0 {}/EMD.jpeg".format(vis_k, save_dir)
        os.system(NN_merge_cmd)

        # Remove the tmp NNs plots
        myCmd = "rm -rf /home/giang/Downloads/KNN-ImageNet/tmp/ImageNet-val-50K-clean/[0-4].jpeg"
        os.system(myCmd)

        save_path_emd = "{}/{}/EMD_{}_{}x{}.jpeg".format(
            save_dir,
            subfol,
            idx,
            RunningParams.layer4_fm_size,
            RunningParams.layer4_fm_size,
        )
        myCmd = "montage {}/query.jpeg {}/EMD.jpeg -tile 2x -geometry +60+0 {}".format(
            save_dir, save_dir, save_path_emd
        )
        os.system(myCmd)

        annotation = "{} - GT Label: {} -- [EMD: {} - Confidence {}/{}]".format(
            val_dataset, "|".join(input_labels), predicted_label, top1_count, 20
        )
        EMD_conf = top1_count
        myCmd = (
            "convert {} -fill {} -pointsize 32 -gravity North -background White "
            '-splice 0x40 -annotate +0+4 "{}" {}'.format(
                save_path_emd, color, annotation, save_path_emd
            )
        )
        os.system(myCmd)

        # -------------------------------------------------------
        # Visualize Nearest Neighbors
        if val_dataset == Dataset.CUB200:
            predicted_label = KNN_dict_entry["Predicted_wnid"]
        else:
            predicted_label = HelperFunctions.id_map.get(
                KNN_dict_entry["Predicted_wnid"]
            ).split(",")[0]
            predicted_label = predicted_label[0].lower() + predicted_label[1:]

        KNN_predicted_label = predicted_label
        KNN_wnid = KNN_dict_entry["Predicted_wnid"]

        if val_dataset == Dataset.CUB200:
            if input_label == predicted_label:
                color = "green"
            else:
                color = "red"
        else:
            if predicted_wnid in reaL_gt_wnids:
                color = "green"
            else:
                color = "red"

        predicted_id = KNN_dict_entry["Predicted_id"]

        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            Query[0], save_dir
        )
        os.system(cmd)
        cmd = "convert {}/query.jpeg -resize 400x400\! {}/query.jpeg".format(
            save_dir, save_dir
        )
        os.system(cmd)

        # -------------------------------------------------------
        KNNs = []
        topk_NNs = NNs[-RunningParams.MajorityVotes_K :]
        topk_NNs = list(reversed(topk_NNs))  # Move the closest to the head

        for NN in topk_NNs:
            if (
                KNN_dict_entry["Predicted_wnid"] == NN[1]
            ):  # A NN have the same category with predicted category
                KNNs.append(NN)

        NN_merge_cmd = "montage "
        for index, NN in enumerate(KNNs):
            if index == vis_k:
                break
            else:
                if val_dataset == Dataset.CUB200:
                    label = NN[1]
                else:
                    label = HelperFunctions.id_map.get(NN[1]).split(",")[0]
                    label = label[0].lower() + label[1:]

            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                NN[0], save_dir, index
            )
            os.system(cmd)
            cmd = "convert {}/{}.jpeg -resize 400x400\! {}/{}.jpeg".format(
                save_dir, index, save_dir, index
            )
            os.system(cmd)

            NN_merge_cmd += "{}/{}.jpeg ".format(save_dir, index)

        NN_merge_cmd += "-tile {}x1 -geometry +0+0 {}/KNN.jpeg".format(vis_k, save_dir)
        os.system(NN_merge_cmd)

        # Remove the tmp NNs plots
        myCmd = "rm -rf /home/giang/Downloads/KNN-ImageNet/tmp/ImageNet-val-50K-clean/[0-4].jpeg"
        os.system(myCmd)

        save_path_knn = "{}/{}/KNN_{}_{}x{}.jpeg".format(
            save_dir,
            subfol,
            idx,
            RunningParams.layer4_fm_size,
            RunningParams.layer4_fm_size,
        )
        myCmd = "montage {}/query.jpeg {}/KNN.jpeg -tile 2x -geometry +60+0 {}".format(
            save_dir, save_dir, save_path_knn
        )
        os.system(myCmd)

        predicted_id = predicted_id.item()
        annotation = "{} - GT Label: {} -- [KNN: {} - Confidence {}/{}]".format(
            val_dataset,
            "|".join(input_labels),
            predicted_label,
            KNN_dict_entry["NNs_id_count_dict"][predicted_id],
            20,
        )
        KNN_conf = KNN_dict_entry["NNs_id_count_dict"][predicted_id]

        myCmd = (
            "convert {} -fill {} -pointsize 32 -gravity North -background White "
            '-splice 0x40 -annotate +0+4 "{}" {}'.format(
                save_path_knn, color, annotation, save_path_knn
            )
        )
        os.system(myCmd)

        save_path_lines = (
            "tmp/{}/Line_Correspondence/line_corres_agg{}-{}x{}.jpeg".format(
                val_dataset,
                idx,
                RunningParams.layer4_fm_size,
                RunningParams.layer4_fm_size,
            )
        )

        save_path_boxes = (
            "tmp/{}/Box_Correspondence/box_corres_agg{}-{}x{}.jpeg".format(
                val_dataset,
                idx,
                RunningParams.layer4_fm_size,
                RunningParams.layer4_fm_size,
            )
        )

        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/Final_vis")
        HelperFunctions.check_and_mkdir("tmp/Final_vis/{}".format(val_dataset))
        HelperFunctions.check_and_mkdir(
            "tmp/Final_vis/{}/{}".format(val_dataset, subfol)
        )

        base_name = os.path.basename(Query[0])
        # save_path = 'tmp/Final_vis/{}/{}/{}'.format(val_dataset, subfol, base_name)
        save_path = "tmp/Final_vis/{}/{}/Agg_{}-{}-KNN{}{}-EMD{}{}.jpeg".format(
            val_dataset,
            subfol,
            idx,
            "".join(reaL_gt_wnids),
            KNN_conf,
            KNN_wnid,
            EMD_conf,
            EMD_wnid,
        )

        save_path_cc = "tmp/{}/path_CC/corres_box_{}_{}x{}.jpeg".format(
            val_dataset, idx, RunningParams.layer4_fm_size, RunningParams.layer4_fm_size
        )

        final_save_path = "tmp/Final_vis/{}/{}/{}".format(
            val_dataset, subfol, base_name
        )
        myCmd = "convert {} {} {} -append '{}'".format(
            save_path_knn, save_path_emd, save_path_boxes, final_save_path
        )
        os.system(myCmd)

    # Visualize N nearest neighbors with the input query and labels
    def visualize_neighbor_explanation(
        self,
        idx,
        Query,
        NNs,
        top1_count,
        predicted_wnid,
        val_dataset,
        subfol,
        sorted_top20_dist,
        KNN_dict_entry,
    ):
        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/{}/".format(val_dataset))
        HelperFunctions.check_and_mkdir("tmp/{}/{}/".format(val_dataset, subfol))

        save_dir = "tmp/{}".format(val_dataset)
        vis_k = self.NN_vis_num

        input_wnid = Query[1]

        if val_dataset == Dataset.CUB200:
            input_label = Query[1]
            predicted_label = predicted_wnid
        else:

            input_label = HelperFunctions.id_map.get(Query[1]).split(",")[0]
            input_label = input_label[0].lower() + input_label[1:]

            # -------------------------------------------------------
            # Visualize EMD
            predicted_label = HelperFunctions.id_map.get(predicted_wnid).split(",")[0]
            predicted_label = predicted_label[0].lower() + predicted_label[1:]

        EMD_predicted_label = predicted_label
        EMD_wnid = predicted_wnid

        if Query[1] == predicted_wnid:
            color = "green"
        else:
            color = "red"

        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            Query[0], save_dir
        )
        os.system(cmd)
        cmd = "convert {}/query.jpeg -resize 400x400\! {}/query.jpeg".format(
            save_dir, save_dir
        )
        os.system(cmd)

        EMD_NNs = [NNs[top_nn_idx] for top_nn_idx in sorted_top20_dist]
        NN_merge_cmd = "montage "
        for index, NN in enumerate(EMD_NNs):
            if index == vis_k:
                break
            if val_dataset == Dataset.CUB200:
                label = NN[1]
            else:
                label = HelperFunctions.id_map.get(NN[1]).split(",")[0]
                label = label[0].lower() + label[1:]

            if NN[1] == Query[1]:
                color = "green"
            else:
                color = "red"

            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                NN[0], save_dir, index
            )
            os.system(cmd)
            cmd = "convert {}/{}.jpeg -resize 400x400\! {}/{}.jpeg".format(
                save_dir, index, save_dir, index
            )
            os.system(cmd)

            NN_merge_cmd += "{}/{}.jpeg ".format(save_dir, index)

        NN_merge_cmd += "-tile {}x1 -geometry +0+0 {}/EMD.jpeg".format(vis_k, save_dir)
        os.system(NN_merge_cmd)

        save_path_emd = "{}/{}/EMD_{}_{}x{}.jpeg".format(
            save_dir,
            subfol,
            idx,
            RunningParams.layer4_fm_size,
            RunningParams.layer4_fm_size,
        )
        myCmd = "montage {}/query.jpeg {}/EMD.jpeg -tile 2x -geometry +60+0 {}".format(
            save_dir, save_dir, save_path_emd
        )
        os.system(myCmd)

        annotation = "{} - GT Label: {} -- [EMD: {} - Confidence {}/{}]".format(
            val_dataset, input_label, predicted_label, top1_count, 20
        )
        EMD_conf = top1_count
        myCmd = (
            "convert {} -fill {} -pointsize 32 -gravity North -background White "
            '-splice 0x40 -annotate +0+4 "{}" {}'.format(
                save_path_emd, color, annotation, save_path_emd
            )
        )
        os.system(myCmd)

        # -------------------------------------------------------
        # Visualize Nearest Neighbors
        if val_dataset == Dataset.CUB200:
            predicted_label = KNN_dict_entry["Predicted_wnid"]
        else:
            predicted_label = HelperFunctions.id_map.get(
                KNN_dict_entry["Predicted_wnid"]
            ).split(",")[0]
            predicted_label = predicted_label[0].lower() + predicted_label[1:]

        KNN_predicted_label = predicted_label
        KNN_wnid = KNN_dict_entry["Predicted_wnid"]

        if Query[1] == KNN_dict_entry["Predicted_wnid"]:
            color = "green"
        else:
            color = "red"

        predicted_id = KNN_dict_entry["Predicted_id"]

        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
            Query[0], save_dir
        )
        os.system(cmd)
        cmd = "convert {}/query.jpeg -resize 400x400\! {}/query.jpeg".format(
            save_dir, save_dir
        )
        os.system(cmd)

        # -------------------------------------------------------
        KNNs = []
        for NN in NNs:
            if (
                KNN_dict_entry["Predicted_wnid"] == NN[1]
            ):  # A NN have the same category with predicted category
                KNNs.append(NN)

        KNNs = KNNs[-vis_k:]
        KNNs = list(reversed(KNNs))  # Move the closest to the head

        NN_merge_cmd = "montage "
        for index, NN in enumerate(KNNs):
            if val_dataset == Dataset.CUB200:
                label = NN[1]
            else:
                label = HelperFunctions.id_map.get(NN[1]).split(",")[0]
                label = label[0].lower() + label[1:]

            if NN[1] == Query[1]:
                color = "green"
            else:
                color = "red"

            cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
                NN[0], save_dir, index
            )
            os.system(cmd)
            cmd = "convert {}/{}.jpeg -resize 400x400\! {}/{}.jpeg".format(
                save_dir, index, save_dir, index
            )
            os.system(cmd)

            NN_merge_cmd += "{}/{}.jpeg ".format(save_dir, index)

        NN_merge_cmd += "-tile {}x1 -geometry +0+0 {}/KNN.jpeg".format(vis_k, save_dir)
        os.system(NN_merge_cmd)

        save_path_knn = "{}/{}/KNN_{}_{}x{}.jpeg".format(
            save_dir,
            subfol,
            idx,
            RunningParams.layer4_fm_size,
            RunningParams.layer4_fm_size,
        )
        myCmd = "montage {}/query.jpeg {}/KNN.jpeg -tile 2x -geometry +60+0 {}".format(
            save_dir, save_dir, save_path_knn
        )
        os.system(myCmd)

        predicted_id = predicted_id.item()
        annotation = "{} - GT Label: {} -- [KNN: {} - Confidence {}/{}]".format(
            val_dataset,
            input_label,
            predicted_label,
            KNN_dict_entry["NNs_id_count_dict"][predicted_id],
            20,
        )
        KNN_conf = KNN_dict_entry["NNs_id_count_dict"][predicted_id]

        myCmd = (
            "convert {} -fill {} -pointsize 32 -gravity North -background White "
            '-splice 0x40 -annotate +0+4 "{}" {}'.format(
                save_path_knn, color, annotation, save_path_knn
            )
        )
        os.system(myCmd)

        save_path_lines = (
            "tmp/{}/Line_Correspondence/line_corres_agg{}-{}x{}.jpeg".format(
                val_dataset,
                idx,
                RunningParams.layer4_fm_size,
                RunningParams.layer4_fm_size,
            )
        )

        save_path_boxes = (
            "tmp/{}/Box_Correspondence/box_corres_agg{}-{}x{}.jpeg".format(
                val_dataset,
                idx,
                RunningParams.layer4_fm_size,
                RunningParams.layer4_fm_size,
            )
        )

        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/Final_vis")
        HelperFunctions.check_and_mkdir("tmp/Final_vis/{}".format(val_dataset))
        HelperFunctions.check_and_mkdir(
            "tmp/Final_vis/{}/{}".format(val_dataset, subfol)
        )

        save_path = "tmp/Final_vis/{}/{}/Agg_{}-{}-KNN{}{}-EMD{}{}.jpeg".format(
            val_dataset,
            subfol,
            idx,
            input_wnid,
            KNN_conf,
            KNN_wnid,
            EMD_conf,
            EMD_wnid,
        )


    def visualize_correspondence_with_boxes(
        self, Query, NNs, idx, vis_k, opt_plans, val_dataset, fm_size, sorted_top20
    ):
        subfol = "Box_Correspondence"
        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/{}/".format(val_dataset))
        HelperFunctions.check_and_mkdir("tmp/{}/{}/".format(val_dataset, subfol))
        HelperFunctions.check_and_mkdir(
            "tmp/{}/{}/Connection/".format(val_dataset, subfol)
        )
        HelperFunctions.check_and_mkdir(
            "tmp/{}/{}/Correct/".format(val_dataset, subfol)
        )
        HelperFunctions.check_and_mkdir("tmp/{}/{}/Wrong/".format(val_dataset, subfol))

        # Remove the query itself
        opt_plans = opt_plans[1:]
        topk_indices = sorted_top20[:vis_k]

        for pair_vis_idx, topk_indx in enumerate(topk_indices):
            NN = NNs[topk_indx]
            opt_plan = opt_plans[topk_indx]

            recons_grids = []
            flow_vals = []

            for pos in range(fm_size * fm_size):
                q2g_plan_at_pos = opt_plan[:, pos]
                plan_grid = q2g_plan_at_pos.view(fm_size, fm_size)
                max_pos = (plan_grid == torch.max(plan_grid)).nonzero()
                recons_grids.append(max_pos)
                flow_vals.append(torch.max(plan_grid))

            flow_vals = [flow_val.item() for flow_val in flow_vals]
            argsort_flow_vals = np.argsort(np.array(flow_vals))
            numb_line = self.num_line

            flow_vals_topk_ids = argsort_flow_vals[-numb_line:]
            flow_vals_topk = [flow_vals[flow_idx] for flow_idx in flow_vals_topk_ids]

            # Patches on gallery
            gallery_topk_ids = [
                [
                    recons_grids[flow_idx].cpu()[0].tolist()[1],
                    recons_grids[flow_idx].cpu()[0].tolist()[0],
                ]
                for flow_idx in flow_vals_topk_ids
            ]
            # Patches on query
            query_topk_ids = [
                [flow_vals_topk_id % fm_size, int(flow_vals_topk_id / fm_size)]
                for flow_vals_topk_id in flow_vals_topk_ids
            ]

            query = Query[0]
            gallery = NN[0]
            query_wnid = Query[1]
            gallery_wnid = NN[1]

            if val_dataset == Dataset.CUB200:
                query_class_name = Query[1]
                gallery_class_name = NN[1]
            else:
                query_class_name = HelperFunctions.id_map.get(query_wnid).split(",")[0]
                query_class_name = query_class_name[0].lower() + query_class_name[1:]

                gallery_class_name = HelperFunctions.id_map.get(gallery_wnid).split(
                    ","
                )[0]
                gallery_class_name = (
                    gallery_class_name[0].lower() + gallery_class_name[1:]
                )

            if query_class_name == gallery_class_name:
                color = "green"
            else:
                color = "red"

            converted_ratio = int(224 / fm_size)
            gallery_topk_pixels = [
                [
                    gallery_topk_id[0] * converted_ratio + int(converted_ratio / 2),
                    gallery_topk_id[1] * converted_ratio + int(converted_ratio / 2),
                ]
                for gallery_topk_id in gallery_topk_ids
            ]
            query_topk_pixels = [
                [
                    query_topk_id[0] * converted_ratio + int(converted_ratio / 2),
                    query_topk_id[1] * converted_ratio + int(converted_ratio / 2),
                ]
                for query_topk_id in query_topk_ids
            ]

            # PLOT
            fig, ax = plt.subplots(
                nrows=2, ncols=1, figsize=(7, 4), gridspec_kw={"wspace": 0, "hspace": 0}
            )

            source_image = Image.open(query).convert("RGB")
            target_image = Image.open(gallery).convert("RGB")

            ax[0].imshow(self.display_transform(source_image))
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].axis("off")

            ax[1].imshow(self.display_transform(target_image))
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].axis("off")

            # When target image (gallery) has multiple overlapped boxes
            # We change the color maps to synchronize the box colors b/w query vs. gallery
            # colors and gallery_topk_pixels have the same size
            # This algorithm uses a hashmap with Time complexity is O(N), Space complexity O(N)
            import copy

            colors = copy.deepcopy(self.colors)

            gallery_coordinates_dict = {}

            for gallery_idx, gallery_coordinates in enumerate(gallery_topk_pixels):
                # Change to tuple for hashing
                gallery_coordinates = tuple(gallery_coordinates)
                if gallery_coordinates not in gallery_coordinates_dict:
                    gallery_coordinates_dict[gallery_coordinates] = colors[gallery_idx]
                else:
                    colors[gallery_idx] = gallery_coordinates_dict[gallery_coordinates]

            for i in range(numb_line):  # Number of Connections
                rect = patches.Rectangle(
                    (query_topk_pixels[i][0] - 16, query_topk_pixels[i][1] - 16),
                    30,
                    30,
                    linewidth=1,
                    edgecolor=colors[i],
                    facecolor="none",
                    alpha=1,
                )
                ax[0].add_patch(rect)
                rect = patches.Rectangle(
                    (gallery_topk_pixels[i][0] - 16, gallery_topk_pixels[i][1] - 16),
                    30,
                    30,
                    linewidth=1,
                    edgecolor=colors[i],
                    facecolor="none",
                    alpha=1,
                )
                ax[1].add_patch(rect)

            save_path = "tmp/{}/{}/Connection/box_corres_{}_{}-{}x{}.jpeg".format(
                val_dataset, subfol, idx, pair_vis_idx + 1, fm_size, fm_size
            )
            fig.savefig(save_path, dpi=300, pad_inches=0, bbox_inches="tight")

            plt.show()
            plt.close()

            if Query[1] == NN[1]:
                copyfile(
                    save_path,
                    "tmp/{}/{}/Correct/box_corres_{}_{}-{}x{}.jpeg".format(
                        val_dataset, subfol, idx, pair_vis_idx + 1, fm_size, fm_size
                    ),
                )
            else:
                copyfile(
                    save_path,
                    "tmp/{}/{}/Wrong/box_corres_{}_{}-{}x{}.jpeg".format(
                        val_dataset, subfol, idx, pair_vis_idx + 1, fm_size, fm_size
                    ),
                )

        vis_path = "tmp/{}/Box_Correspondence/box_corres_agg{}-{}x{}.jpeg".format(
            val_dataset, idx, fm_size, fm_size
        )
        myCmd = (
            "montage tmp/{}/Box_Correspondence/Connection/box_corres_{}_[1-{}]-{}x{}.jpeg "
            "-tile {}x1 -geometry +4+0 {}".format(
                val_dataset, idx, vis_k, fm_size, fm_size, vis_k, vis_path
            )
        )
        os.system(myCmd)

        cmd = "convert {} -resize 2032x800\! {}".format(vis_path, vis_path)
        os.system(cmd)

        cmd = "convert tmp/original.jpeg -resize 400x400\! tmp/original.jpeg"
        os.system(cmd)

        cmd = "montage tmp/original.jpeg {} -tile 2x1 -geometry +60+0 {}".format(
            vis_path, vis_path
        )
        os.system(cmd)

    def visualize_correlation_maps_with_box(
        self,
        Query,
        NNs,
        idx,
        q2g_att,
        g2q_att,
        vis_k,
        upsample_size,
        val_dataset,
        sorted_top20,
    ):

        if not RunningParams.UNIFORM:
            subfol = "path_CC"
        else:
            subfol = "path_Uniform"
        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/{}/".format(val_dataset))
        HelperFunctions.check_and_mkdir("tmp/{}/{}/".format(val_dataset, subfol))
        HelperFunctions.check_and_mkdir(
            "tmp/{}/{}/Cross_Correlation/".format(val_dataset, subfol)
        )
        HelperFunctions.check_and_mkdir(
            "tmp/{}/{}/Correct/".format(val_dataset, subfol)
        )
        HelperFunctions.check_and_mkdir("tmp/{}/{}/Wrong/".format(val_dataset, subfol))

        topk_indices = sorted_top20[:vis_k]
        Galleries = [NNs[topk_idx] for topk_idx in topk_indices]

        # Remove the query itself
        q2g_att = q2g_att[1:]
        # TODO: check upsample_ratio must be int , please remove conversion
        N, C, _ = q2g_att.size()
        upsample_ratio = int(upsample_size / C)
        # Get the max-intensity coordinates
        q2g_att_max_indices = [
            (q2g_att[nn_idx] == torch.max(q2g_att[nn_idx])).nonzero()
            for nn_idx in topk_indices
        ]
        # Get att maps vs. 3 nearest neighbors of EMD
        q2g_att = q2g_att[:, None, :, :][topk_indices]
        # Upsample the heatmap
        q2g_att = torch.nn.functional.interpolate(
            q2g_att,
            size=(upsample_size, upsample_size),
            mode="bilinear",
            align_corners=False,
        )

        # Remove query itself
        g2q_att = g2q_att[1:]
        g2q_att_max_indices = [
            (g2q_att[nn_idx] == torch.max(g2q_att[nn_idx])).nonzero()
            for nn_idx in topk_indices
        ]
        g2q_att = g2q_att[:, None, :, :][topk_indices]
        g2q_att = torch.nn.functional.interpolate(
            g2q_att,
            size=(upsample_size, upsample_size),
            mode="bilinear",
            align_corners=False,
        )

        size = upsample_size

        for i in range(vis_k):
            topk_idx = i
            if i >= len(Galleries):
                continue
            image_path = Galleries[topk_idx][0]
            wnid = Galleries[topk_idx][1]

            if val_dataset == Dataset.CUB200:
                label = image_path.split("/")[-2]
            else:
                label = HelperFunctions.id_map.get(wnid).split(",")[0]
                label = label[0].lower() + label[1:]

            # Original image
            input_image = Image.open(image_path)
            input_image = self.display_transform(input_image)
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            plt.imshow(input_image)
            plt.savefig("tmp/original.jpeg", pad_inches=0, bbox_inches="tight")
            plt.close()

            q2g_att_arr = q2g_att[i][0].to("cpu")
            # Normalize the att map
            q2g_att_map = (q2g_att_arr - q2g_att_arr.min()) / (
                q2g_att_arr.max() - q2g_att_arr.min()
            )
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            plt.imshow(q2g_att_map, cmap=self.cMap, vmin=0, vmax=1)

            plt.savefig("tmp/heatmap.jpeg", pad_inches=0, bbox_inches="tight")
            plt.close()

            # Get overlay version
            myCmd = "composite -blend 40 tmp/original.jpeg -gravity SouthWest tmp/heatmap.jpeg tmp/{}/q2g_{}.jpeg".format(
                val_dataset, i
            )
            os.system(myCmd)

            image_path = Query[0]
            gt_wnid = Query[1]

            if val_dataset == Dataset.CUB200:
                label = image_path.split("/")[-2]
            else:
                label = HelperFunctions.id_map.get(gt_wnid).split(",")[0]
                label = label[0].lower() + label[1:]

            # Original image
            input_image = Image.open(image_path)
            input_image = self.display_transform(input_image)
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            plt.imshow(input_image)
            plt.savefig("tmp/original.jpeg", pad_inches=0, bbox_inches="tight")
            plt.close()

            g2q_att_arr = g2q_att[i][0].to("cpu")
            min_value = g2q_att_arr.min()
            max_value = g2q_att_arr.max()
            # Normalize the att map
            g2q_att_map = (g2q_att_arr - min_value) / (max_value - min_value)
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            # plt.title('Query: {}'.format(label))
            plt.imshow(g2q_att_map, cmap=self.cMap, vmin=0, vmax=1)

            plt.savefig("tmp/heatmap.jpeg", pad_inches=0, bbox_inches="tight")
            plt.close()

            # Get overlay version
            myCmd = "composite -blend 40 tmp/original.jpeg -gravity SouthWest tmp/heatmap.jpeg tmp/{}/g2q_{}.jpeg".format(
                val_dataset, i
            )
            os.system(myCmd)

            save_path = (
                "tmp/{}/{}/Cross_Correlation/corres_box_{}_{}-{}x{}.jpeg".format(
                    val_dataset,
                    subfol,
                    idx,
                    i + 1,
                    RunningParams.layer4_fm_size,
                    RunningParams.layer4_fm_size,
                )
            )
            myCmd = "montage tmp/{}/g2q_{}.jpeg tmp/{}/q2g_{}.jpeg -tile 1x2 -geometry +0+0 {}".format(
                val_dataset, i, val_dataset, i, save_path
            )
            os.system(myCmd)

            cmd = "convert {} -resize 500x1000\! {}".format(save_path, save_path)
            os.system(cmd)

            if Query[1] == Galleries[topk_idx][1]:
                copyfile(
                    save_path,
                    "tmp/{}/{}/Correct/corres_box_{}.jpeg".format(
                        val_dataset, subfol, i
                    ),
                )
            else:
                copyfile(
                    save_path,
                    "tmp/{}/{}/Wrong/corres_box_{}.jpeg".format(val_dataset, subfol, i),
                )

        # Merge query and gallery to one image
        myCmd = (
            "montage tmp/{}/{}/Cross_Correlation/corres_box_{}_[1-5]-{}x{}.jpeg -tile 5x1 "
            "-geometry +0+0 tmp/{}/{}/corres_box_{}_{}x{}.jpeg".format(
                val_dataset,
                subfol,
                idx,
                RunningParams.layer4_fm_size,
                RunningParams.layer4_fm_size,
                val_dataset,
                subfol,
                idx,
                RunningParams.layer4_fm_size,
                RunningParams.layer4_fm_size,
            )
        )
        os.system(myCmd)

    # Cross correlation maps
    def visualize_correlation_maps(
        self,
        imagenet_train_data,
        idx,
        q2g_att,
        g2q_att,
        Query,
        vis_k,
        upsample_size,
        val_dataset,
        sorted_top20,
        train_loader,
        scores,
    ):
        if not RunningParams.UNIFORM:
            subfol = "corr_map_CC"
        else:
            subfol = "corr_map_Uniform"
        HelperFunctions.check_and_mkdir("tmp/{}/".format(val_dataset))
        HelperFunctions.check_and_mkdir("tmp/{}/{}/".format(val_dataset, subfol))

        topk_indices = sorted_top20[:vis_k]

        q2g_att = q2g_att[1:]
        q2g_att = q2g_att[:, None, :, :][
            topk_indices
        ]  # Get att maps vs. 3 nearest neighbors of EMD
        q2g_att = torch.nn.functional.interpolate(
            q2g_att,
            size=(upsample_size, upsample_size),
            mode="bilinear",
            align_corners=False,
        )

        g2q_att = g2q_att[1:]
        g2q_att = g2q_att[:, None, :, :][topk_indices]
        g2q_att = torch.nn.functional.interpolate(
            g2q_att,
            size=(upsample_size, upsample_size),
            mode="bilinear",
            align_corners=False,
        )

        size = upsample_size

        for i in range(vis_k):
            image_id = scores[topk_indices][i]

            if val_dataset in (
                [Dataset.IMAGENET_A, Dataset.IMAGENET_R, Dataset.OBJECTNET_5K]
                + Dataset.IMAGENET_C_NOISE
            ):
                image_path = imagenet_train_data.samples[
                    train_loader.dataset.indices[image_id]
                ][0]
            else:
                image_path = imagenet_train_data.samples[image_id][0]

            wnid = HelperFunctions.train_extract_wnid(image_path)
            label = HelperFunctions.id_map.get(wnid).split(",")[0]
            label = label[0].lower() + label[1:]

            # Original image
            input_image = Image.open(image_path)
            input_image = input_image.resize((size, size), Image.ANTIALIAS)
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            # plt.title('Neighbor {}: {}'.format(i + 1, label))
            plt.imshow(input_image)
            plt.savefig("tmp/original.jpeg", bbox_inches="tight")
            plt.close()

            img = cv.resize(cv.imread(image_path, 0), ((size, size)))
            edges = cv.Canny(img, 100, 200)
            edges = edges - 255
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            plt.imshow(edges, cmap="gray")
            plt.savefig("tmp/Edge.jpeg", bbox_inches="tight")
            plt.close()

            q2g_att_arr = q2g_att[i][0].to("cpu")
            # Normalize the att map
            q2g_att_map = (q2g_att_arr - q2g_att_arr.min()) / (
                q2g_att_arr.max() - q2g_att_arr.min()
            )
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            # plt.title(label)
            plt.imshow(q2g_att_map, cmap=self.cMap, vmin=0, vmax=1)
            plt.colorbar(orientation="vertical")
            plt.savefig("tmp/heatmap.jpeg", bbox_inches="tight")
            plt.close()

            # Get overlay version
            myCmd = "composite -blend 10 tmp/Edge.jpeg -gravity SouthWest tmp/heatmap.jpeg tmp/overlay.jpeg"
            os.system(myCmd)

            myCmd = "montage tmp/original.jpeg tmp/overlay.jpeg -tile 1x2 -geometry +0+0 tmp/{}/q2g_{}.jpeg".format(
                val_dataset, i
            )
            os.system(myCmd)

            #         if i == 0: # Plot once for the query image
            gt_wnid = Query[1]
            label = HelperFunctions.id_map.get(gt_wnid).split(",")[0]
            label = label[0].lower() + label[1:]
            image_path = Query[0]

            # Original image
            input_image = Image.open(image_path)
            input_image = input_image.resize((size, size), Image.ANTIALIAS)
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            # plt.title('Query: {}'.format(label))
            plt.imshow(input_image)
            plt.savefig("tmp/original.jpeg", bbox_inches="tight")
            plt.close()

            img = cv.resize(cv.imread(image_path, 0), ((size, size)))
            edges = cv.Canny(img, 100, 200)
            edges = edges - 255
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            plt.imshow(edges, cmap="gray")
            plt.savefig("tmp/Edge.jpeg", bbox_inches="tight")
            plt.close()

            g2q_att_arr = g2q_att[i][0].to("cpu")
            min_value = g2q_att_arr.min()
            max_value = g2q_att_arr.max()
            # Normalize the att map
            g2q_att_map = (g2q_att_arr - min_value) / (max_value - min_value)
            fig = plt.figure()
            plt.figure(figsize=(6.0, 4.5), dpi=96)
            plt.axis("off")
            # plt.title(label)
            plt.imshow(g2q_att_map, cmap=self.cMap, vmin=0, vmax=1)
            plt.colorbar(orientation="vertical")
            plt.savefig("tmp/heatmap.jpeg", bbox_inches="tight")
            plt.close()

            # Get overlay version
            myCmd = "composite -blend 10 tmp/Edge.jpeg -gravity SouthWest tmp/heatmap.jpeg tmp/overlay.jpeg"
            os.system(myCmd)

            myCmd = "montage tmp/original.jpeg tmp/overlay.jpeg -tile 1x2 -geometry +0+0 tmp/{}/query.jpeg".format(
                val_dataset
            )
            os.system(myCmd)

            # Merge query and gallery to one image
            myCmd = "montage tmp/{}/query.jpeg tmp/{}/q2g_{}.jpeg -tile 1x2 -geometry +0+0 tmp/{}/q2g_{}.jpeg".format(
                val_dataset, val_dataset, i, val_dataset, i
            )
            os.system(myCmd)

        myCmd = "montage tmp/{}/q2g_[0-4].jpeg -tile 5x1 -geometry +0+0 tmp/{}/{}/{}_{}x{}.jpeg".format(
            val_dataset,
            val_dataset,
            subfol,
            idx,
            RunningParams.layer4_fm_size,
            RunningParams.layer4_fm_size,
        )
        os.system(myCmd)

    @staticmethod
    def visualize_optimal_plans(
        NNs, opt_plan, vis_k, val_dataset, sorted_top20
    ):

        subfol = "Optimal_plans"
        HelperFunctions.check_and_mkdir("tmp")
        HelperFunctions.check_and_mkdir("tmp/{}/".format(val_dataset))
        HelperFunctions.check_and_mkdir("tmp/{}/{}/".format(val_dataset, subfol))
        HelperFunctions.check_and_mkdir(
            "tmp/{}/{}/Optimal_plans/".format(val_dataset, subfol)
        )
        HelperFunctions.check_and_mkdir(
            "tmp/{}/{}/Correct/".format(val_dataset, subfol)
        )
        HelperFunctions.check_and_mkdir("tmp/{}/{}/Wrong/".format(val_dataset, subfol))

        topk_indices = sorted_top20[:vis_k]
        Galleries = [NNs[topk_idx] for topk_idx in topk_indices]

        # Remove the query itself
        opt_plan = opt_plan[1:]

        topk_plans = [opt_plan[nn_idx] for nn_idx in topk_indices]

        #  RECONSTRUCT GALLERY IMAGE
        for topk_idx, topk_plan in enumerate(topk_plans):
            recons_grids = []
            flow_vals = []
            for pos in range(
                RunningParams.layer4_fm_size * RunningParams.layer4_fm_size
            ):
                plan_at_pos = topk_plan[:, pos]
                plan_grid = plan_at_pos.view(
                    RunningParams.layer4_fm_size, RunningParams.layer4_fm_size
                )
                max_pos = (plan_grid == torch.max(plan_grid)).nonzero()
                recons_grids.append(max_pos)
                flow_vals.append(torch.max(plan_grid))

            flow_vals = [flow_val.item() for flow_val in flow_vals]

            max_flow_val = max(flow_vals)
            min_flow_val = min(flow_vals)
            # Normalize the optimal flow
            flow_vals = [flow_val / max_flow_val for flow_val in flow_vals]

            tmp_image_path = "/home/giang/Downloads/KNN-ImageNet/sliced_data/tmp.jpeg"
            copyfile(
                "/home/giang/Downloads/KNN-ImageNet/tmp/cub200_test/1.jpeg",
                tmp_image_path,
            )
            num_tiles = RunningParams.layer4_fm_size * RunningParams.layer4_fm_size
            # Slice the gallery images
            tiles = list(image_slicer.slice(tmp_image_path, num_tiles))

            for tile_idx, tile in enumerate(tiles):
                coords = recons_grids[tile_idx].tolist()[0]
                filename = "/home/giang/Downloads/KNN-ImageNet/sliced_data/tmp_0{}_0{}.png".format(
                    coords[0] + 1, coords[1] + 1
                )
                img = cv2.imread(filename)
                # Overwrite the patch by a new patch
                cv2.imwrite(
                    "tmp/tmp_{}.png".format(tile_idx), img * flow_vals[tile_idx]
                )
                # Change the meta-data of the tile
                tile.image = Image.open("tmp/tmp_{}.png".format(tile_idx))

            image = join(tiles)
            image.save("tmp/recons_gallery.png")
            cmd = "convert '{}' -resize 600x600\! tmp/recons_gallery.jpeg".format(
                "tmp/recons_gallery.png"
            )
            os.system(cmd)

            #  RECONSTRUCT QUERY IMAGE
            for topk_idx, topk_plan in enumerate(topk_plans):
                recons_grids = []
                flow_vals = []
                for pos in range(
                    RunningParams.layer4_fm_size * RunningParams.layer4_fm_size
                ):
                    plan_at_pos = topk_plan[pos, :]
                    plan_grid = plan_at_pos.view(
                        RunningParams.layer4_fm_size, RunningParams.layer4_fm_size
                    )
                    max_pos = (plan_grid == torch.max(plan_grid)).nonzero()
                    recons_grids.append(max_pos)
                    flow_vals.append(torch.max(plan_grid))

                flow_vals = [flow_val.item() for flow_val in flow_vals]

                max_flow_val = max(flow_vals)
                min_flow_val = min(flow_vals)
                # Normalize the optimal flow
                flow_vals = [flow_val / max_flow_val for flow_val in flow_vals]

                tmp_image_path = (
                    "/home/giang/Downloads/KNN-ImageNet/sliced_data/tmp.jpeg"
                )
                copyfile(
                    "/home/giang/Downloads/KNN-ImageNet/tmp/cub200_test/query.jpeg",
                    tmp_image_path,
                )
                num_tiles = RunningParams.layer4_fm_size * RunningParams.layer4_fm_size
                # Slice the query image
                tiles = list(image_slicer.slice(tmp_image_path, num_tiles))

                for tile_idx, tile in enumerate(tiles):
                    coords = recons_grids[tile_idx].tolist()[0]
                    filename = "/home/giang/Downloads/KNN-ImageNet/sliced_data/tmp_0{}_0{}.png".format(
                        coords[0] + 1, coords[1] + 1
                    )
                    img = cv2.imread(filename)
                    # Overwrite the patch by a new patch
                    cv2.imwrite(
                        "tmp/tmp_{}.png".format(tile_idx), img * flow_vals[tile_idx]
                    )
                    # Change the meta-data of the tile
                    tile.image = Image.open("tmp/tmp_{}.png".format(tile_idx))

                image = join(tiles)
                image.save("tmp/recons_query.png")
                cmd = "convert '{}' -resize 600x600\! tmp/recons_query.jpeg".format(
                    "tmp/recons_query.png"
                )
                os.system(cmd)
