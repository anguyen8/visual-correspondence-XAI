import os
import shutil
from shutil import *

import numpy as np
import torch
from PIL import Image


class HelperFunctions(object):
    def __init__(self):
        # Some Helper functions
        self.id_map = self.load_imagenet_id_map()
        self.key_list = list(self.id_map.keys())
        self.val_list = list(self.id_map.values())

    @staticmethod
    def concat(x):
        return np.concatenate(x, axis=0)

    @staticmethod
    def to_np(x):
        return x.data.to("cpu").numpy()

    @staticmethod
    def to_ts(x):
        return torch.from_numpy(x)

    @staticmethod
    def train_extract_wnid(x):
        return x.split("train/")[1].split("/")[0]

    @staticmethod
    def val_extract_wnid(x):
        return x.split("/")[-2]

    @staticmethod
    def rm_and_mkdir(path):
        if os.path.isdir(path) is True:
            rmtree(path)
        os.mkdir(path)

    @staticmethod
    def copy_files(files, dir):
        for file in files:
            shutil.copy(file, dir)

    @staticmethod
    def is_grey_scale(img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b:
                    return False
        return True

    @staticmethod
    def check_and_mkdir(f):
        if not os.path.exists(f):
            os.mkdir(f)
        else:
            pass

    @staticmethod
    def check_and_rm(f):
        if os.path.exists(f):
            shutil.rmtree(f)
        else:
            pass

    # Added for loading ImageNet classes
    @staticmethod
    def load_imagenet_id_map():
        """
        Load ImageNet ID dictionary.
        return;
        """

        input_f = open('/home/giang/Downloads/visual-correspondence-XAI/EMD-Corr/synset_words.txt')
        label_map = {}
        for line in input_f:
            parts = line.strip().split(" ")
            (num, label) = (parts[0], " ".join(parts[1:]))
            label_map[num] = label

        input_f.close()
        return label_map

    @staticmethod
    def convert_imagenet_label_to_id(
        label_map, key_list, val_list, prediction_class
    ):
        """
        Convert imagenet label to ID: for example - 245 -> "French bulldog" -> n02108915
        :param label_map:
        :param key_list:
        :param val_list:
        :param prediction_class:
        :return:
        """
        class_to_label = label_map[prediction_class]
        prediction_id = key_list[val_list.index(class_to_label)]
        return prediction_id

    @staticmethod
    def convert_imagenet_id_to_label(key_list, class_id):
        """
        Convert imagenet label to ID: for example - n02108915 -> "French bulldog" -> 245
        :param class_id:
        :param key_list:
        :return:
        """
        return key_list.index(str(class_id))

    @staticmethod
    def get_candidate_wnids(
        predicted_wnid, top_predicted_wnids, gt_wnids, candidate_num=3
    ):
        """
        This function returns a list of candidate wnids for Task 2 Human study.
        Please check return value of this functions. Skip the query if None
        :param predicted_wnid: The predicted wnid
        :param top_predicted_wnids: The top predicted wnids from your classifiers in a list []
        :param gt_wnids: The wnids from ImageNetReaL in a list []
        :param candidate_num: The numbers of candidate wnids
        :return: candidate_wnids: The list of candidate wnids in a list []
        """

        # unresolvable distinctions: (sunglass, sunglasses), (bathtub, tub), (promontory, cliff) and (laptop,
        # notebook), (projectile, missile), (maillot, maillot), (crane, crane)
        # translate to wnids: (n04355933, n04356056), (n02808440, n04493381), (n09399592, n09246464),
        # (n03642806, n03832673), (n04008634, n03773504), (n03710637, n03710721), (n02012849, n03126707)

        unresolvable_distinctions = [
            ("n04355933", "n04356056"),
            ("n02808440", "n04493381"),
            ("n09399592", "n09246464"),
            ("n03642806", "n03832673"),
            ("n04008634", "n03773504"),
            ("n03710637", "n03710721"),
            ("n02012849", "n03126707"),
        ]
        ud_dict = {}
        for unresolvable_distinction in unresolvable_distinctions:
            ud_dict[unresolvable_distinction[0]] = unresolvable_distinction[1]
            ud_dict[unresolvable_distinction[1]] = unresolvable_distinction[0]

        candidate_wnids = [
            predicted_wnid
        ]  # Add the AI prediction to the candidate list
        visited_wnid = [predicted_wnid]

        for wnid in gt_wnids + top_predicted_wnids:
            if wnid == predicted_wnid:
                continue

            # If the similar label was visited and added to candidate_wnids already --> skip
            if (
                wnid in ud_dict
                and ud_dict[wnid] in visited_wnid
                or wnid in candidate_wnids
            ):
                continue
            else:
                visited_wnid.append(wnid)
                candidate_wnids.append(wnid)

        if len(candidate_wnids) >= candidate_num:
            return candidate_wnids[:candidate_num]
        else:
            return None
