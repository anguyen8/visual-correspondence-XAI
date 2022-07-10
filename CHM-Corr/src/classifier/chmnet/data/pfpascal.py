r""" PF-PASCAL dataset """

import os

import scipy.io as sio
import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset


class PFPascalDataset(CorrespondenceDataset):

    def __init__(self, benchmark, datapath, thres, split):
        r""" PF-PASCAL dataset constructor """
        super(PFPascalDataset, self).__init__(benchmark, datapath, thres, split)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1

        if split == 'trn':
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if len(torch.isnan(src_kk).nonzero()) != 0 or \
                        len(torch.isnan(trg_kk).nonzero()) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

    def __getitem__(self, idx):
        r""" Constructs and returns a batch for PF-PASCAL dataset """
        batch = super(PFPascalDataset, self).__getitem__(idx)

        # Object bounding-box (resized following self.img_size)
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'])
        batch['pckthres'] = self.get_pckthres(batch,  batch['trg_imsize'])

        # Horizontal flipping key-points during training
        if self.split == 'trn' and self.flip[idx]:
            self.horizontal_flip(batch)
            batch['flip'] = 1
        else:
            batch['flip'] = 0

        return batch

    def get_bbox(self, bbox_list, idx, imsize):
        r""" Returns object bounding-box """
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.img_size / imsize[0])
        bbox[1::2] *= (self.img_size / imsize[1])
        return bbox

    def horizontal_flip(self, batch):
        tmp = batch['src_bbox'][0].clone()
        batch['src_bbox'][0] = batch['src_img'].size(2) - batch['src_bbox'][2]
        batch['src_bbox'][2] = batch['src_img'].size(2) - tmp

        tmp = batch['trg_bbox'][0].clone()
        batch['trg_bbox'][0] = batch['trg_img'].size(2) - batch['trg_bbox'][2]
        batch['trg_bbox'][2] = batch['trg_img'].size(2) - tmp

        batch['src_kps'][0][:batch['n_pts']] = batch['src_img'].size(2) - batch['src_kps'][0][:batch['n_pts']]
        batch['trg_kps'][0][:batch['n_pts']] = batch['trg_img'].size(2) - batch['trg_kps'][0][:batch['n_pts']]

        batch['src_img'] = torch.flip(batch['src_img'], dims=(2,))
        batch['trg_img'] = torch.flip(batch['trg_img'], dims=(2,))


def read_mat(path, obj_name):
    r""" Reads specified objects from Matlab data file. (.mat) """
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj
