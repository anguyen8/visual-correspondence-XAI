import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import matlib as mb
from params import *

RunningParams = RunningParams()


class EMDFunctions(object):
    def __init__(self):
        pass

    @ staticmethod
    def Sinkhorn(K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-1
        for i in range(100):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(
                K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)
            ).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        if i == 99:
            print("Sinkhorn no solutions!")
            solution = False
        else:
            solution = True
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T, solution

    @ staticmethod
    def compute_emd_sim(
        K_value, fb_center, fb, use_uniform, num_patch
    ):
        cc_q2g_maps = []
        cc_g2q_maps = []
        query = fb[0].cpu().detach().numpy()
        for i in range(K_value + 1):
            gallery = fb[i].cpu().detach().numpy()
            heatmap1, heatmap2 = EMDFunctions.compute_spatial_similarity(query, gallery)
            # 7x7
            cc_g2q_maps.append(torch.from_numpy(heatmap1))
            cc_q2g_maps.append(torch.from_numpy(heatmap2))

        # 51x7x7
        cc_q2g_maps = torch.stack(cc_q2g_maps, dim=0)
        cc_g2q_maps = torch.stack(cc_g2q_maps, dim=0)

        # 51x49
        cc_q2g_weights = torch.flatten(cc_q2g_maps, start_dim=1)
        cc_g2q_weights = torch.flatten(cc_g2q_maps, start_dim=1)

        N = fb.size()[0]  # 51
        R = fb.size()[2] * fb.size()[3]  # 7*7=49

        # 51x2048x7x7 -> 51x2048x49
        fb = torch.flatten(fb, start_dim=2)
        fb = torch.nn.functional.normalize(fb, p=2, dim=1)

        fb_center = torch.flatten(fb_center, start_dim=2)
        fb_center = torch.nn.functional.normalize(fb_center, p=2, dim=1)

        # 51x49x49
        sim = torch.einsum("cm,ncs->nsm", fb[0], fb_center).contiguous().view(N, R, R)
        dis = 1.0 - sim
        K = torch.exp(-dis / 0.05)

        if use_uniform:
            u = torch.zeros(N, R).fill_(1.0 / R)
            v = torch.zeros(N, R).fill_(1.0 / R)
        else:
            if RunningParams.UNEQUAL_W is True:
                sum_g2q_att = cc_q2g_weights.sum(dim=1, keepdims=True) + 1e-5
                u = cc_q2g_weights / sum_g2q_att
                v = cc_g2q_weights / sum_g2q_att
            else:
                u = cc_q2g_weights
                u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
                v = cc_g2q_weights
                v = v / (v.sum(dim=1, keepdims=True) + 1e-5)

        u = u.to(sim.dtype).cuda()
        v = v.to(sim.dtype).cuda()
        T, solution = EMDFunctions.Sinkhorn(K, u, v)

        if RunningParams.CLOSEST_PATCH_COMPARISON:
            dists = []
            for i in range(K_value + 1):
                pair_opt_plan = torch.flatten(T[i], start_dim=0).to("cpu")
                pair_sim = torch.flatten(sim[i], start_dim=0).to("cpu")
                sorted_ts = torch.argsort(pair_opt_plan)
                # sorted_top = sorted_ts[-num_patch:]
                sorted_top = sorted_ts[-num_patch:]
                opt_point_top = np.array([pair_opt_plan[idx] for idx in sorted_top])
                sim_point_top = np.array([pair_sim[idx] for idx in sorted_top])
                dist = torch.as_tensor(np.sum(opt_point_top * sim_point_top))
                dists.append(dist)
            sim = torch.stack(dists, dim=0).cuda()
        else:
            sim = torch.sum(T * sim, dim=(1, 2))
        return sim, cc_q2g_maps, cc_g2q_maps, T

    @ staticmethod
    def compute_spatial_similarity(conv1, conv2):
        """
        Takes in the last convolutional layer from two images, computes the pooled output
        feature, and then generates the spatial similarity map for both images.
        """
        conv1 = conv1.reshape(-1, conv1.shape[1] * conv1.shape[2]).T
        conv2 = conv2.reshape(-1, conv2.shape[1] * conv2.shape[2]).T

        # conv2 = conv2.reshape(-1, 7 * 7).T

        pool1 = np.mean(conv1, axis=0)
        pool2 = np.mean(conv2, axis=0)
        out_sz = (int(np.sqrt(conv1.shape[0])), int(np.sqrt(conv1.shape[0])))
        conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
        conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
        im_similarity = np.zeros((conv1_normed.shape[0], conv1_normed.shape[0]))

        for zz in range(conv1_normed.shape[0]):
            repPx = mb.repmat(conv1_normed[zz, :], conv1_normed.shape[0], 1)
            im_similarity[zz, :] = np.multiply(repPx, conv2_normed).sum(axis=1)
        similarity1 = np.reshape(np.sum(im_similarity, axis=1), out_sz)
        similarity2 = np.reshape(np.sum(im_similarity, axis=0), out_sz)
        return similarity1, similarity2

    # classmethod so that we do not need to instantiate
    @classmethod
    def compute_emd_distance(self, K_value, fb_center, fb, use_uniform, num_patch):
        sim, q2g_att, g2q_att, opt_plan = self.compute_emd_sim(
            K_value, fb_center, fb, use_uniform, num_patch
        )
        if RunningParams.MIX_DISTANCE:
            cosine_sim = torch.einsum("chw,nchw->n", fb[0], fb_center)
            cosine_sim_max = cosine_sim.max()
            cosine_sim = cosine_sim / cosine_sim_max
            return sim + cosine_sim, q2g_att, g2q_att, opt_plan
        else:
            return sim, q2g_att, g2q_att, opt_plan

