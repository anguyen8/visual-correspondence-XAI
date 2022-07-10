import numpy as np
import torch
import torchvision.models as models
from numpy import matlib as mb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from DeformableResNet import resnet50_features

to_np = lambda x: x.data.to("cpu").numpy()


def compute_spatial_similarity(conv1, conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    conv1 = conv1.reshape(-1, 7 * 7).T
    conv2 = conv2.reshape(-1, 7 * 7).T

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


def normalize_array(x):
    x = np.asarray(x).copy()
    x -= np.min(x)
    x /= np.max(x)
    return x


def apply_threshold(x, t):
    x = np.asarray(x).copy()
    x[x < t] = 0
    return x


def generate_mask(x, t):
    v = np.zeros_like(x)
    v[x >= t] = 1
    return v


def get_transforms(args_transform, chm_args):
    # TRANSFORMS
    cosine_transform_target = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    chm_transform_target = transforms.Compose(
        [
            transforms.Resize(chm_args["img_size"]),
            transforms.CenterCrop((chm_args["img_size"], chm_args["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args_transform == "multi":
        cosine_transform_source = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        chm_transform_source = transforms.Compose(
            [
                transforms.Resize((chm_args["img_size"], chm_args["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif args_transform == "single":
        cosine_transform_source = transforms.Compose(
            [
                transforms.Resize(chm_args["img_size"]),
                transforms.CenterCrop((chm_args["img_size"], chm_args["img_size"])),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        chm_transform_source = transforms.Compose(
            [
                transforms.Resize(chm_args["img_size"]),
                transforms.CenterCrop((chm_args["img_size"], chm_args["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return (
        chm_transform_source,
        chm_transform_target,
        cosine_transform_source,
        cosine_transform_target,
    )


def clamp(x, min_value, max_value):
    return max(min_value, min(x, max_value))


def keep_top5(input_array, K=5):
    top_5 = np.sort(input_array.reshape(-1))[::-1][K - 1]
    masked = np.zeros_like(input_array)
    masked[input_array >= top_5] = 1
    return masked


def arg_topK(input_array, topK=5):
    return np.argsort(input_array.T.reshape(-1))[::-1][:topK]


class KNNSupportSet:
    def __init__(self, train_folder, val_folder, knn_scores, custom_val_labels=None):
        self.train_data = ImageFolder(root=train_folder)
        self.val_data = ImageFolder(root=val_folder)
        self.knn_scores = knn_scores

        if custom_val_labels is None:
            self.val_labels = np.asarray([x[1] for x in self.val_data.imgs])
        else:
            self.val_labels = custom_val_labels

        self.train_labels = np.asarray([x[1] for x in self.train_data.imgs])

    def get_knn_predictions(self, k=20):
        knn_predictions = [
            np.argmax(np.bincount(self.train_labels[self.knn_scores[I][::-1][:k]]))
            for I in range(len(self.knn_scores))
        ]
        knn_accuracy = (
                100
                * np.sum((np.asarray(knn_predictions) == self.val_labels))
                / len(self.val_labels)
        )
        return knn_predictions, knn_accuracy

    def get_support_set(self, selected_index, top_N=20):
        support_set = self.knn_scores[selected_index][-top_N:][::-1]
        return [self.train_data.imgs[x][0] for x in support_set]

    def get_support_set_labels(self, selected_index, top_N=20):
        support_set = self.knn_scores[selected_index][-top_N:][::-1]
        return [self.train_data.imgs[x][1] for x in support_set]

    def get_image_and_label_by_id(self, q_id):
        q = self.val_data.imgs[q_id][0]
        ql = self.val_data.imgs[q_id][1]
        return (q, ql)

    def get_folder_name(self, q_id):
        q = self.val_data.imgs[q_id][0]
        return q.split("/")[-2]

    def get_top5_knn(self, query_id, k=20):
        knn_pred, knn_acc = self.get_knn_predictions(k=k)
        top_5s_index = np.where(
            np.equal(
                self.train_labels[self.knn_scores[query_id][::-1]], knn_pred[query_id]
            )
        )[0][:5]
        top_5s = self.knn_scores[query_id][::-1][top_5s_index]
        top_5s_files = [self.train_data.imgs[x][0] for x in top_5s]
        return top_5s_files

    def get_topK_knn(self, query_id, k=20):
        knn_pred, knn_acc = self.get_knn_predictions(k=k)
        top_ks_index = np.where(
            np.equal(
                self.train_labels[self.knn_scores[query_id][::-1]], knn_pred[query_id]
            )
        )[0][:k]
        top_ks = self.knn_scores[query_id][::-1][top_ks_index]
        top_ks_files = [self.train_data.imgs[x][0] for x in top_ks]
        return top_ks_files

    def get_foldername_for_label(self, label):
        for i in range(len(self.train_data)):
            if self.train_data.imgs[i][1] == label:
                return self.train_data.imgs[i][0].split("/")[-2]

    def get_knn_confidence(self, query_id, k=20):
        return np.max(
            np.bincount(self.train_labels[self.knn_scores[query_id][::-1][:k]])
        )


class CosineCustomDataset(Dataset):
    r"""Parent class of PFPascal, PFWillow, and SPair"""

    def __init__(self, query_image, supporting_set, source_transform, target_transform):
        r"""XAICustomDataset constructor"""
        super(CosineCustomDataset, self).__init__()

        self.supporting_set = supporting_set
        self.query_image = [query_image] * len(supporting_set)

        self.source_transform = source_transform
        self.target_transform = target_transform

    def __len__(self):
        r"""Returns the number of pairs"""
        return len(self.supporting_set)

    def __getitem__(self, idx):
        r"""Constructs and return a batch"""

        # Image name
        batch = dict()
        batch["src_imname"] = self.query_image[idx]
        batch["trg_imname"] = self.supporting_set[idx]

        # Image as numpy (original width, original height)
        src_pil = self.get_image(self.query_image, idx)
        trg_pil = self.get_image(self.supporting_set, idx)

        batch["src_imsize"] = src_pil.size
        batch["trg_imsize"] = trg_pil.size

        # Image as tensor
        batch["src_img"] = self.source_transform(src_pil)
        batch["trg_img"] = self.target_transform(trg_pil)

        # Total number of pairs in training split
        batch["datalen"] = len(self.query_image)
        return batch

    def get_image(self, image_pathes, idx):
        r"""Reads PIL image from path"""
        path = image_pathes[idx]
        return Image.open(path).convert("RGB")


class PairedLayer4Extractor(torch.nn.Module):
    """
    Extracting layer-4 embedding for source and target images using ResNet-50 features
    """
    def __init__(self):
        super(PairedLayer4Extractor, self).__init__()

        self.modelA = models.resnet50(pretrained=True)
        self.modelA.cuda()
        self.modelA.eval()

        self.modelB = models.resnet50(pretrained=True)
        self.modelB.cuda()
        self.modelB.eval()

        self.a_embeddings = None
        self.b_embeddings = None

        def a_hook(module, input, output):
            self.a_embeddings = output

        def b_hook(module, input, output):
            self.b_embeddings = output

        self.modelA._modules.get("layer4").register_forward_hook(a_hook)
        self.modelB._modules.get("layer4").register_forward_hook(b_hook)

    def forward(self, inputs):
        inputA, inputB = inputs
        self.modelA(inputA)
        self.modelB(inputB)

        return self.a_embeddings, self.b_embeddings

    def __repr__(self):
        return "PairedLayer4Extractor"


class iNaturalistPairedLayer4Extractor(torch.nn.Module):
    """
    Extracting layer-4 embedding for source and target images using iNaturalist ResNet-50 features
    """
    def __init__(self):
        super(iNaturalistPairedLayer4Extractor, self).__init__()

        self.modelA = resnet50_features(inat=True, pretrained=True)
        self.modelA.cuda()
        self.modelA.eval()

        self.modelB = resnet50_features(inat=True, pretrained=True)
        self.modelB.cuda()
        self.modelB.eval()

        self.source_embedding = None
        self.target_embedding = None

    def forward(self, inputs):
        source_image, target_image = inputs
        self.source_embedding = self.modelA(source_image)
        self.target_embedding = self.modelB(target_image)

        return self.a_embeddings, self.b_embeddings

    def __repr__(self):
        return "iNatPairedLayer4Extractor"
