import os
import os.path as osp
import glob
import numpy as np
import cv2
from utils.augmentation import *
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_mask(img_gt_path):
    """"
    Thay đổi ảnh mask trong grayscale về 0, 1, 2(background/non-neoplastic/neoplastic).
    @param: img_gt_path(path to folder contains gt) 
    @return: img_gt sau khi quy về mask 0, 1, 2
    """
    img_gt = cv2.imread(img_gt_path, 0)
    img_gt[img_gt < 10] = 0
    img_gt[(img_gt > 140) & (img_gt < 160)] = 1
    img_gt[(img_gt > 65) & (img_gt < 90)] = 2
    return img_gt


def class_prob(tr_gt_paths):
    """
    Do dữ liệu mất cân bằng, pp đơn giản sử dụng stratify để đảm bảo phân phối nhãn
    @param: tr_gt_paths(list): danh sách path đến gt
    @return: cls_prob(list): gồm 0, 1 đánh dấu lớp neo_plastic
    """
    cls_prob = []
    non_path = []

    for index, img_gt_path in enumerate(tr_gt_paths):
        img_gt = read_mask(img_gt_path)
        if(1 in np.unique(img_gt)):
            cls_prob.append(1)
            non_path.append(img_gt_path)
        else:
            cls_prob.append(0)

    return cls_prob, non_path


def make_datafile_path_1(root_path):
    """
    Hàm lấy danh sách các đường dẫn đến train/gt/test
    @param: root_path(String): Đường dẫn đến thư mục project
    @return: (train_path_list, train_gt_path_list, test_path_list): tuple gồm 3 danh sách đường dẫn
    """
    train_path = osp.join(root_path, 'data/classes/train.txt')
    val_path = osp.join(root_path, 'data/classes/val.txt')
    test_path = osp.join(root_path, 'data/test/test/*')

    template = osp.join(root_path, 'data/train/train/%s')

    train_path_list = []
    val_path_list = []
    test_path_list = []

    for file_path in open(train_path):
        file_path = file_path.strip()
        train_path_list.append(template % file_path)

    for file_path in open(val_path):
        file_path = file_path.strip()
        val_path_list.append(template % file_path)

    for file_path in glob.glob(test_path):
        test_path_list.append(file_path)

    return train_path_list, val_path_list, test_path_list

def make_datafile_path(root_path):
    """
    Hàm lấy danh sách các đường dẫn đến train/gt/test
    @param: root_path(String): Đường dẫn đến thư mục project
    @return: (train_path_list, test_path_list): tuple gồm 3 danh sách đường dẫn
    """
    train_path = osp.join(root_path, 'data/train/train/*')
    test_path = osp.join(root_path, 'data/test/test/*')

    train_path_list = []
    test_path_list = []

    for file_path in glob.glob(train_path):
        train_path_list.append(file_path)

    for file_path in glob.glob(test_path):
        test_path_list.append(file_path)

    return train_path_list, test_path_list


class DataTransform():
    def __init__(self, input_size):
        self.data_augmentation = Compose([
            RandomRotate(angle=[0, 30], p=1),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            # RandomGaussianNoise(p=0.3),
            # ShiftScaleRotate(0.3),
            RandomScale(p=1),
            Deformation(p=0.3),
            ColorJitter(p=0.2),
            Resize(input_size=input_size),
            NormalizeTensor()
        ])

        self.data_transform = Compose([
            Resize(input_size=input_size),
            NormalizeTensor()
        ])

    def __call__(self, phase, img, img_gt):
        if phase == "train":
            img, img_gt = self.data_augmentation(img, img_gt)
        else: 
            img, img_gt = self.data_transform(img, img_gt)
        return img, img_gt


class MyDataset(data.Dataset):
    def __init__(self, img_paths, phase, transform):
        self.img_paths = img_paths
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_gt_path = img_path.replace('train', 'train_gt')
        img = Image.open(img_path)
        img_gt = read_mask(img_gt_path)
        img_gt = T.ToPILImage()(img_gt)
        img, img_gt = self.transform(phase=self.phase, img=img, img_gt=img_gt)

        return img, img_gt