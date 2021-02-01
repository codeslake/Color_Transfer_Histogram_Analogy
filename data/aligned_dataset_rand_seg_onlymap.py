import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform_lab, no_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class AlignedDataset_Rand_Seg_onlymap(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.img_type = opt.img_type

        self.dir_A =   os.path.join(opt.dataroot, 'input')
        self.dir_B =   os.path.join(opt.dataroot, 'target')
        self.dir_A_Map = os.path.join(opt.dataroot, 'seg_in')
        self.dir_B_Map = os.path.join(opt.dataroot, 'seg_tar')

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)

        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)

        self.A_paths_map = make_dataset(self.dir_A_Map)
        self.A_paths_map = sorted(self.A_paths_map)

        self.B_paths_map = make_dataset(self.dir_B_Map)
        self.B_paths_map = sorted(self.B_paths_map)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_type = get_transform_lab(opt)
        self.transform_no = no_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        if self.opt.is_SR is True:
            A_path_map = self.A_paths_map[index % self.A_size]
            B_path_map = self.B_paths_map[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_type(A_img)
        B = self.transform_type(B_img)

        if self.opt.is_SR is False:
            A_map = np.zeros_like(np.array(A))
            B_map = np.zeros_like(np.array(B))
        else:
            A_map=self.transform_no(Image.open(A_path_map))
            B_map=self.transform_no(Image.open(B_path_map))

        return {'A': A, 'B': B, 'A_map': A_map, 'B_map': B_map,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDataset_Rand_Seg_onlymap'

