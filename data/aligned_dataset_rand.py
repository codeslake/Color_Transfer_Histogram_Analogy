import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, no_transform, get_transform_lab, get_transform_filter_sat, get_transform_filter_red, get_transform_filter_blue, get_transform_hueshiftlab
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class AlignedDataset_Rand(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.img_type = opt.img_type

        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B_exC_resized')
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'C_temp1')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'C_temp2')

        #print(self.opt.phase)
        #print(VIURN)

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        
        self.A_size = len(self.A_paths)

        self.B_size = len(self.B_paths)

        if self.img_type == 'rgb':
            self.transform_type = no_transform(opt)
        elif self.img_type == 'hsv':
            self.transform_type = get_transform_hsv(opt)
        elif self.img_type == 'lab':
            #self.transform_type = get_transform_lab(opt)
            self.transform_type = get_transform_hueshiftlab(opt)
        else:
            print(ERROR)

        
        self.transform_type = get_transform_lab(opt)
        self.transform_type_test = get_transform_hueshiftlab(opt)

        #self.transform = get_transform_lab(opt)
        self.transform_no = no_transform(opt)
        self.transform_lab = get_transform_lab(opt)
        self.transform_lab_sat = get_transform_filter_sat(opt)
        self.transform_lab_red = get_transform_filter_red(opt)
        self.transform_lab_blue = get_transform_filter_blue(opt)

    def __getitem__(self, index):
        # For Aligned
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]

        # For Unaligned
        #index_B = random.randint(0, self.B_size - 1)
        #B_path = self.B_paths[index_B % self.B_size]

        #if self.opt.serial_batches:
        #    index_B = index % self.B_size
        #else:
        #    index_B = random.randint(0, self.B_size - 1)
        
        A_img = Image.open(A_path)#.convert('RGB')
        B_img = Image.open(B_path)#.convert('RGB')


        #if self.opt.phase == 'Train':
        #    if random.random()>0.5:
        #        A = self.transform_lab(A_img)
        #    else:
        #        B = self.transform_lab(B_img)
        #else:
        #    A = self.transform_lab(A_img)

        #r = random.random()

        #if self.opt.phase == 'Train':
        #    if r>0.6:
        #        B = self.transform_lab_sat(B_img)
        #    elif r > 0.3 and r<= 0.6:
        #        B = self.transform_lab_red(B_img)
        #    else:
        #        B = self.transform_lab_blue(B_img)
        #else:
        #    B = self.transform_lab(B_img)

        A = self.transform_type(A_img)
        B = self.transform_type(B_img)
        C = self.transform_type_test(A_img)
        
        #if self.opt.phase == 'Train':
        #    if random.random()>0.5:
        #        A = self.transform_lab(A_img)
        #    else:
        #        A = self.transform_lab(B_img)
        #else:
        #    A = self.transform_lab(A_img)
        
        #B = self.transform_lab(B_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'C': C,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDataset_Rand'

