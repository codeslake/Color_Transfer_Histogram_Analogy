import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, no_transform, get_transform_lab, get_transform_filter_sat, get_transform_filter_red, get_transform_filter_blue, get_transform_hueshiftlab, get_transform_hueshiftlab2
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torchvision.transforms as transforms

class AlignedDataset_Rand_Multi(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.img_type = opt.img_type
        self.pair_ratio = opt.pair_ratio

        # FiveK dataset
        self.dir_R = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'B_exA_resized')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B_exB_resized')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'B_exC_resized')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'B_exD_resized')
        self.dir_E = os.path.join(opt.dataroot, opt.phase + 'B_exE_resized')

        # DPED dataset
        #self.dir_dped_sn_src = os.path.join('/root/Mango/Common/Dataset/dped/sony/training_data/sony')
        #self.dir_dped_sn_tgt = os.path.join('/root/Mango/Common/Dataset/dped/sony/training_data/canon')
        #self.dir_dped_bb_src = os.path.join('/root/Mango/Common/Dataset/dped/blackberry/training_data/blackberry')
        #self.dir_dped_bb_tgt = os.path.join('/root/Mango/Common/Dataset/dped/blackberry/training_data/canon')
        #self.dir_dped_ip_src = os.path.join('/root/Mango/Common/Dataset/dped/iphone/training_data/iphone')
        #self.dir_dped_ip_tgt = os.path.join('/root/Mango/Common/Dataset/dped/iphone/training_data/canon')

        
        self.R_paths = make_dataset(self.dir_R)
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)
        self.D_paths = make_dataset(self.dir_D)
        self.E_paths = make_dataset(self.dir_E)

        #self.DPED_sn_src = make_dataset(self.dir_dped_sn_src)
        #self.DPED_sn_tgt = make_dataset(self.dir_dped_sn_tgt)
        #self.DPED_bb_src = make_dataset(self.dir_dped_bb_src)
        #self.DPED_bb_tgt = make_dataset(self.dir_dped_bb_tgt)
        #self.DPED_ip_src = make_dataset(self.dir_dped_ip_src)
        #self.DPED_ip_tgt = make_dataset(self.dir_dped_ip_tgt)
        

        self.R_paths = sorted(self.R_paths)
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.D_paths = sorted(self.D_paths)
        self.E_paths = sorted(self.E_paths)

        
        self.R_size = len(self.R_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        self.E_size = len(self.E_paths)


        # Channel Decision
        if self.img_type == 'rgb':
            self.transform_type = no_transform(opt)
        elif self.img_type == 'hsv':
            self.transform_type = get_transform_hsv(opt)
        elif self.img_type == 'lab':
            #self.transform_type = get_transform_lab(opt)

            self.transform_type = get_transform_hueshiftlab(opt)
            self.transform_type2 = get_transform_hueshiftlab2(opt)
        else:
            print(ERROR)

        #self.transform = get_transform_lab(opt)
        self.transform_no = no_transform(opt)
        self.transform_lab = get_transform_lab(opt)
        self.transform_lab_sat = get_transform_filter_sat(opt)
        self.transform_lab_red = get_transform_filter_red(opt)
        self.transform_lab_blue = get_transform_filter_blue(opt)

    def __getitem__(self, index):

        if random.random() >= self.pair_ratio:
            pair = True 
        else:
            pair = False

        R_path = self.R_paths[(index % self.R_size) if pair == True else ((random.randint(0, self.R_size -1)) % self.R_size)]
        A_path = self.A_paths[(index % self.A_size) if pair == True else ((random.randint(0, self.A_size -1)) % self.A_size)]
        B_path = self.B_paths[(index % self.B_size) if pair == True else ((random.randint(0, self.B_size -1)) % self.B_size)]
        C_path = self.C_paths[(index % self.C_size) if pair == True else ((random.randint(0, self.C_size -1)) % self.C_size)]
        D_path = self.D_paths[(index % self.D_size) if pair == True else ((random.randint(0, self.D_size -1)) % self.D_size)]
        E_path = self.E_paths[(index % self.E_size) if pair == True else ((random.randint(0, self.E_size -1)) % self.E_size)]

        ## For Aligned
        #R_path = self.R_paths[index %self.R_size]
        #A_path = self.A_paths[index %self.A_size]
        #B_path = self.B_paths[index %self.B_size]
        #C_path = self.C_paths[index %self.C_size]
        #D_path = self.D_paths[index %self.D_size]
        #E_path = self.E_paths[index %self.E_size]

        # For Unaligned
        #index_B = random.randint(0, self.B_size - 1)
        #B_path = self.B_paths[index_B % self.B_size]
        #index_B = random.randint(0, self.B_size - 1)
        #B_path = self.B_paths[index_B % self.B_size]

        #if self.opt.serial_batches:
        #    index_B = index % self.B_size
        #else:
        #    index_B = random.randint(0, self.B_size - 1)
        
        r = random.random() * 1.2
        s = random.random() * 1.2

        if r>1.0:
            A_img = Image.open(R_path)
        elif r>0.8 and r<=1.0:
            A_img = Image.open(A_path)
        elif r>0.6 and r<=0.8:
            A_img = Image.open(B_path)
        elif r>0.4 and r<=0.6:
            A_img = Image.open(C_path)
        elif r>0.2 and r<=0.4:
            A_img = Image.open(D_path)
        else:
            A_img = Image.open(E_path)

        if s>1.0:
            B_img = Image.open(R_path)
        elif s>0.8 and s<=1.0:
            B_img = Image.open(A_path)
        elif s>0.6 and s<=0.8:
            B_img = Image.open(B_path)
        elif s>0.4 and s<=0.6:
            B_img = Image.open(C_path)
        elif s>0.2 and s<=0.4:
            B_img = Image.open(D_path)
        else:
            B_img = Image.open(E_path)

        #i, j, h, w
        A = self.transform_type(A_img)
        B = self.transform_type2(B_img)


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
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path, 'pair': pair}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDataset_Rand_Multi'

