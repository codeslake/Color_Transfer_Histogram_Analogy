import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
#from torch.utils.serialization import load_lua
import torch.nn as nn
import torchvision
import random
import cv2
import torchfile
import torch.nn.functional as F
import copy
import seaborn as sns
import matplotlib.pyplot as plt


class ColorHistogram_Model(BaseModel):
    def name(self):
        return 'ColorHistogram_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize 

        self.hist_l = opt.l_bin
        self.hist_ab = opt.ab_bin
        self.img_type = opt.img_type

        self.IRN = networks.IRN(3, 3, opt.ngf, opt.network, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.HEN = networks.HEN((self.hist_l+1), 64, opt.ngf, opt.network_H, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids).cuda()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          
            self.load_network(self.IRN, 'G_A', which_epoch)
            self.load_network(self.HEN, 'C_A', which_epoch)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        
        input_A = input['A']
        input_B = input['B']
        input_A_Map = input['A_map']
        input_B_Map = input['B_map']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])

        self.input_A = input_A
        self.input_B = input_B

        self.input_A_Map = input_A_Map
        self.input_B_Map = input_B_Map
        self.input_A_Seg, self.input_B_Seg, self.input_SegNum = self.MakeLabelFromMap(self.input_A_Map,self.input_B_Map)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def forward(self):
        self.real_A = self.input_A
        self.real_B = self.input_B
        self.A_seg = self.input_A_Seg
        self.B_seg = self.input_B_Seg

    def test(self):
        with torch.no_grad():

            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
    
            self.A_seg = Variable(self.input_A_Seg)
            self.B_seg = Variable(self.input_B_Seg)
            self.A_map = Variable(self.input_A_Map)

            # max 700 px with aspect ratio
            if (self.real_A.size(2)>700) or (self.real_A.size(3)>700):
                aspect_ratio = self.real_A.size(2) / self.real_A.size(3)
                if (self.real_A.size(2) > self.real_A.size(3)):
                    self.real_A = F.upsample(self.real_A , size = ( 700 ,  int(700 / aspect_ratio)), mode = 'bilinear')
                else:
                    self.real_A = F.upsample(self.real_A , size = ( int(700 * aspect_ratio),  700 ), mode = 'bilinear')
            # max 700 px with aspect ratio
            if (self.real_B.size(2)>700) or (self.real_B.size(3)>700):
                aspect_ratio = self.real_B.size(2) / self.real_B.size(3)
                if (self.real_B.size(2) > self.real_B.size(3)):
                    self.real_B = F.upsample(self.real_B , size = ( 700 ,  int(700 / aspect_ratio)), mode = 'bilinear')
                else:
                    self.real_B = F.upsample(self.real_B , size = ( int(700 * aspect_ratio),  700 ), mode = 'bilinear')

            # In case of mis-alignment
            self.A_seg = F.upsample(self.A_seg.unsqueeze(0).float() , size = (self.real_A.size(2),  self.real_A.size(3)),  mode = 'bilinear').squeeze(0).long()
            self.B_seg = F.upsample(self.B_seg.unsqueeze(0).float() , size = (self.real_B.size(2),  self.real_B.size(3)),  mode = 'bilinear').squeeze(0).long()

            self.real_A = self.real_A.float()
            self.real_B = self.real_B.float()


            ## HEN ##
            with torch.no_grad():
                hist_A_real_ab = self.getHistogram2d_np(self.real_A, self.hist_ab)
                hist_A_real_l =  self.getHistogram1d_np(self.real_A, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_A_real = torch.cat((hist_A_real_ab,hist_A_real_l),1)
    
                hist_B_real_ab =  self.getHistogram2d_np(self.real_B, self.hist_ab)
                hist_B_real_l  =  self.getHistogram1d_np(self.real_B, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_B_real = torch.cat((hist_B_real_ab,hist_B_real_l),1) 

                hist_A_feat  = self.HEN(hist_A_real)
                hist_B_feat  = self.HEN(hist_B_real)
            
                hist_A_feat_tile =    hist_A_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
                hist_B_feat_tile =    hist_B_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))
                self.final_result_A = hist_A_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))
                self.final_result_B = hist_B_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
                self.final_result_A_self = hist_A_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
                self.final_result_B_self = hist_B_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))


            if self.opt.is_SR:
                for i in range(0, self.input_SegNum):
                    seg_num_A = torch.sum(torch.sum(self.A_seg == i ,1),1)
                    seg_num_B = torch.sum(torch.sum(self.B_seg == i ,1),1)
                    if (seg_num_A > 0) and (seg_num_B > 0):
                        self.segmentwise_tile( self.real_A, self.A_seg, self.B_seg, self.final_result_A, i)
                        self.segmentwise_tile( self.real_B, self.B_seg, self.A_seg, self.final_result_B, i)
                        self.segmentwise_tile( self.real_A, self.A_seg, self.A_seg, self.final_result_A_self, i)
                        self.segmentwise_tile( self.real_B, self.B_seg, self.B_seg, self.final_result_B_self, i)

            # Padding
            p = 30
            reppad = nn.ReplicationPad2d(p)

            pad_real_A          = reppad(self.real_A)
            pad_real_B          = reppad(self.real_B)
            pad_hist_A_feat_tile= reppad(hist_A_feat_tile)
            pad_hist_B_feat_tile= reppad(hist_B_feat_tile)
            pad_final_result_A  = reppad(self.final_result_A)
            pad_final_result_B  = reppad(self.final_result_B)
            pad_final_result_A_self  = reppad(self.final_result_A_self)
            pad_final_result_B_self  = reppad(self.final_result_B_self)

            # Network Fake
            if self.opt.is_SR:
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.IRN(pad_real_A, pad_hist_A_feat_tile, pad_final_result_B)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.IRN(pad_real_B, pad_hist_B_feat_tile, pad_final_result_A)
            else:
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.IRN(pad_real_A, pad_hist_A_feat_tile, pad_hist_B_feat_tile)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.IRN(pad_real_B, pad_hist_B_feat_tile, pad_hist_A_feat_tile)
    
            ## IRN ## 
            _, _, _, _, fake_A_idt = self.IRN(pad_real_A, pad_hist_A_feat_tile, pad_hist_A_feat_tile)
            _, _, _, _, fake_B_idt = self.IRN(pad_real_B, pad_hist_B_feat_tile, pad_hist_B_feat_tile)
    
            # Crop the pad
            fake_A     =      fake_A[:,:,p:(p+self.real_B.size(2)),p:(p+self.real_B.size(3))]
            fake_B     =      fake_B[:,:,p:(p+self.real_A.size(2)),p:(p+self.real_A.size(3))]
            fake_A_idt =  fake_A_idt[:,:,p:(p+self.real_A.size(2)),p:(p+self.real_A.size(3))]
            fake_B_idt =  fake_B_idt[:,:,p:(p+self.real_B.size(2)),p:(p+self.real_B.size(3))]

            # Get Histogram
            bin_size = self.opt.ab_bin
            hist_A_fake_ab = self.getHistogram2d_np(fake_A, bin_size)
            hist_B_fake_ab = self.getHistogram2d_np(fake_B, bin_size)
            hist_A_real_ab = self.getHistogram2d_np(self.real_A, bin_size)
            hist_B_real_ab = self.getHistogram2d_np(self.real_B, bin_size)

            self.real_A = self.real_A
            self.real_B = self.real_B
            self.fake_A = fake_A
            self.fake_B = fake_B
            self.fake_A_idt = fake_A_idt
            self.fake_B_idt = fake_B_idt
    
            ups = nn.Upsample(scale_factor = 4 , mode = 'bilinear')
            lab_spectrum = cv2.cvtColor(cv2.imread('./ab_space_low.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            lab_spectrum= cv2.cvtColor(lab_spectrum, cv2.COLOR_RGB2LAB)
            lab_spectrum= cv2.cvtColor(lab_spectrum.astype(np.uint8), cv2.COLOR_LAB2RGB)
            lab_spectrum = cv2.resize(lab_spectrum, (bin_size, bin_size)).astype(np.float32) / 255.

            hist_A_real_ab = hist_A_real_ab.detach().cpu().numpy()[0].transpose(1, 2, 0)
            hist_A_real_ab = hist_A_real_ab * 1.1
            hist_A_real_ab = hist_A_real_ab / hist_A_real_ab.max()
            hist_A_real_ab = np.repeat(hist_A_real_ab, 3, axis = 2)
            hist_A_real_ab = hist_A_real_ab * lab_spectrum
            hist_A_real_ab = np.expand_dims(hist_A_real_ab, axis = 0)
            hist_A_real_ab = hist_A_real_ab.transpose(0, 3, 1, 2)
            self.real_A_hist = ups(torch.FloatTensor(hist_A_real_ab).cuda()) * 2 - 1

            hist_B_real_ab = hist_B_real_ab.detach().cpu().numpy()[0].transpose(1, 2, 0)
            hist_B_real_ab = hist_B_real_ab * 1.1
            hist_B_real_ab = hist_B_real_ab / hist_B_real_ab.max()
            hist_B_real_ab = np.repeat(hist_B_real_ab, 3, axis = 2)
            hist_B_real_ab = hist_B_real_ab * lab_spectrum
            hist_B_real_ab = np.expand_dims(hist_B_real_ab, axis = 0)
            hist_B_real_ab = hist_B_real_ab.transpose(0, 3, 1, 2)
            self.real_B_hist = ups(torch.FloatTensor(hist_B_real_ab).cuda()) * 2 - 1

            hist_B_fake_ab = hist_B_fake_ab.detach().cpu().numpy()[0].transpose(1, 2, 0)
            hist_B_fake_ab = hist_B_fake_ab * 1.1
            hist_B_fake_ab = hist_B_fake_ab / hist_B_fake_ab.max()
            hist_B_fake_ab = np.repeat(hist_B_fake_ab, 3, axis = 2)
            hist_B_fake_ab = hist_B_fake_ab * lab_spectrum
            hist_B_fake_ab = np.expand_dims(hist_B_fake_ab, axis = 0)
            hist_B_fake_ab = hist_B_fake_ab.transpose(0, 3, 1, 2)
            self.fake_B_hist = ups(torch.FloatTensor(hist_B_fake_ab).cuda()) * 2 - 1


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A,self.img_type)
        real_B = util.tensor2im(self.real_B,self.img_type)
        fake_A = util.tensor2im(self.fake_A,self.img_type)
        fake_B = util.tensor2im(self.fake_B,self.img_type)
        fake_A_idt = util.tensor2im(self.fake_A_idt,self.img_type)
        fake_B_idt = util.tensor2im(self.fake_B_idt,self.img_type)

        real_A_hist = util.tensor2im(self.real_A_hist,'rgb') 
        real_B_hist = util.tensor2im(self.real_B_hist,'rgb')
        fake_B_hist = util.tensor2im(self.fake_B_hist,'rgb')

        img_A_map = util.tensor2im(self.input_A_Map,'rgb')
        img_B_map = util.tensor2im(self.input_B_Map,'rgb')

        ret_visuals = OrderedDict([('01_input', real_A),
                                   ('02_target', real_B),
                                   ('03_fake_B', fake_B),
                                   ('04_A_seg', img_A_map),
                                   ('05_B_seg', img_B_map),
         ])

        return ret_visuals


    def getHistogram2d_np(self, img_torch, num_bin): # AB space # num_bin = self.hist_ab = 64
        arr = img_torch.detach().cpu().numpy()

        # Exclude Zeros and Make value 0 ~ 1
        arr1 = ( arr[0][1].ravel()[np.flatnonzero(arr[0][1])] + 1 ) /2 
        arr2 = ( arr[0][2].ravel()[np.flatnonzero(arr[0][2])] + 1 ) /2 


        if (arr1.shape[0] != arr2.shape[0]):
            if arr2.shape[0] < arr1.shape[0]:
                arr2 = np.concatenate([arr2, np.array([0])])
            else:
                arr1 = np.concatenate([arr1, np.array([0])])

        # AB space
        arr_new = [arr1, arr2]
        H,edges = np.histogramdd(arr_new, bins = [num_bin, num_bin], range = ((0,1),(0,1)))

        H = np.rot90(H)
        H = np.flip(H,0)

        H_torch = torch.from_numpy(H).float().cuda() #10/224/224
        H_torch = H_torch.unsqueeze(0).unsqueeze(0)

        # Normalize
        total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
        H_torch = H_torch / total_num

        return H_torch #1/1/64/64

    def getHistogram1d_np(self,img_torch, num_bin): # L space # Idon't know why but they(np, conv) are not exactly same
        # Preprocess
        arr = img_torch.detach().cpu().numpy()
        arr0 = ( arr[0][0].ravel()[np.flatnonzero(arr[0][0])] + 1 ) / 2 
        arr1 = np.zeros(arr0.size)

        arr_new = [arr0, arr1]
        H, edges = np.histogramdd(arr_new, bins = [num_bin, 1], range =((0,1),(-1,2)))
        H_torch = torch.from_numpy(H).float().cuda() #10/224/224
        H_torch = H_torch.unsqueeze(0).unsqueeze(0).permute(0,2,1,3)

        total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
        H_torch = H_torch / total_num

        return H_torch

    def segmentwise_tile(self, img, seg_src, seg_tgt, final_tensor,segment_num):
        
        # Mask only Specific Segmentation
        mask_seg = torch.mul( img , (seg_src == segment_num).cuda().float() )

        #Calc Each Histogram
        with torch.no_grad():
            hist_2d = self.getHistogram2d_np(mask_seg, self.hist_ab)
            hist_1d = self.getHistogram1d_np(mask_seg, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
            hist_cat   = torch.cat((hist_2d,hist_1d),1)

        #Encode Each Histogram Tensor
        hist_feat = self.HEN(hist_cat)

        #Embeded to Final Tensor
        final_tensor[:,:,seg_tgt.squeeze(0)==segment_num] = hist_feat.repeat(1,1, final_tensor[:,:,seg_tgt.squeeze(0)==segment_num].size(2), 1).squeeze(0).permute(2,0,1)

    def MakeLabelFromMap(self,input_A_Map, input_B_Map):
        label_A = self.LabelFromMap(input_A_Map)
        label_B = self.LabelFromMap(input_B_Map)

        label_AB2 = np.concatenate((label_A,label_B), axis = 0)
        label_AB2 = np.unique(label_AB2, axis = 0)
        label_AB  = torch.from_numpy(label_AB2)

        A_seg = torch.zeros(1,input_A_Map.size(2),input_A_Map.size(3))
        B_seg = torch.zeros(1,input_B_Map.size(2),input_B_Map.size(3))

        for i in range(0,label_AB.size(0)):            
            A_seg[ (input_A_Map.squeeze(0) == label_AB[i].unsqueeze(0).unsqueeze(0).permute(2,0,1))[0:1,:,:] ] = i
            B_seg[ (input_B_Map.squeeze(0) == label_AB[i].unsqueeze(0).unsqueeze(0).permute(2,0,1))[0:1,:,:] ] = i

        A_seg = A_seg.cuda().float()
        B_seg = B_seg.cuda().float()

        return A_seg, B_seg, label_AB.size(0)

    def LabelFromMap(self, tensor_map):
        np1 = tensor_map.squeeze(0).detach().cpu().numpy()
        np2 = np.transpose(np1, (1,2,0))
        np3 = np.reshape(np2, (np.shape(np2)[0] * np.shape(np2)[1], 3))
        np4 = np.unique(np3, axis= 0)

        return np4    
