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
        self.pad = 30
        self.reppad = nn.ReplicationPad2d(self.pad)

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
        self.inp = self.input_A
        self.tar = self.input_B
        self.A_seg = self.input_A_Seg
        self.B_seg = self.input_B_Seg

    def test(self):
        with torch.no_grad():

            self.inp = Variable(self.input_A)
            self.tar = Variable(self.input_B)
    
            self.A_seg = Variable(self.input_A_Seg)
            self.B_seg = Variable(self.input_B_Seg)
            self.A_map = Variable(self.input_A_Map)

            # max 700 px with aspect ratio
            if (self.inp.size(2)>700) or (self.inp.size(3)>700):
                aspect_ratio = self.inp.size(2) / self.inp.size(3)
                if (self.inp.size(2) > self.inp.size(3)):
                    self.inp = F.upsample(self.inp , size = ( 700 ,  int(700 / aspect_ratio)), mode = 'bilinear')
                else:
                    self.inp = F.upsample(self.inp , size = ( int(700 * aspect_ratio),  700 ), mode = 'bilinear')
            # max 700 px with aspect ratio
            if (self.tar.size(2)>700) or (self.tar.size(3)>700):
                aspect_ratio = self.tar.size(2) / self.tar.size(3)
                if (self.tar.size(2) > self.tar.size(3)):
                    self.tar = F.upsample(self.tar , size = ( 700 ,  int(700 / aspect_ratio)), mode = 'bilinear')
                else:
                    self.tar = F.upsample(self.tar , size = ( int(700 * aspect_ratio),  700 ), mode = 'bilinear')

            # In case of mis-alignment
            self.A_seg = F.upsample(self.A_seg.unsqueeze(0).float() , size = (self.inp.size(2),  self.inp.size(3)),  mode = 'bilinear').squeeze(0).long()
            self.B_seg = F.upsample(self.B_seg.unsqueeze(0).float() , size = (self.tar.size(2),  self.tar.size(3)),  mode = 'bilinear').squeeze(0).long()

            self.inp = self.inp.float()
            self.tar = self.tar.float()


            ## HEN ##
            with torch.no_grad():
                hist_inp_ab = self.getHistogram2d_np(self.inp, self.hist_ab)
                hist_inp_l =  self.getHistogram1d_np(self.inp, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_inp = torch.cat((hist_inp_ab,hist_inp_l),1)
    
                hist_tar_ab =  self.getHistogram2d_np(self.tar, self.hist_ab)
                hist_tar_l  =  self.getHistogram1d_np(self.tar, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_tar = torch.cat((hist_tar_ab,hist_tar_l),1) 

                hist_inp_feat  = self.HEN(hist_inp)
                hist_tar_feat  = self.HEN(hist_tar)
            
                hist_inp_feat_tile =    hist_inp_feat.repeat(1,1,self.inp.size(2),self.inp.size(3))
                hist_tar_feat_tile =    hist_tar_feat.repeat(1,1,self.tar.size(2),self.tar.size(3))
                self.final_result_tar = hist_tar_feat.repeat(1,1,self.inp.size(2),self.inp.size(3))


            if self.opt.is_SR:
                for i in range(0, self.input_SegNum):
                    seg_num_A = torch.sum(torch.sum(self.A_seg == i ,1),1)
                    seg_num_B = torch.sum(torch.sum(self.B_seg == i ,1),1)
                    if (seg_num_A > 0) and (seg_num_B > 0):
                        self.segmentwise_tile( self.tar, self.B_seg, self.A_seg, self.final_result_tar, i)

            # Padding
            self.inp   = self.reppad(self.inp)
            self.inp_H = self.reppad(hist_inp_feat_tile)
            self.tar_H = self.reppad(hist_tar_feat_tile)

            # Network Fake
            if self.opt.is_SR:
                self.tar_H_SR  = self.reppad(self.final_result_tar)
                _, _, _, _, out = self.IRN(self.inp, self.inp_H , self.tar_H_SR)
            else:
                _, _, _, _, out = self.IRN(self.inp, self.inp_H , self.tar_H )

            self.inp = self.inp[:,:,self.pad:(self.inp.size(2)-2*self.pad),self.pad:(self.inp.size(3)-2*self.pad)]
            self.tar = self.tar[:,:,self.pad:(self.tar.size(2)-2*self.pad),self.pad:(self.tar.size(3)-2*self.pad)]
            self.out = out[:,:,self.pad:(out.size(2)-2*self.pad),self.pad:(out.size(3)-2*self.pad)]


    def get_current_visuals(self):
        ret_visuals = OrderedDict([('01_input', util.tensor2im(self.inp,self.img_type)),
                                   ('02_target', util.tensor2im(self.tar,self.img_type)),
                                   ('03_output', util.tensor2im(self.out,self.img_type))])

        return ret_visuals

###########################################################################################################################
    def getHistogram2d_np(self, img_torch, num_bin):
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

        return H_torch

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
