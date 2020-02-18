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

class Network(torch.nn.Module):
    def __init__(self, opt, gpu_ids):
        self.gpu_ids = gpu_ids
        super(Network, self).__init__()
        self.opt = opt

        nb = opt.batchSize
        size = opt.fineSize
        self.hist_l = 8    # Number of bin L originally 10
        self.hist_ab = opt.ab_bin # Number of bin AB

        if opt.ab_bin != 64:
            opt.network_H = 'basic_{}'.format(opt.ab_bin)  # Number of bin AB
        self.hist_enc = 64 # Number of Channel
        self.img_type = opt.img_type
        self.netG_A = networks.define_G(3, 3, opt.ngf, opt.network, opt.norm, not opt.no_dropout, opt.init_type, gpu_ids)
        self.netC_A = networks.define_C((self.hist_l+1), 64, opt.ngf, opt.network_H, opt.norm, not opt.no_dropout, opt.init_type, gpu_ids).cuda() # out_nc=32

    def set_input_test(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']
        input_A_Map=input['A_map']
        input_B_Map=input['B_map']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])

        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C

        self.input_A_Map = input_A_Map.cuda(self.gpu_ids[0])
        self.input_B_Map = input_B_Map.cuda(self.gpu_ids[0])
        #self.input_A_Seg, self.input_B_Seg, self.input_SegNum = self.MakeLabelFromMap(self.input_A_Map,self.input_B_Map)

    def forward(self, A, B, C, A_Map, B_Map):
        with torch.no_grad():
            input = {'A': A, 'B': B, 'C': C, 'A_map': A_Map, 'B_map': B_Map}
            self.set_input_test(input)

            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
            self.real_C = Variable(self.input_C)
            #self.A_seg = Variable(self.input_A_Seg)
            #self.B_seg = Variable(self.input_B_Seg)
            self.A_map = Variable(self.input_A_Map)
            self.B_map = Variable(self.input_B_Map)

            self.orig_real_A = self.real_A
            self.orig_real_B = self.real_B

            self.real_A = self.real_A.float()
            self.real_B = self.real_B.float()
            self.real_C = self.real_C.float()

            test_mode = 'semantic_replacement'
            # Original
            hist_A_real_ab = self.getHistogram2d_np(self.real_A, self.hist_ab)
            hist_A_real_l =  self.getHistogram1d_np(self.real_A, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
            hist_A_real = torch.cat((hist_A_real_ab,hist_A_real_l),1)

            hist_B_real_ab =  self.getHistogram2d_np(self.real_B, self.hist_ab)
            hist_B_real_l  =  self.getHistogram1d_np(self.real_B, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
            hist_B_real = torch.cat((hist_B_real_ab,hist_B_real_l),1)

            # Network C
            hist_A_feat  = self.netC_A(hist_A_real)
            hist_B_feat  = self.netC_A(hist_B_real)

            #Target
            hist_A_feat_tile =    hist_A_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
            hist_B_feat_tile =    hist_B_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))
            self.final_result_A = hist_A_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))
            self.final_result_B = hist_B_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
            self.final_result_A_self = hist_A_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
            self.final_result_B_self = hist_B_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))

            if test_mode == 'semantic_replacement' and self.opt.mode in ['gsrt', 'rsrt', 'rtrt']:
                for i in range(0, self.input_SegNum):
                    seg_num_A = torch.sum(torch.sum(self.A_seg == i ,1),1)
                    seg_num_B = torch.sum(torch.sum(self.B_seg == i ,1),1)
                    if (seg_num_A > 0) and (seg_num_B > 0):
                        self.segmentwise_tile_test( self.real_A, self.A_seg, self.B_seg, self.final_result_A, i)
                        self.segmentwise_tile_test( self.real_B, self.B_seg, self.A_seg, self.final_result_B, i)
                        self.segmentwise_tile_test( self.real_A, self.A_seg, self.A_seg, self.final_result_A_self, i)
                        self.segmentwise_tile_test( self.real_B, self.B_seg, self.B_seg, self.final_result_B_self, i)

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
            if self.opt.mode == 'gsrt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(pad_real_A, pad_hist_A_feat_tile, pad_final_result_B)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(pad_real_B, pad_hist_B_feat_tile, pad_final_result_A)
            elif self.opt.mode == 'gsgt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(pad_real_A, pad_hist_A_feat_tile, pad_hist_B_feat_tile)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(pad_real_B, pad_hist_B_feat_tile, pad_hist_A_feat_tile)
            elif self.opt.mode == 'rsrt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(pad_real_A, pad_final_result_A_self, pad_final_result_B)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(pad_real_B, pad_final_result_B_self, pad_final_result_A)
            elif self.opt.mode == 'gtgt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(pad_real_A, pad_hist_B_feat_tile, pad_hist_B_feat_tile)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(pad_real_B, pad_hist_A_feat_tile, pad_hist_A_feat_tile)
            elif self.opt.mode == 'rtrt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(pad_real_A, pad_final_result_B_self, pad_final_result_B)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(pad_real_B, pad_final_result_A_self, pad_final_result_A)

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
            print("Histogram Size Not Match!")

        # AB space
        arr_new = [arr1, arr2]
        print(arr1.shape)
        print(arr2.shape)
        print(num_bin)
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
        # print("np 1d", H.shape)

        H_torch = torch.from_numpy(H).float().cuda() #10/224/224
        H_torch = H_torch.unsqueeze(0).unsqueeze(0).permute(0,2,1,3)
        # Normalize

        total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
        H_torch = H_torch / total_num
        # print('1d_hist: ', H_torch.size())

        return H_torch

    def segmentwise_tile_test(self, img, seg_src, seg_tgt, final_tensor,segment_num):

        # Mask only Specific Segmentation
        # print('img: ', img.size())
        # print('seg_src: ', seg_src.size())
        # print('seg_tgt: ', seg_tgt.size())
        # print('final_tensor: ', final_tensor.size())
        mask_seg = torch.mul( img , (seg_src == segment_num).cuda().float() )
        # print('mask_seg: ', mask_seg.size())

        #Calc Each Histogram
        with torch.no_grad():
            hist_2d = self.getHistogram2d_np(mask_seg, self.hist_ab)
            hist_1d = self.getHistogram1d_np(mask_seg, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
            hist_cat   = torch.cat((hist_2d,hist_1d),1)
        # print('hist_cat: ', hist_cat.size())

        #Encode Each Histogram Tensor
        hist_feat = self.netC_A(hist_cat)
        # print('hist_feat: ', hist_feat.size())

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
            abi = label_AB[i].cuda(self.gpu_ids[0])
            A_seg[ (input_A_Map.squeeze(0) == abi.unsqueeze(0).unsqueeze(0).permute(2,0,1))[0:1,:,:] ] = i
            B_seg[ (input_B_Map.squeeze(0) == abi.unsqueeze(0).unsqueeze(0).permute(2,0,1))[0:1,:,:] ] = i

        A_seg = A_seg.cuda().float()
        B_seg = B_seg.cuda().float()

        return A_seg, B_seg, label_AB.size(0)

    def LabelFromMap(self, tensor_map):
        np1 = tensor_map.squeeze(0).detach().cpu().numpy()
        np2 = np.transpose(np1, (1,2,0))
        np3 = np.reshape(np2, (np.shape(np2)[0] * np.shape(np2)[1], 3))
        np4 = np.unique(np3, axis= 0)

        return np4

class ColorHistogram_Model(BaseModel):
    def name(self):
        return 'ColorHistogram_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        #self.netG_A = networks.define_G(3, 3, opt.ngf, opt.network, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        #self.netC_A = networks.define_C((self.hist_l+1), 64, opt.ngf, opt.network_H, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids).cuda() # out_nc=32
        self.network = Network(opt, self.gpu_ids)

    def forward(self, input):
        input = {'A': input_A, 'B': input_B, 'C': input_C, 'A_map': input_A_Map, 'B_map': input_B_Map}
        out = self.network(input)
