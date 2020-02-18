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

        self.hist_l = 8    # Number of bin L originally 10
        self.hist_ab = opt.ab_bin # Number of bin AB
        if opt.ab_bin != 64:
            opt.network_H = 'basic_{}'.format(opt.ab_bin)  # Number of bin AB
        self.hist_enc = 64 # Number of Channel
        self.img_type = opt.img_type


        print(opt.no_dropout)
        self.netG_A = networks.define_G(3, 3, opt.ngf, opt.network, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        # self.netG_A = networks.define_G(3, 3, opt.ngf, 'cvpr', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netC_A = networks.define_C((self.hist_l+1), 64, opt.ngf, opt.network_H, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids).cuda() # out_nc=32

        self.netHist_64 = networks.define_Hist(self.hist_ab)
        self.netHist_32 =networks.define_Hist(32)
        self.netHist_8 = networks.define_Hist(self.hist_l)

        self.netHist_64.cuda()
        self.netHist_64.eval()
        self.netHist_32.cuda()
        self.netHist_32.eval()
        self.netHist_8.cuda()
        self.netHist_8.eval()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if opt.network_H != 'image':
                self.load_network(self.netC_A, 'C_A', which_epoch)
            print('***********LOADED*************')


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                                 self.netG_A.parameters(),
                                                 self.netC_A.parameters(),
                                                 ),lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netHist_64)
        networks.print_network(self.netC_A)

        print('-----------------------------------------------')
        print(opt.mode)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']
        input_A_Map=input['A_map']
        input_B_Map=input['A_map']
        pair = input['pair']

        if len(self.gpu_ids) > 0:
            # input_A = input_A.cuda(self.gpu_ids[0], async=True)
            # input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C

        #self.input_A_Seg = self.read_t7(input_A_Seg)
        #self.input_B_Seg = self.read_t7(input_B_Seg)
        self.input_A_Map = input_A_Map
        self.input_B_Map = input_B_Map
        self.input_A_Seg, self.input_B_Seg, self.input_SegNum = self.MakeLabelFromMap(self.input_A_Map,self.input_B_Map)

        self.pair = pair

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_test(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']
        input_A_Map=input['A_map']
        input_B_Map=input['B_map']
        pair = input['pair']

        print('***************')
        print(input_A.size())
        print(input_B.size())
        print(input_C.size())
        print(input_A_Map.size())
        print(input_B_Map.size())
        print('***************')

        if len(self.gpu_ids) > 0:
            # input_A = input_A.cuda(self.gpu_ids[0], async=True)
            # input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C

        #self.input_A_Seg = self.read_t7(input_A_Seg)
        #self.input_B_Seg = self.read_t7(input_B_Seg)
        self.input_A_Map = input_A_Map
        self.input_B_Map = input_B_Map
        self.input_A_Seg, self.input_B_Seg, self.input_SegNum = self.MakeLabelFromMap(self.input_A_Map,self.input_B_Map)

        self.pair = pair

        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def forward(self):
        self.real_A = self.input_A
        self.real_B = self.input_B
        self.real_C = self.input_C
        self.A_seg = self.input_A_Seg
        self.B_seg = self.input_B_Seg
        self.A_map = self.input_A_Map
        self.B_map = self.input_B_Map

        # print ("[ColorHistogram] End-to-end color histogram")
        # print('')

    def backward_all(self):

        self.real_A = self.real_A.float() 
        self.real_B = self.real_B.float()
        
        if self.opt.network_H != 'image':
            with torch.no_grad():
                # Original
                hist_A_real_ab = self.getHistogram2d_np(self.real_A, self.hist_ab)
                hist_A_real_l =  self.getHistogram1d_np(self.real_A, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_A_real = torch.cat((hist_A_real_ab,hist_A_real_l),1)

                hist_B_real_ab =  self.getHistogram2d_np(self.real_B, self.hist_ab)
                hist_B_real_l  =  self.getHistogram1d_np(self.real_B, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_B_real = torch.cat((hist_B_real_ab,hist_B_real_l),1) 

        elif self.opt.network_H == 'image':
            # hist_A_real = F.upsample(self.real_A, size = (64, 64), mode = 'bilinear')
            # hist_B_real = F.upsample(self.real_B, size = (64, 64), mode = 'bilinear')
            hist_A_real = self.real_A
            hist_B_real = self.real_B

        # Network C
        hist_A_feat  = self.netC_A(hist_A_real)
        hist_B_feat  = self.netC_A(hist_B_real)
        
        #Target 
        hist_A_feat_tile = hist_A_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
        hist_B_feat_tile = hist_B_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))


        # Switch Tensor

        # Network Fake
        #if random.random() > 0.5:
        #iccv
        if self.opt.mode in ['gsrt', 'rsrt', 'rtrt']:
            self.final_result_A = hist_A_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))
            self.final_result_B = hist_B_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
            self.final_result_A_self = hist_A_feat.repeat(1,1,self.real_B.size(2),self.real_B.size(3))
            self.final_result_B_self = hist_B_feat.repeat(1,1,self.real_A.size(2),self.real_A.size(3))
            for i in range(0,8):
                seg_num_A = torch.sum(torch.sum(self.A_seg == i ,1),1)
                seg_num_B = torch.sum(torch.sum(self.B_seg == i ,1),1)
                if (seg_num_A > 0) and (seg_num_B > 0):
                    self.segmentwise_tile_test( self.real_A, self.A_seg, self.B_seg, self.final_result_A , i)
                    self.segmentwise_tile_test( self.real_B, self.B_seg, self.A_seg, self.final_result_B , i) # Seg_A == Seg_B
                    self.segmentwise_tile_test( self.real_A, self.A_seg, self.A_seg, self.final_result_A_self, i)
                    self.segmentwise_tile_test( self.real_B, self.B_seg, self.B_seg, self.final_result_B_self, i) # Seg_A == Seg_B

            if self.opt.mode == 'gsrt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(self.real_A,hist_A_feat_tile,self.final_result_B)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(self.real_B,hist_B_feat_tile,self.final_result_A)
            elif self.opt.mode == 'rsrt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(self.real_A,self.final_result_A_self,self.final_result_B)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(self.real_B,self.final_result_B_self,self.final_result_A)
            elif self.opt.mode == 'rtrt':
                fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(self.real_A,self.final_result_B_self,self.final_result_B_self)
                fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(self.real_B,self.final_result_A_self,self.final_result_A_self)

        elif self.opt.mode == 'gtgt':
            fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(self.real_A,hist_B_feat_tile,hist_B_feat_tile)
            fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(self.real_B,hist_A_feat_tile,hist_A_feat_tile)
        elif self.opt.mode == 'gsgt':
            fake_B1, fake_B2, fake_B3, fake_B4, fake_B = self.netG_A(self.real_A,hist_A_feat_tile,hist_B_feat_tile)
            fake_A1, fake_A2, fake_A3, fake_A4, fake_A = self.netG_A(self.real_B,hist_B_feat_tile,hist_A_feat_tile)


        # Netowkr Idt
        _, _, _, _, fake_A_idt = self.netG_A(self.real_A,hist_A_feat_tile, hist_A_feat_tile)
        _, _, _, _, fake_B_idt = self.netG_A(self.real_B,hist_B_feat_tile, hist_B_feat_tile)

        # Visualize
        #with torch.no_grad():

        hist_A_real_ab_sps = self.getHistogram2d_conv(self.real_A, self.netHist_32)
        hist_A_fake_ab_sps = self.getHistogram2d_conv(     fake_A, self.netHist_32)
        hist_B_real_ab_sps = self.getHistogram2d_conv(self.real_B, self.netHist_32)
        hist_B_fake_ab_sps = self.getHistogram2d_conv(     fake_B, self.netHist_32)

        if self.pair == True:
            # Loss_IMG
            loss_sl =  self.criterionMSE(fake_B,self.real_B) * 1.0
            loss_sl2 = self.criterionMSE(fake_A,self.real_A) *  1.0

            loss_sl_B1 = self.criterionMSE(fake_B1, F.upsample(self.real_B, size = (fake_B1.size(2), fake_B1.size(3)), mode = 'bilinear') ) * 0.0
            loss_sl_A1 = self.criterionMSE(fake_A1, F.upsample(self.real_A, size = (fake_A1.size(2), fake_A1.size(3)), mode = 'bilinear') ) * 0.0 # 'fake_A' may be equal to 'fake_A1'

            loss_sl_B2 = self.criterionMSE(fake_B2, F.upsample(self.real_B, size = (fake_B2.size(2), fake_B2.size(3)), mode = 'bilinear') ) * 0.5
            loss_sl_A2 = self.criterionMSE(fake_A2, F.upsample(self.real_A, size = (fake_A2.size(2), fake_A2.size(3)), mode = 'bilinear') ) * 0.5

            loss_sl_B3 = self.criterionMSE(fake_B3, F.upsample(self.real_B, size = (fake_B3.size(2), fake_B3.size(3)), mode = 'bilinear') ) * 0.5
            loss_sl_A3 = self.criterionMSE(fake_A3, F.upsample(self.real_A, size = (fake_A3.size(2), fake_A3.size(3)), mode = 'bilinear') ) * 0.5

            loss_sl_B4 = self.criterionMSE(fake_B4, F.upsample(self.real_B, size = (fake_B4.size(2), fake_B4.size(3)), mode = 'bilinear') ) * 0.5
            loss_sl_A4 = self.criterionMSE(fake_A4, F.upsample(self.real_A, size = (fake_A4.size(2), fake_A4.size(3)), mode = 'bilinear') ) * 0.5

            loss_sl_idt  = self.criterionMSE(fake_A_idt,self.real_A) * 1.0
            loss_sl_idt2 = self.criterionMSE(fake_B_idt,self.real_B) * 1.0

        # Loss_HIST
        loss_sl_hist = self.criterionMSE(hist_B_fake_ab_sps,hist_B_real_ab_sps.detach()) * 1.5
        loss_sl_hist2 = self.criterionMSE(hist_A_fake_ab_sps,hist_A_real_ab_sps.detach()) * 1.5

        # combined loss
        loss_G = ( 
                    loss_sl
                    + loss_sl2

                    + loss_sl_idt
                    + loss_sl_idt2
                    + loss_sl_hist
                    + loss_sl_hist2
                    + loss_sl_B1
                    + loss_sl_A1
                    + loss_sl_B2
                    + loss_sl_A2
                    + loss_sl_B3
                    + loss_sl_A3
                    + loss_sl_B4
                    + loss_sl_A4
                 )

        loss_G.backward()

        # Save Loss
        self.loss_sl = loss_sl.item()
        self.loss_sl2 = loss_sl2.item()
        self.loss_sl_idt = loss_sl_idt.item()
        self.loss_sl_idt2 = loss_sl_idt2.item()
        self.loss_sl_hist = loss_sl_hist.item()
        self.loss_sl_hist2 = loss_sl_hist2.item()

        self.loss_sl_B1 = loss_sl_B1.item()
        self.loss_sl_A1 = loss_sl_A1.item()
        self.loss_sl_B2 = loss_sl_B2.item()
        self.loss_sl_A2 = loss_sl_A2.item()
        self.loss_sl_B3 = loss_sl_B3.item()
        self.loss_sl_A3 = loss_sl_A3.item()
        self.loss_sl_B4 = loss_sl_B4.item()
        self.loss_sl_A4 = loss_sl_A4.item()

        # Save IMG
        self.fake_A = fake_A
        self.fake_B = fake_B        
        self.fake_B_idt = fake_B_idt
        self.fake_A_idt = fake_A_idt
        self.fake_B1 = fake_B1
        self.fake_A1 = fake_A1
        self.fake_B2 = fake_B2
        self.fake_A2 = fake_A2
        self.fake_B3 = fake_B3
        self.fake_A3 = fake_A3
        self.fake_B4 = fake_B4
        self.fake_A4 = fake_A4

        # self.fake_A = fake_A
        # self.fake_B = fake_B.detach().cpu().numpy()        
        # self.fake_B_idt = fake_B_idt.detach().cpu().numpy()
        # self.fake_A_idt = fake_A_idt.detach().cpu().numpy()
        # self.fake_B1 = fake_B1.detach().cpu().numpy()
        # self.fake_A1 = fake_A1.detach().cpu().numpy()
        # self.fake_B2 = fake_B2.detach().cpu().numpy()
        # self.fake_A2 = fake_A2.detach().cpu().numpy()
        # self.fake_B3 = fake_B3.detach().cpu().numpy()
        # self.fake_A3 = fake_A3.detach().cpu().numpy()
        # self.fake_B4 = fake_B4.detach().cpu().numpy()
        # self.fake_A4 = fake_A4.detach().cpu().numpy()

        # ups = nn.Upsample(scale_factor = 4 , mode = 'bilinear')
        # self.real_A_hist = ups( hist_A_real_ab )
        # self.real_B_hist = ups( hist_B_real_ab )
        # self.fake_A_hist = ups( hist_A_fake_ab_sps )
        # self.fake_B_hist = ups( hist_B_fake_ab_sps )

    def test(self):
        with torch.no_grad():

            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
            self.real_C = Variable(self.input_C)
    
            self.A_seg = Variable(self.input_A_Seg)
            self.B_seg = Variable(self.input_B_Seg)
            self.A_map = Variable(self.input_A_Map)
            self.B_map = Variable(self.input_B_Map)

            self.orig_real_A = self.real_A
            self.orig_real_B = self.real_B

            #self.real_A = self.GCT(self.real_A,self.real_B)
            #self.real_A = self.real_A.cuda().float()

            # If image is too big
            divisor = 1
            self.real_A = F.upsample(self.real_A , size = (self.real_A.size(2) * divisor,  self.real_A.size(3) * divisor), mode = 'bilinear')
            self.real_B = F.upsample(self.real_B , size = (self.real_B.size(2) * divisor,  self.real_B.size(3) * divisor), mode = 'bilinear')

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
            self.real_C = self.real_C.float()

            print('********')
            print('real_A: ', self.real_A.size())
            print('real_B: ', self.real_B.size())
            print('A_seg: ', self.A_seg.size())
            print('B_seg: ', self.B_seg.size())
            print('********')
    
    
            test_mode = 'semantic_replacement'
            # test_mode = 'stretch'
            # test_mode = 'L_equalize'
            print('TEST_MODE: ' + test_mode)
            # Histogram
            with torch.no_grad():
                # Original
                hist_A_real_ab = self.getHistogram2d_np(self.real_A, self.hist_ab)
                hist_A_real_l =  self.getHistogram1d_np(self.real_A, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_A_real = torch.cat((hist_A_real_ab,hist_A_real_l),1)
    
                hist_B_real_ab =  self.getHistogram2d_np(self.real_B, self.hist_ab)
                hist_B_real_l  =  self.getHistogram1d_np(self.real_B, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
                hist_B_real = torch.cat((hist_B_real_ab,hist_B_real_l),1) 

            # Stretch Histogram
            if test_mode == 'stretch':
                self.hist_ab = 160
                # dresize = 96
                dresize = 102
                print(dresize / self.hist_ab) # Resized Ratio  # 76 = * 1.2       # 82 = * 1.3        # 90 = * 1.4        # 96 = * 1.5        # 102 = * 1.6        # 108 = * 1.7        # 114 = * 1.8        # 122 = * 1.9        # 128 = * 2.0 # 160 = 2.5
                gap = int((dresize - self.hist_ab)/2)
                hist_A_real_np = hist_A_real.squeeze(0).permute(1,2,0).cpu().numpy()
                hist_A_real_np_resize = cv2.resize(hist_A_real_np, dsize = (dresize, dresize), interpolation = cv2.INTER_LINEAR)
                hist_A_real_np_resize_crop = hist_A_real_np_resize[gap:gap+self.hist_ab,gap:gap+self.hist_ab,:]
                hist_B_real_str = torch.from_numpy(hist_A_real_np_resize_crop).permute(2,0,1).unsqueeze(0).cuda()        
                # Normalization
                total_num = sum(sum(hist_B_real_str[:,0:1,:,:].squeeze(0).squeeze(0)))  # should be sum to 1
                hist_B_real_str[:,0:1,:,:] = hist_B_real_str[:,0:1,:,:] / total_num
                hist_B_real = hist_B_real_str 
                hist_B_real_ab = hist_B_real[:,0:1,:,:]

            if test_mode == 'L_equalize':
                self.hist_ab = 160
                # dresize = 96
                dresize = 102
                print(dresize / self.hist_ab) # Resized Ratio  # 76 = * 1.2       # 82 = * 1.3        # 90 = * 1.4        # 96 = * 1.5        # 102 = * 1.6        # 108 = * 1.7        # 114 = * 1.8        # 122 = * 1.9        # 128 = * 2.0 # 160 = 2.5
                gap = int((dresize - self.hist_ab)/2)
                hist_A_real_np = hist_A_real.squeeze(0).permute(1,2,0).cpu().numpy()
                hist_A_real_np_resize = cv2.resize(hist_A_real_np, dsize = (dresize, dresize), interpolation = cv2.INTER_LINEAR)
                hist_A_real_np_resize_crop = hist_A_real_np_resize[gap:gap+self.hist_ab,gap:gap+self.hist_ab,:]
                hist_B_real_str = torch.from_numpy(hist_A_real_np_resize_crop).permute(2,0,1).unsqueeze(0).cuda()        
                # Normalization
                total_num = sum(sum(hist_B_real_str[:,0:1,:,:].squeeze(0).squeeze(0)))  # should be sum to 1
                hist_B_real_str[:,0:1,:,:] = hist_B_real_str[:,0:1,:,:] / total_num
                hist_B_real = hist_B_real_str 
                hist_B_real_ab = hist_B_real[:,0:1,:,:]

            if self.opt.network_H == 'image':
                hist_A_real = F.upsample(self.real_A, size = (256, 256), mode = 'bilinear')
                hist_B_real = F.upsample(self.real_B, size = (256, 256), mode = 'bilinear')
                # hist_A_real = self.real_A
                # hist_B_real = self.real_B
    
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
    
            # Netowkr Idt
            _, _, _, _, fake_A_idt = self.netG_A(pad_real_A, pad_hist_A_feat_tile, pad_hist_A_feat_tile)
            _, _, _, _, fake_B_idt = self.netG_A(pad_real_B, pad_hist_B_feat_tile, pad_hist_B_feat_tile)
    
    
            # Crop the pad
            fake_A     =      fake_A[:,:,p:(p+self.real_B.size(2)),p:(p+self.real_B.size(3))]
            fake_B     =      fake_B[:,:,p:(p+self.real_A.size(2)),p:(p+self.real_A.size(3))]
            fake_A_idt =  fake_A_idt[:,:,p:(p+self.real_A.size(2)),p:(p+self.real_A.size(3))]
            fake_B_idt =  fake_B_idt[:,:,p:(p+self.real_B.size(2)),p:(p+self.real_B.size(3))]

            # Transfer only AB space
            #fake_A[:,0,:,:] = self.real_B[:,0,:,:]
            #fake_B[:,0,:,:] = self.real_A[:,0,:,:]

            # Get Histogram
            bin_size = self.opt.ab_bin
            hist_A_fake_ab = self.getHistogram2d_np(fake_A, bin_size)
            hist_B_fake_ab = self.getHistogram2d_np(fake_B, bin_size)
            hist_A_real_ab = self.getHistogram2d_np(self.real_A, bin_size)
            hist_B_real_ab = self.getHistogram2d_np(self.real_B, bin_size)


            #fake_Bgct = self.GCT(fake_B,self.real_B)
            
            #self.reinhard = self.GCT_segmentwise(self.real_A,self.real_B,self.A_seg,self.B_seg)
            self.real_A = self.real_A
            self.real_B = self.real_B
            self.fake_A = fake_A
            self.fake_B = fake_B
            self.fake_A_idt = fake_A_idt
            self.fake_B_idt = fake_B_idt
    
            ups = nn.Upsample(scale_factor = 4 , mode = 'bilinear')
            lab_spectrum = cv2.cvtColor(cv2.imread('./ab_space_low.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            lab_spectrum= cv2.cvtColor(lab_spectrum, cv2.COLOR_RGB2LAB)
            # lab_spectrum[:, :, 0] = np.clip((lab_spectrum[:, :, 0] - 40), 0, 100)
            lab_spectrum= cv2.cvtColor(lab_spectrum.astype(np.uint8), cv2.COLOR_LAB2RGB)
            # crop_size = 20
            # lab_spectrum= lab_spectrum[crop_size:215 - crop_size, crop_size:215 - crop_size, :]
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


            # cmap = plt.get_cmap('gray')
            # hist_A_real_ab = np.squeeze(hist_A_real_ab.detach().cpu().numpy()[0].transpose(1, 2, 0))
            # hist_A_real_ab = hist_A_real_ab / hist_A_real_ab.max()
            # hist_A_real_ab = cmap(hist_A_real_ab)
            # hist_A_real_ab = np.delete(hist_A_real_ab, 3, 2)
            # hist_A_real_ab = np.expand_dims(hist_A_real_ab, axis = 0)
            # hist_A_real_ab = hist_A_real_ab.transpose(0, 3, 1, 2)
            # self.real_A_hist = ups(torch.FloatTensor(hist_A_real_ab).cuda()) * 2 - 1

            # hist_B_real_ab = np.squeeze(hist_B_real_ab.detach().cpu().numpy()[0].transpose(1, 2, 0))
            # hist_B_real_ab = hist_B_real_ab / hist_B_real_ab.max()
            # hist_B_real_ab = cmap(hist_B_real_ab)
            # hist_B_real_ab = np.delete(hist_B_real_ab, 3, 2)
            # hist_B_real_ab = np.expand_dims(hist_B_real_ab, axis = 0)
            # hist_B_real_ab = hist_B_real_ab.transpose(0, 3, 1, 2)
            # self.real_B_hist = ups(torch.FloatTensor(hist_B_real_ab).cuda()) * 2 - 1

            # hist_B_fake_ab = np.squeeze(hist_B_fake_ab.detach().cpu().numpy()[0].transpose(1, 2, 0))
            # hist_B_fake_ab = hist_B_fake_ab / hist_B_fake_ab.max()
            # hist_B_fake_ab = cmap(hist_B_fake_ab)
            # hist_B_fake_ab = np.delete(hist_B_fake_ab, 3, 2)
            # hist_B_fake_ab = np.expand_dims(hist_B_fake_ab, axis = 0)
            # hist_B_fake_ab = hist_B_fake_ab.transpose(0, 3, 1, 2)
            # self.fake_B_hist = ups(torch.FloatTensor(hist_B_fake_ab).cuda()) * 2 - 1



            # self.real_B_hist = torch.FloatTensor(np.expand_dims(sns.heatmap(ups(hist_B_real_ab.detach().cpu().numpy()[0].transpose(1, 2, 0))).transpose(2, 0, 1), axis = 0)).cuda()
            # self.fake_B_hist = torch.FloatTensor(np.expand_dims(sns.heatmap(ups(hist_B_fakd_ab.detach().cpu().numpy()[0].transpose(1, 2, 0))).transpose(2, 0, 1), axis = 0)).cuda()

            # self.real_B_hist = sns.heatmap(ups(hist_B_real_ab))
            # # self.fake_A_hist = ups(hist_A_fake_ab) * 10
            # self.fake_B_hist = sns.heatmap(ups(hist_B_fake_ab))

            self.fake_B1 = fake_B1
            self.fake_A1 = fake_A1
            self.fake_B2 = fake_B2
            self.fake_A2 = fake_A2
            self.fake_B3 = fake_B3
            self.fake_A3 = fake_A3
            self.fake_B4 = fake_B4
            self.fake_A4 = fake_A4
            #self.reinhard = self.GCT(self.real_A,self.real_B)
            
            self.reinhard  = self.real_A #self.GCT_segmentwise(self.real_A,self.real_B,self.A_seg,self.B_seg)
            self.reinhard2 = self.real_A #self.GCT_segmentwise(self.real_B,self.real_A,self.B_seg,self.A_seg)
            self.fake_Bgct = self.real_A

            # print(fake_A.size())
            # print(self.real_A.size())
            # print(fake_B.size())
            # print(self.real_B.size())

            if self.opt.is_psnr:
                loss_sl =  torch.nn.MSELoss()(fake_B,self.real_B)
                loss_sl2 = torch.nn.MSELoss()(fake_A,self.real_A)
            else:
                loss_sl = 0.
                loss_sl2 = 0.
            return loss_sl, loss_sl2


    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_all()
        self.optimizer_G.step()


    def get_current_errors(self): 
        ret_errors = OrderedDict([
            ('sl', self.loss_sl), 
            ('sl2', self.loss_sl2), 
            ('sl_idt', self.loss_sl_idt), 
            ('sl_idt2', self.loss_sl_idt2), 
            ('sl_sl_hist', self.loss_sl_hist),
            ('sl_sl_hist2', self.loss_sl_hist2),


            ('loss_sl_B1',self.loss_sl_B1),
            ('loss_sl_A1',self.loss_sl_A1),
            ('loss_sl_B2',self.loss_sl_B2),
            ('loss_sl_A2',self.loss_sl_A2),
            ('loss_sl_B3',self.loss_sl_B3),
            ('loss_sl_A3',self.loss_sl_A3),
            ('loss_sl_B4',self.loss_sl_B4),
            ('loss_sl_A4',self.loss_sl_A4),

            ])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A,self.img_type)
        real_B = util.tensor2im(self.real_B,self.img_type)
        fake_A = util.tensor2im(self.fake_A,self.img_type)
        fake_B = util.tensor2im(self.fake_B,self.img_type)
        fake_A_idt = util.tensor2im(self.fake_A_idt,self.img_type)
        fake_B_idt = util.tensor2im(self.fake_B_idt,self.img_type)

        # real_A_hist = util.tensor2im(self.real_A_hist,'lab') 
        # real_B_hist = util.tensor2im(self.real_B_hist,'lab')
        # fake_A_hist = util.tensor2im(self.fake_A_hist,'lab')
        # fake_B_hist = util.tensor2im(self.fake_B_hist,'lab')

        img_A_map = util.tensor2im(self.input_A_Map,'rgb')
        img_B_map = util.tensor2im(self.input_B_Map,'rgb')

        # fake_B1 = util.tensor2im(self.fake_B1,self.img_type)
        # fake_A1 = util.tensor2im(self.fake_A1,self.img_type)
        # fake_B2 = util.tensor2im(self.fake_B2,self.img_type)
        # fake_A2 = util.tensor2im(self.fake_A2,self.img_type)
        # fake_B3 = util.tensor2im(self.fake_B3,self.img_type)
        # fake_A3 = util.tensor2im(self.fake_A3,self.img_type)
        # fake_B4 = util.tensor2im(self.fake_B4,self.img_type)
        # fake_A4 = util.tensor2im(self.fake_A4,self.img_type)

        # reinhard = util.tensor2im(self.reinhard,self.img_type)
        # reinhard2 = util.tensor2im(self.reinhard2,self.img_type)
        # fake_Bgct =util.tensor2im(self.fake_Bgct,self.img_type)


        ret_visuals = OrderedDict([('1.real_A', real_A),
                                   ('2.fake_A', fake_A),
                                   ('3.fake_A_idt', fake_A_idt),
                                   ('4.real_B', real_B),
                                   ('5.fake_B', fake_B),
                                   ('6.fake_B_idt', fake_B_idt),
                                   #('fake_Bgct', fake_Bgct),
                                   #('reinhard', reinhard),
                                   #('reinhard2', reinhard2),

                                   # ('real_A_hist', real_A_hist),
                                   # ('real_B_hist', real_B_hist),                                   
                                   # ('fake_A_hist', fake_A_hist),
                                   # ('fake_B_hist', fake_B_hist),


                                   # ('fake_B1', fake_B1),
                                   # ('fake_A1', fake_A1),
                                   # ('fake_B2', fake_B2),
                                   # ('fake_A2', fake_A2),
                                   # ('fake_B3', fake_B3),
                                   # ('fake_A3', fake_A3),
                                   # ('fake_B4', fake_B4),
                                   # ('fake_A4', fake_A4),


                                   ('7.Map_A', img_A_map),
                                   ('8.Map_B', img_B_map),

         ])
        return ret_visuals

    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.real_A,self.img_type)
        real_B = util.tensor2im(self.real_B,self.img_type)
        fake_A = util.tensor2im(self.fake_A,self.img_type)
        fake_B = util.tensor2im(self.fake_B,self.img_type)
        fake_A_idt = util.tensor2im(self.fake_A_idt,self.img_type)
        fake_B_idt = util.tensor2im(self.fake_B_idt,self.img_type)

        real_A_hist = util.tensor2im(self.real_A_hist,'rgb') 
        real_B_hist = util.tensor2im(self.real_B_hist,'rgb')
        # fake_A_hist = util.tensor2im(self.fake_A_hist,'rgb')
        fake_B_hist = util.tensor2im(self.fake_B_hist,'rgb')

        img_A_map = util.tensor2im(self.input_A_Map,'rgb')
        img_B_map = util.tensor2im(self.input_B_Map,'rgb')

        # fake_B1 = util.tensor2im(self.fake_B1,self.img_type)
        # fake_A1 = util.tensor2im(self.fake_A1,self.img_type)
        # fake_B2 = util.tensor2im(self.fake_B2,self.img_type)
        # fake_A2 = util.tensor2im(self.fake_A2,self.img_type)
        # fake_B3 = util.tensor2im(self.fake_B3,self.img_type)
        # fake_A3 = util.tensor2im(self.fake_A3,self.img_type)
        # fake_B4 = util.tensor2im(self.fake_B4,self.img_type)
        # fake_A4 = util.tensor2im(self.fake_A4,self.img_type)

        # reinhard = util.tensor2im(self.reinhard,self.img_type)
        # reinhard2 = util.tensor2im(self.reinhard2,self.img_type)
        # fake_Bgct =util.tensor2im(self.fake_Bgct,self.img_type)


        ret_visuals = OrderedDict([('1_real_A', real_A),
                                   ('2_real_B', real_B),
                                   ('3_fake_B', fake_B),
                                   ('4_fake_A', fake_A),
                                   ('5_fake_A_idt', fake_A_idt),
                                   #('fake_Bgct', fake_Bgct),
                                   #('reinhard', reinhard),
                                   #('reinhard2', reinhard2),
                                   ('6_fake_B_idt', fake_B_idt),

                                   ('real_A_hist', real_A_hist),
                                   ('real_B_hist', real_B_hist),                                   
                                   # ('fake_A_hist', fake_A_hist),
                                   ('fake_B_hist', fake_B_hist),

                                   # ('fake_B1', fake_B1),
                                   # ('fake_A1', fake_A1),
                                   # ('fake_B2', fake_B2),
                                   # ('fake_A2', fake_A2),
                                   # ('fake_B3', fake_B3),
                                   # ('fake_A3', fake_A3),
                                   # ('fake_B4', fake_B4),
                                   # ('fake_A4', fake_A4),


                                   ('7_Map_A', img_A_map),
                                   ('8_Map_B', img_B_map),

         ])
        return ret_visuals



    # def get_current_visuals_test(self):
    #     real_A = util.tensor2im(self.real_A,self.img_type)
    #     real_B = util.tensor2im(self.real_B,self.img_type)
    #     fake_A = util.tensor2im(self.fake_A,self.img_type)
    #     fake_B = util.tensor2im(self.fake_B,self.img_type)
    #     fake_A_idt = util.tensor2im(self.fake_A_idt,self.img_type)
    #     fake_B_idt = util.tensor2im(self.fake_B_idt,self.img_type)

    #     real_A_hist = util.tensor2im(self.real_A_hist,self.img_type)
    #     real_B_hist = util.tensor2im(self.real_B_hist,self.img_type)
    #     fake_A_hist = util.tensor2im(self.fake_A_hist,self.img_type)
    #     fake_B_hist = util.tensor2im(self.fake_B_hist,self.img_type)

    #     img_A_map = util.tensor2im(self.input_A_Map,self.img_type)
    #     img_B_map = util.tensor2im(self.input_B_Map,self.img_type)

    #     reinhard = util.tensor2im(self.reinhard,self.img_type)
        


    #     ret_visuals = OrderedDict([('real_A', real_A),
    #                                #('fake_A', fake_A),
    #                                ('real_B', real_B),
    #                                ('fake_B', fake_B),
    #                                # ('reinhard', reinhard),

    #                                #('real_A_hist', real_A_hist),
    #                                #('real_B_hist', real_B_hist),                                   
    #                                #('fake_A_hist', fake_A_hist),
    #                                #('fake_B_hist', fake_B_hist),


    #                                #('Map_A', img_A_map),
    #                                #('Map_B', img_B_map)
    #      ])
    #     return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netC_A, 'C_A', label, self.gpu_ids)




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


    def getHistogram2d_conv(self,tensor, netConv): # AB space
        # Preprocess
        hist_a = netConv((tensor[:,1,:,:].unsqueeze(0)+1)/2)
        hist_b = netConv((tensor[:,2,:,:].unsqueeze(0)+1)/2) 

        # Network
        BIN = netConv.getBin()
        tensor1 = hist_a.repeat(1,BIN,1,1)
        tensor2 = hist_b.repeat(1,1,BIN,1).view(1,BIN*BIN,256,256)

        pool = nn.AvgPool2d(256)
        hist2d = pool(tensor1*tensor2)
        hist2d = hist2d.view(1,1,BIN,BIN)
        # Self - Normalized 

        return hist2d

    def getHistogram1d_conv(self,tensor, netConv): # L space
        # Preprocess
        tensor1 = netConv((tensor[:,0,:,:].unsqueeze(0)+1)/2) # Make it center

        # Tile
        pool = nn.AvgPool2d(256)
        hist1d = pool(tensor1)

        hist1d = hist1d.repeat(1,1,self.hist_ab,self.hist_ab)
        return hist1d

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
        
    def segmentwise_tile(self,img, seg, final_tensor,segment_num):

        mask_seg = torch.mul( img , (seg == segment_num).cuda().float() )
        #Tile Tensor
        with torch.no_grad():
            hist_2d = self.getHistogram2d_np(mask_seg, self.hist_ab)
            hist_1d = self.getHistogram1d_np(mask_seg, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
            hist_cat   = torch.cat((hist_2d,hist_1d),1)

        #Encode Tensor
        hist_feat = self.netC_A(hist_cat)

        #Embeded Tensor
        final_tensor[:,:,seg.squeeze(0)==segment_num] = hist_feat.repeat(1,1, final_tensor[:,:,seg.squeeze(0)==segment_num].size(2), 1).squeeze(0).permute(2,0,1)


    #######These are not used############
    def read_t7(self,path):
        t7_np = torchfile.load(path[0])
        t7_tensor = torch.from_numpy(t7_np).cuda()

        # Get Label
        values, indicies = torch.max(t7_tensor, 1)

        return indicies

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
  
    def GCT(self,img_A,img_B):
        #Reinhard, Erik, et al. "Color transfer between images."

        mean_A_L = np.mean(img_A[:,0,:,:].cpu().numpy())
        mean_A_a = np.mean(img_A[:,1,:,:].cpu().numpy())
        mean_A_b = np.mean(img_A[:,2,:,:].cpu().numpy())

        mean_B_L = np.mean(img_B[:,0,:,:].cpu().numpy())
        mean_B_a = np.mean(img_B[:,1,:,:].cpu().numpy())
        mean_B_b = np.mean(img_B[:,2,:,:].cpu().numpy())

        std_A_L = np.std(img_A[:,0,:,:].cpu().numpy())
        std_A_a = np.std(img_A[:,1,:,:].cpu().numpy())
        std_A_b = np.std(img_A[:,2,:,:].cpu().numpy())

        std_B_L = np.std(img_B[:,0,:,:].cpu().numpy())
        std_B_a = np.std(img_B[:,1,:,:].cpu().numpy())
        std_B_b = np.std(img_B[:,2,:,:].cpu().numpy())


        reinhard = torch.zeros(1,3,img_A.size(2),img_A.size(3))
        reinhard_L = ((img_A[:,0,:,:].cpu().numpy() - mean_A_L) * std_B_L / std_A_L) + mean_B_L
        reinhard_a = ((img_A[:,1,:,:].cpu().numpy() - mean_A_a) * std_B_a / std_A_a) + mean_B_a
        reinhard_b = ((img_A[:,2,:,:].cpu().numpy() - mean_A_b) * std_B_b / std_A_b) + mean_B_b

        reinhard[:,0,:,:] = torch.from_numpy(reinhard_L)
        reinhard[:,1,:,:] = torch.from_numpy(reinhard_a)
        reinhard[:,2,:,:] = torch.from_numpy(reinhard_b)

        return reinhard

    def GCT_segmentwise(self,img_src_tp,img_tgt_tp, seg_src_tp, seg_tgt_tp):
        img_src = copy.deepcopy(img_src_tp)
        img_tgt = copy.deepcopy(img_tgt_tp)
        seg_src = copy.deepcopy(seg_src_tp)
        seg_tgt = copy.deepcopy(seg_tgt_tp)

        reinhard = img_src

        for i in range(0, self.input_SegNum):
            seg_num_A = torch.sum(torch.sum(seg_src == i ,1),1)
            seg_num_B = torch.sum(torch.sum(seg_tgt == i ,1),1)

            #print(img_src.squeeze(0).squeeze(0))
           
            if (seg_num_A > 0) and (seg_num_B > 0):
                mask_seg_src = torch.mul( img_src , (seg_src == i).cuda().float() ).cpu().numpy()
                mask_seg_tgt = torch.mul( img_tgt , (seg_tgt == i).cuda().float() ).cpu().numpy()

                mask_seg_src_nan = np.where(mask_seg_src!=0,mask_seg_src,np.nan)
                mask_seg_tgt_nan = np.where(mask_seg_tgt!=0,mask_seg_tgt,np.nan)


                mean_A_L = np.nanmean(mask_seg_src_nan[:,0,:,:])
                mean_A_a = np.nanmean(mask_seg_src_nan[:,1,:,:])
                mean_A_b = np.nanmean(mask_seg_src_nan[:,2,:,:])

                mean_B_L = np.nanmean(mask_seg_tgt_nan[:,0,:,:])
                mean_B_a = np.nanmean(mask_seg_tgt_nan[:,1,:,:])
                mean_B_b = np.nanmean(mask_seg_tgt_nan[:,2,:,:])

                std_A_L = np.nanstd(mask_seg_src_nan[:,0,:,:])
                std_A_a = np.nanstd(mask_seg_src_nan[:,1,:,:])
                std_A_b = np.nanstd(mask_seg_src_nan[:,2,:,:])

                std_B_L = np.nanstd(mask_seg_tgt_nan[:,0,:,:])
                std_B_a = np.nanstd(mask_seg_tgt_nan[:,1,:,:])
                std_B_b = np.nanstd(mask_seg_tgt_nan[:,2,:,:])


                reinhard_L = ((img_src[:,0,:,:].cpu().numpy() - mean_A_L) * std_B_L / std_A_L) + mean_B_L
                reinhard_a = ((img_src[:,1,:,:].cpu().numpy() - mean_A_a) * std_B_a / std_A_a) + mean_B_a
                reinhard_b = ((img_src[:,2,:,:].cpu().numpy() - mean_A_b) * std_B_b / std_A_b) + mean_B_b


                #print(torch.from_numpy(reinhard_L)[:,seg_src.squeeze(0)==i].size())
                #print(reinhard[:,0,seg_src.squeeze(0)==i].size())


                reinhard[:,0,seg_src.squeeze(0)==i] = torch.from_numpy(reinhard_L)[:,seg_src.squeeze(0)==i].cuda().float()
                reinhard[:,1,seg_src.squeeze(0)==i] = torch.from_numpy(reinhard_a)[:,seg_src.squeeze(0)==i].cuda().float()
                reinhard[:,2,seg_src.squeeze(0)==i] = torch.from_numpy(reinhard_b)[:,seg_src.squeeze(0)==i].cuda().float()

        return reinhard          

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def segmentwise_tile_test_stretch(self, img, seg_src, final_tensor):

        mask_seg = torch.mul( img , (seg_src).cuda().float() )
        #Tile Tensor
        with torch.no_grad():
            hist_2d = self.getHistogram2d_np(mask_seg, self.hist_ab)
            hist_1d = self.getHistogram1d_np(mask_seg, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)

            dresize = 82
            self.hist_ab = 64
            gap = int((dresize - self.hist_ab)/2)
            hist_2d_np = hist_2d.squeeze(0).permute(1,2,0).cpu().numpy() # 64/64
            hist_2d_np_resize = cv2.resize(hist_2d_np, dsize = (dresize, dresize), interpolation = cv2.INTER_LINEAR)
            hist_2d_np_resize_crop = hist_2d_np_resize[gap:gap+self.hist_ab, gap:gap+self.hist_ab]
            hist_2d_ts_resize_crop_tensor = torch.from_numpy(hist_2d_np_resize_crop).cuda() #64/64

            #Norm
            total_num = sum(sum(hist_2d_ts_resize_crop_tensor))
            hist_2d_ts_resize_crop_tensor_norm = hist_2d_ts_resize_crop_tensor / total_num


            hist_cat   = torch.cat((hist_2d_ts_resize_crop_tensor_norm.unsqueeze(0).unsqueeze(0),hist_1d),1)

        #Encode Tensor
        hist_feat = self.netC_A(hist_cat)

        final_tensor[:,:,seg_src.squeeze(0).cuda()==1] = hist_feat.repeat(1,1, final_tensor[:,:,seg_src.squeeze(0).cuda()==1].size(2), 1).squeeze(0).permute(2,0,1)


    

