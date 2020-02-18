import math
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import functools
import torch.nn.functional as F

# More Deeper Unet
class StdUnet_woIN(nn.Module): 
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
        super(StdUnet_woIN, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        enc_nc = 64

        #Init
        self.block_init  = self.build_init_block(input_nc,enc_nc,padding_type, use_bias)
        #self.block_init2 = self.build_init_block(input_nc,enc_nc,padding_type, use_bias)

        #Enc
        self.HistUnetEnc = StdUnetEnc_deep(input_nc, 64, norm_layer)

        #Dec
        self.HistUnetDec1 = StdUnetDec_deep2(512 + 512 + enc_nc + enc_nc,  512, norm_layer) #512 + 512 + 72 + 72
        self.HistUnetDec2 = StdUnetDec_deep2(512 + 512 + enc_nc + enc_nc,  256, norm_layer)
        self.HistUnetDec3 = StdUnetDec_deep2(256 + 256 + enc_nc + enc_nc,  128, norm_layer)
        self.HistUnetDec4 = StdUnetDec_deep2(128 + 128 + enc_nc + enc_nc,   64, norm_layer)
        self.HistUnetDec5 = StdUnetDec_deep1( 64 +  64 + enc_nc + enc_nc,   output_nc, norm_layer)

        #Out
        self.InterOut1 = self.build_inter_out2(512, output_nc, 'zero', use_bias)
        self.InterOut2 = self.build_inter_out2(256, output_nc, 'zero', use_bias)
        self.InterOut3 = self.build_inter_out2(128, output_nc, 'zero', use_bias)
        self.InterOut4 = self.build_inter_out2( 64, output_nc, 'zero', use_bias)

        self.block_last = self.build_last_block(64,output_nc,padding_type,use_bias) 
 
    
    def build_init_block(self, input_nc,dim_img, padding_type, use_bias): # 3 -> 64
        block_init =[]

        p = 0
        if padding_type == 'reflect':
            block_init += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_init += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        block_init += [nn.Conv2d(input_nc,dim_img,kernel_size=3,padding=p,stride=1),
                       nn.InstanceNorm2d(dim_img),
                       nn.ReLU(True),
                       nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,stride=1),
        ]
        
        return nn.Sequential(*block_init)

    def build_last_block(self,dim_img,output_nc,padding_type,use_bias):
        block_last = []

        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        block_last += [nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,bias=use_bias),
                        nn.ReLU(True),
                        nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,bias=use_bias),
                        nn.ReLU(True),
                        nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,bias=use_bias),
                        nn.ReLU(True),
                        nn.Conv2d(dim_img,output_nc,kernel_size=3,padding=p,bias=use_bias)
                        ]

        return nn.Sequential(*block_last)


    def build_inter_out2(self,dim_img,dim_out,padding_type,use_bias):
        block_last = []

        p = 0
        if padding_type == 'reflect':
            block_last += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            block_last += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        block_last += [
                        # 1
                        nn.Conv2d(dim_img, dim_img, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(dim_img),
                        nn.ReLU(True),

                        # 2
                        nn.Conv2d(dim_img, dim_img, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(dim_img),
                        nn.ReLU(True),

                        # 3
                        nn.Conv2d(dim_img, dim_out, kernel_size=3,stride=1, padding=1),

                        ]
        return nn.Sequential(*block_last)


    def forward(self, input_img, hist1_enc, hist2_enc):

        # Enc
        mid_img1, mid_img2, mid_img3, mid_img4, mid_img5 = self.HistUnetEnc(input_img) # 1/3/256/256



        # Dec
        out_img2 = self.HistUnetDec2(mid_img5, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3)) # 
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3)) # 
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3)) # 
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3)) #
        out_img5 = F.upsample(out_img5,size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')
        
        # Out
        out_img1 = out_img5 
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)

        # Residual
        out_img = out_img5

        return out_img1, out_img2, out_img3, out_img4, out_img




class StdUnetEnc_deep(nn.Module):
    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(StdUnetEnc_deep, self).__init__()

        self.netEnc1 = StdUnetEnc_deep1(input_channel,ngf,norm_layer)
        self.netEnc2 = StdUnetEnc_deep2(ngf * 1, ngf * 2, norm_layer) #64, 128
        self.netEnc3 = StdUnetEnc_deep2(ngf * 2, ngf * 4, norm_layer) #128, 256
        self.netEnc4 = StdUnetEnc_deep2(ngf * 4, ngf * 8, norm_layer) #256, 512
        self.netEnc5 = StdUnetEnc_deep2(ngf * 8, ngf * 8, norm_layer) #512, 512

    def forward(self, input):
        output1 = self.netEnc1.forward(input)  # 256/256/64 -> 128/128/64
        output2 = self.netEnc2.forward(output1) # 128/128/64 -> 64/64/128
        output3 = self.netEnc3.forward(output2) # 64/64/128 -> 32/32/256
        output4 = self.netEnc4.forward(output3) # 32/32/256 -> 16/16/512
        output5 = self.netEnc5.forward(output4) # 16/16/512 -> 8/8/512

        return output1, output2, output3, output4, output5

class StdUnetEnc_deep1(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(StdUnetEnc_deep1, self).__init__()
        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [

        #1
        nn.Conv2d(input_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.LeakyReLU(0.2, True),

        #2
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.LeakyReLU(0.2, True),

        #2 Down
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.LeakyReLU(0.2, True),
        ]

        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class StdUnetEnc_deep2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(StdUnetEnc_deep2, self).__init__()

        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [

        #1
        nn.Conv2d(input_nc, output_nc, kernel_size=4,stride=2, padding=1),
        nn.LeakyReLU(0.2, True),

        #2
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.LeakyReLU(0.2, True),

        #3
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.LeakyReLU(0.2, True),
        ]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)

class StdUnetDec_deep2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(StdUnetDec_deep2, self).__init__()

        self.up    = self.build_up()
        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_up(self):
        block_full = [  
                        # 0
                        nn.Upsample(scale_factor=2, mode = 'bilinear'),

        ]
        return nn.Sequential(*block_full)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [  

                        # 1
                        nn.Conv2d(input_nc,output_nc,kernel_size=3,stride=1,padding=1, bias=use_bias),
                        nn.ReLU(True),

                        # 2
                        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
                        nn.ReLU(True),

                        # 3
                        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
                        nn.ReLU(True),

                        ]

        return nn.Sequential(*block_full)

    def forward(self, input_prev, input_skip, enc1, enc2, target_size1, target_size2):


        input_prev = self.up(input_prev)

        input_prev= F.upsample(input_prev, size = (target_size1, target_size2), mode = 'bilinear')
        enc1      = F.upsample(enc1,       size = (target_size1, target_size2), mode = 'bilinear')
        enc2      = F.upsample(enc2,       size = (target_size1, target_size2), mode = 'bilinear')
        input2    = F.upsample(input_skip, size = (target_size1, target_size2), mode = 'bilinear')

        # Should be this format
        out = torch.cat([input_prev, input2, enc1, enc2],1)
        out = self.block(out)

        return out

class StdUnetDec_deep1(nn.Module):
    def __init__(self, input_nc, output_nc,use_tanh,norm_layer=nn.BatchNorm2d):
        super(StdUnetDec_deep1, self).__init__()

        self.up    = self.build_up()
        self.block = self.build_block(input_nc,output_nc,use_tanh,norm_layer=norm_layer)

    def build_up(self):
        block_full = [  
                        # 0
                        nn.Upsample(scale_factor=2, mode = 'bilinear'),

        ]
        return nn.Sequential(*block_full)

    def build_block(self,input_nc,output_nc,use_tanh,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [

                        # 1
                        nn.Conv2d(input_nc,128,kernel_size=3,stride=1,padding=1, bias=use_bias),
                        nn.ReLU(True),

                        # 2
                        nn.Conv2d(128, 64, kernel_size=3,stride=1, padding=1),
                        nn.ReLU(True),

                        # 3
                        nn.Conv2d(64, output_nc, kernel_size=3,stride=1, padding=1),
                        #nn.InstanceNorm2d(64),
                        #nn.ReLU(True),

                        ]

        #if use_tanh:
        #    block_full += [nn.Tanh()]

        return nn.Sequential(*block_full)

    def forward(self, input_prev, input_skip, enc1, enc2, target_size1, target_size2):

        input_prev = self.up(input_prev)

        input_prev= F.upsample(input_prev,size= (target_size1, target_size2), mode = 'bilinear')
        enc1 =      F.upsample(enc1, size = (target_size1, target_size2), mode = 'bilinear')
        enc2 =      F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')
        input_skip= F.upsample(input_skip,size= (target_size1, target_size2), mode = 'bilinear')


        #out = self.block(torch.cat([input1, input2, enc1, enc2],1))
        out = torch.cat([input_prev, input_skip, enc1, enc2],1)
        out = self.block(out)


        return out