import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def IRN(input_nc, output_nc, ngf, which_model_netG, norm='none', use_dropout=False, init_type='normal', gpu_ids=[]): # Batch norm -> nonw
    IRN = None

    IRN = HISTUnet3_Res(input_nc, output_nc, ngf, norm_layer=get_norm_layer(norm_type=norm), use_dropout=use_dropout).cuda()
    init_weights(IRN, init_type=init_type)

    return IRN


def HEN(input_nc, output_nc, ngf, which_model_netC, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    HEN = None

    HEN = ConditionNetwork2(input_nc, output_nc, ngf, norm_layer=get_norm_layer(norm_type=norm), use_dropout=use_dropout, gpu_ids=gpu_ids).cuda()
    init_weights(HEN, init_type=init_type)

    return HEN

##############################################################################
# Classes
##############################################################################

class ConditionNetwork2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork2, self).__init__()
        self.input_nc = input_nc # 10
        self.output_nc = output_nc # 32
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dim2 = 128

        model = [
                #### Half Size
                 nn.Conv2d(self.input_nc, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                #### -1 Size
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                #### Out Change
                 nn.Conv2d(dim2, self.output_nc, kernel_size=1, padding=0,bias=use_bias),
        ]

        self.model = nn.Sequential(*model)

        self.model2 = nn.Sequential(nn.Linear(output_nc,output_nc))

        self.model3 = nn.Sequential(nn.Linear(output_nc,324))

    def forward(self, input):

        # Original
        #a1 = self.model(input) # 1/32/6/6
        #a2 = a1.view(a1.size(0),-1)
        #a3 = self.model3(a2)
        #a3 = a3.unsqueeze(0).unsqueeze(0).permute(0,3,1,2) # 1,32,1,1

        # Ver.2
        a1 = self.model(input)
        a2 = a1.view(a1.size(0),-1)
        a3 = self.model2(a2)
        a3 = a3.unsqueeze(0).unsqueeze(0).permute(2,3,0,1) # 1/64/1/1

        return a3

class HISTUnet3_Res(nn.Module): 
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
        super(HISTUnet3_Res, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        enc_nc = 64

        #Init
        self.block_init  = self.build_init_block(input_nc,enc_nc,padding_type, use_bias)
        #self.block_init2 = self.build_init_block(input_nc,enc_nc,padding_type, use_bias)

        #Enc
        self.HistUnetEnc = UnetEnc_deep(input_nc, 64, norm_layer)

        #Dec
        self.HistUnetDec1 = UnetDec_deep2(512 + 512 + enc_nc + enc_nc,  512, norm_layer) #512 + 512 + 72 + 72
        self.HistUnetDec2 = UnetDec_deep2(512 + 512 + enc_nc + enc_nc,  256, norm_layer)
        self.HistUnetDec3 = UnetDec_deep2(256 + 256 + enc_nc + enc_nc,  128, norm_layer)
        self.HistUnetDec4 = UnetDec_deep2(128 + 128 + enc_nc + enc_nc,   64, norm_layer)
        self.HistUnetDec5 = UnetDec_deep1( 64 +  64 + enc_nc + enc_nc,   64, norm_layer)

        #Res
        self.ENC_Block1 = ENCResnetBlock(enc_nc, padding_type, norm_layer, use_dropout, use_bias)
        self.ENC_Block2 = ENCResnetBlock(enc_nc, padding_type, norm_layer, use_dropout, use_bias)
        self.ENC_Block3 = ENCResnetBlock(enc_nc, padding_type, norm_layer, use_dropout, use_bias)

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

    def build_inter_out(self,dim_img,dim_out,padding_type,use_bias):
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

        block_last += [nn.Conv2d(dim_img,dim_out,kernel_size=3,padding=p,bias=use_bias),
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
                        # 2
                        nn.ReLU(True),
                        nn.Conv2d(dim_img, dim_img, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(dim_img),

                        # 2
                        nn.ReLU(True),
                        nn.Conv2d(dim_img, dim_out, kernel_size=3,stride=1, padding=1),
                        ]
        return nn.Sequential(*block_last)


    def forward(self, input_img, hist1_enc, hist2_enc):

        # Enc
        mid_img1, mid_img2, mid_img3, mid_img4, mid_img5 = self.HistUnetEnc(input_img) # 1/3/256/256

        # Dec
        out_img1 = self.HistUnetDec1(mid_img5, mid_img5, hist1_enc, hist2_enc, mid_img5.size(2), mid_img5.size(3)) #
        out_img2 = self.HistUnetDec2(out_img1, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3)) # 
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3)) # 
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3)) # 
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3)) #
        out_img5 = F.upsample(out_img5,size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')

        out_img6 = self.ENC_Block1.forward(out_img5 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img7 = self.ENC_Block2.forward(out_img6 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img  = self.ENC_Block3.forward(out_img7 + self.block_init(input_img), hist1_enc, hist2_enc)

        out_img1 = self.InterOut1(out_img1)
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)

        out_img    = self.block_last(out_img + self.block_init(input_img))

        return out_img1, out_img2, out_img3, out_img4, out_img

#################################################################################################################################################################################################################################################################################################

class UnetEnc_deep(nn.Module):
    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep, self).__init__()

        self.netEnc1 = UnetEnc_deep1(input_channel,ngf,norm_layer)
        self.netEnc2 = UnetEnc_deep2(ngf * 1, ngf * 2, norm_layer) #64, 128
        self.netEnc3 = UnetEnc_deep2(ngf * 2, ngf * 4, norm_layer) #128, 256
        self.netEnc4 = UnetEnc_deep2(ngf * 4, ngf * 8, norm_layer) #256, 512
        self.netEnc5 = UnetEnc_deep2(ngf * 8, ngf * 8, norm_layer) #512, 512

    def forward(self, input):
        output1 = self.netEnc1.forward(input)  # 256/256/64 -> 128/128/64
        output2 = self.netEnc2.forward(output1) # 128/128/64 -> 64/64/128
        output3 = self.netEnc3.forward(output2) # 64/64/128 -> 32/32/256
        output4 = self.netEnc4.forward(output3) # 32/32/256 -> 16/16/512
        output5 = self.netEnc5.forward(output4) # 16/16/512 -> 8/8/512

        return output1, output2, output3, output4, output5

class UnetEnc_deep1(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep1, self).__init__()
        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [

        #1
        nn.Conv2d(input_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),

        #2
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),

        #2 Down
        nn.Conv2d(output_nc, output_nc, kernel_size=4,stride=2, padding=1),
        nn.InstanceNorm2d(output_nc),
        ]

        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class UnetEnc_deep2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep2, self).__init__()

        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [

        #1
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(input_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),

        #2
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),

        #3
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(output_nc, output_nc, kernel_size=4,stride=2, padding=1),
        nn.InstanceNorm2d(output_nc),
        ]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)

class UnetDec_deep1(nn.Module):
    def __init__(self, input_nc, output_nc,use_tanh,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep1, self).__init__()

        self.block = self.build_block(input_nc,output_nc,use_tanh,norm_layer=norm_layer)


    def build_block(self,input_nc,output_nc,use_tanh,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [

                        # 1
                        nn.ReLU(True),
                        nn.Upsample(scale_factor=2, mode = 'bilinear'),
                        nn.Conv2d(input_nc,128,kernel_size=3,stride=1,padding=1, bias=use_bias),
                        nn.InstanceNorm2d(128),

                        # 2
                        nn.ReLU(True),
                        nn.Conv2d(128, 64, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(64),

                        # 2
                        nn.ReLU(True),
                        nn.Conv2d(64, output_nc, kernel_size=3,stride=1, padding=1),

                        ]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2, target_size1, target_size2):
        input1=F.upsample(input1,size= (target_size1, target_size2), mode = 'bilinear')
        enc1 = F.upsample(enc1, size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size= (target_size1, target_size2), mode = 'bilinear')

        out = self.block(torch.cat([input1, input2, enc1, enc2],1))
        return out

class UnetDec_deep2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep2, self).__init__()

        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)


    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [  
                        # 1
                        nn.ReLU(True),
                        nn.Upsample(scale_factor=2, mode = 'bilinear'),
                        nn.Conv2d(input_nc,output_nc,kernel_size=3,stride=1,padding=1, bias=use_bias),
                        nn.InstanceNorm2d(output_nc),

                        # 2
                        nn.ReLU(True),
                        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(output_nc),

                        # 3
                        nn.ReLU(True),
                        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(output_nc),

                        ]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2, target_size1, target_size2):

        input1=F.upsample(input1,size = (target_size1, target_size2), mode = 'bilinear')
        enc1 = F.upsample(enc1,  size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2,  size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size = (target_size1, target_size2), mode = 'bilinear')

        out = self.block(torch.cat([input1, input2, enc1, enc2],1))

        return out

class ENCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ENCResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d( dim * 3, dim * 3, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim * 3),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim * 3, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, enc1, enc2): #64/64/64
        enc1 = F.upsample(enc1,size = (x.size(2), x.size(3)), mode = 'bilinear')
        enc2 = F.upsample(enc2,size = (x.size(2), x.size(3)), mode = 'bilinear')



        x_cat = torch.cat((x,enc1,enc2),1) # 64/64/64/
        out = x + self.conv_block(x_cat) # 192 -> 64        
        return out
