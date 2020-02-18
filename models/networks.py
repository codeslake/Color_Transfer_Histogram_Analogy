import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
#from modules.architecture import VGGFeatureExtractor
import torchvision
import torch.nn.functional as F

#from .architecture import VGGFeatureExtractor
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
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


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='none', use_dropout=False, init_type='normal', gpu_ids=[]): # Batch norm -> nonw
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)
    print('Net_G: ', which_model_netG)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'iccv_submitted':
        netG = HISTUnet3_Res(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    elif which_model_netG == 'cvpr':
        netG = HISTUnet3_Res2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    
    elif which_model_netG == 'hist_stdunet':
        netG = HISTStdUnet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    
    elif which_model_netG == 'stdunet_woIN':    
        from models.modules.stdunet_woIN import StdUnet_woIN
        netG = StdUnet_woIN(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    
    #Ablation Study

        
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)


    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_F(gpu_ids, use_bn=False):
    tensor = torch.cuda.FloatTensor if gpu_ids else torch.FloatTensor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, tensor=tensor)
    if gpu_ids:
        netF = nn.DataParallel(netF).cuda()
    netF.eval()  # No need to train
    return netF

def define_C(input_nc, output_nc, ngf, which_model_netC, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netCon = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    #netCon = ConditionNetwork2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)

    if use_gpu:
        assert(torch.cuda.is_available())
        
    if which_model_netC == 'basic': #size 64
        netCon = ConditionNetwork2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netC == 'image': 
        netCon = ConditionNetwork3(3, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netC == 'basic_8':
        netCon = ConditionNetwork_8(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netC == 'basic_32':
        netCon = ConditionNetwork_32(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netC == 'basic_128':
        netCon = ConditionNetwork_128(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)    
    elif which_model_netC == 'fc':
        netCon = ConditionNetwork_fc(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('network C model name [%s] is not recognized' % which_model_netC)
    if use_gpu:
        netCon.cuda(gpu_ids[0])
    init_weights(netCon, init_type=init_type)
    return netCon

#####################################################
# Histogram
#####################################################
class HistogramNet(nn.Module):
    def __init__(self,bin_num):
        super(HistogramNet,self).__init__()
        self.bin_num = bin_num
        self.LHConv_1 = BiasedConv1(1,bin_num)
        self.relu = nn.ReLU(True)

    def forward(self,input):
        a1 = self.LHConv_1(input)
        a2 = torch.abs(a1)
        a3 = 1- a2*(self.bin_num-1)
        a4 = self.relu(a3)
        return a4

    def getBiasedConv1(self):
        return self.LHConv_1

    def getBin(self):
        return self.bin_num

    def init_biased_conv1(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            #m.bias.data = -torch.arange(0,1,1/(self.bin_num-1)) # Originally this was it... 19/03/02
            m.bias.data = -torch.arange(0,1,1/(self.bin_num))
            m.weight.data = torch.ones(self.bin_num,1,1,1)


class BiasedConv1(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(BiasedConv1, self).__init__()
        model = []
        model += [nn.Conv2d(dim_in,dim_out,kernel_size=1,padding=0,stride=1,bias=True),]
        self.model = nn.Sequential(*model)

    def forward(self,input):
        a = self.model(input)
        return a

def define_Hist(bin_num):
    netHist = HistogramNet(bin_num)
    netHist.getBiasedConv1().apply(netHist.init_biased_conv1)
    netHist.eval()  # No need to train
    return netHist

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 tensor=torch.FloatTensor):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = Variable(tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = Variable(tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class ConditionNetwork3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork3, self).__init__()
        self.input_nc = input_nc # 3
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

class ConditionNetwork_8(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork_8, self).__init__()
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
                 ##nn.Conv2d(self.input_nc, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 ##nn.LeakyReLU(0.1, True),
                 #nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 #nn.LeakyReLU(0.1, True),
                 ##nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 ##nn.LeakyReLU(0.1, True),
                 ##nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 ##nn.LeakyReLU(0.1, True),
                #### -1 Size
                 nn.Conv2d(self.input_nc, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
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

class ConditionNetwork_32(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork_32, self).__init__()
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
                 #nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 #nn.LeakyReLU(0.1, True),
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

class ConditionNetwork_128(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork_128, self).__init__()
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

                 # Added This Layer
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

class ConditionNetwork_fc(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork_fc, self).__init__()
        self.input_nc = input_nc # 1
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
        a3 = a3.unsqueeze(0).unsqueeze(0).permute(0,3,1,2) # 1/64/1/1

        return a3


# For Numpy Histogram
class ConditionNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(ConditionNetwork, self).__init__()
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
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
                 nn.LeakyReLU(0.1, True),

                #### -1 Size
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),
                 nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
                 nn.LeakyReLU(0.1, True),

                #### Out Change
                 nn.Conv2d(dim2, self.output_nc, kernel_size=1, padding=0,bias=use_bias),
        ]

        self.model = nn.Sequential(*model)

        self.model2 = nn.Sequential(nn.Linear(32,32))

        self.model3 = nn.Sequential(nn.Linear(32,324))

    def forward(self, input):

        a1 = self.model(input)
        a2 = a1.view(a1.size(0),-1)
        a3 = self.model3(a2)
        a3 = a3.unsqueeze(0).unsqueeze(0).permute(0,3,1,2) # 1,32,1,1
        return a3

#        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#        else:
#            a1 = self.model(input)
#            a2 = a1.view(a1.size(0),-1)
#            a3 = self.model2(a2)
#
#            print(a2)
#            print(ke)
#
#            return a3
#




# More Deeper Unet
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
        #print("SAENGAK")
        # print('m1', mid_img1.size())# 1/64/128/128
        # print('m2', mid_img2.size())# 1/128/64/64
        # print('m3', mid_img3.size())# 1/256/32/32
        # print('m4', mid_img4.size())# 1/512/16/16
        # print('m5', mid_img5.size())# 1/512/8/8


        # Dec
        out_img1 = self.HistUnetDec1(mid_img5, mid_img5, hist1_enc, hist2_enc, mid_img5.size(2), mid_img5.size(3)) #
        out_img2 = self.HistUnetDec2(out_img1, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3)) # 
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3)) # 
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3)) # 
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3)) #
        out_img5 = F.upsample(out_img5,size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')
        #print("SAENGAK2")
        # print('o1 ', out_img1.size())# 
        # print('o2 ', out_img2.size())# 
        # print('o3 ', out_img3.size())# 
        # print('o4 ', out_img4.size())# 
        # print('o5 ', out_img5.size())# 



        ###### ADDED NOW
        #print("ALRA")
        #print(out_img5.size()) # 1/64/416/704
        #print(input_img.size())# 1/3/440/704
        #temp = self.block_init2(input_img)
        #print(temp.size())     # 1/64/440/704   
        #out_img5 = out_img5 + temp




        ##### SMAE STRUCTURE
        out_img6 = self.ENC_Block1.forward(out_img5 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img7 = self.ENC_Block2.forward(out_img6 + self.block_init(input_img), hist1_enc, hist2_enc)
        out_img  = self.ENC_Block3.forward(out_img7 + self.block_init(input_img), hist1_enc, hist2_enc)

        #print(out_img1.size()) # 1/512/16/16 
        #print(out_img2.size()) # 1/256/32/32
        #print(out_img3.size()) # 1/128/64/64
        #print(out_img4.size()) # 1/ 64/128/128
        #print(out_img5.size()) # 1/ 3/256/256
        
        # Out
        out_img1 = self.InterOut1(out_img1)
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)

        # Residual
        out_img    = self.block_last(out_img + self.block_init(input_img))

        #print(out_img1.size()) # 1/3/16/16 
        #print(out_img2.size()) # 1/3/32/32
        #print(out_img3.size()) # 1/3/64/64
        #print(out_img4.size()) # 1/3/128/128
        #print(out_img5.size()) # 1/3/256/256


        return out_img1, out_img2, out_img3, out_img4, out_img

# More Deeper Unet
class HISTStdUnet(nn.Module): 
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
        super(HISTStdUnet, self).__init__()

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
        #print("SAENGAK")
        #print(mid_img1.size())# 1/64 /256/256
        #print(mid_img2.size())# 1/128/128/128
        #print(mid_img3.size())# 1/256/ 64/ 64
        #print(mid_img4.size())# 1/512/ 32/ 32
        #print(mid_img5.size())# 1/512/ 16/ 16


        # Dec
        
        #out_img1 = self.HistUnetDec1(mid_img5, mid_img5, hist1_enc, hist2_enc, mid_img5.size(2), mid_img5.size(3)) #
        out_img2 = self.HistUnetDec2(mid_img5, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3)) # 
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3)) # 
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3)) # 
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3)) #
        out_img5 = F.upsample(out_img5,size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')

        #print("SAENGAK2")
        #print(out_img1.size())# 
        #print(out_img2.size())# 
        #print(out_img3.size())# 
        #print(out_img4.size())# 
        #print(out_img5.size())# 



        ###### ADDED NOW
        #print("ALRA")
        #print(out_img5.size()) # 1/64/416/704
        #print(input_img.size())# 1/3/440/704
        #temp = self.block_init2(input_img)
        #print(temp.size())     # 1/64/440/704   
        #out_img5 = out_img5 + temp




        ##### SMAE STRUCTURE
        #out_img6 = self.ENC_Block1.forward(out_img5 + self.block_init(input_img), hist1_enc, hist2_enc)
        #out_img7 = self.ENC_Block2.forward(out_img6 + self.block_init(input_img), hist1_enc, hist2_enc)
        #out_img  = self.ENC_Block3.forward(out_img7 + self.block_init(input_img), hist1_enc, hist2_enc)

        #print(out_img1.size()) # 1/512/16/16 
        #print(out_img2.size()) # 1/256/32/32
        #print(out_img3.size()) # 1/128/64/64
        #print(out_img4.size()) # 1/ 64/128/128
        #print(out_img5.size()) # 1/ 3/256/256
        
        # Out
        #out_img1 = self.InterOut1(out_img1)
        out_img1 = out_img5 
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)

        # Residual
        out_img = out_img5
        #out_img    = self.block_last(out_img + self.block_init(input_img))

        #print(out_img1.size()) # 1/3/16/16 
        #print(out_img2.size()) # 1/3/32/32
        #print(out_img3.size()) # 1/3/64/64
        #print(out_img4.size()) # 1/3/128/128
        #print(out_img5.size()) # 1/3/256/256


        return out_img1, out_img2, out_img3, out_img4, out_img



class UnetEnc_deep_Front(nn.Module):
    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep_Front, self).__init__()

        enc = 64

        self.netEnc1 = UnetEnc_deep1_Front(input_channel            , ngf * 1, norm_layer) #64 64 64 , 64 64 64 
        self.netEnc2 = UnetEnc_deep2_Front(ngf * 1 + (enc + enc)    , ngf * 2 , norm_layer) #64 64 64 , 128 64 64 
        self.netEnc3 = UnetEnc_deep2_Front(ngf * 2 + (enc + enc)    , ngf * 4 , norm_layer) #128, 256
        self.netEnc4 = UnetEnc_deep2_Front(ngf * 4 + (enc + enc)    , ngf * 8 , norm_layer) #256, 512
        self.netEnc5 = UnetEnc_deep2_Front(ngf * 8 + (enc + enc)    , ngf * 8 , norm_layer) #512, 512

    def forward(self, input, enc1, enc2):
        output1 = self.netEnc1.forward(input  , enc1, enc2)  # 256/256/64 -> 128/128/64
        output2 = self.netEnc2.forward(output1, enc1, enc2)  # 
        output3 = self.netEnc3.forward(output2, enc1, enc2)  # 
        output4 = self.netEnc4.forward(output3, enc1, enc2)  # 
        output5 = self.netEnc5.forward(output4, enc1, enc2)  # 

        return output1, output2, output3, output4, output5

class UnetEnc_deep1_Front(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep1_Front, self).__init__()
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

    def forward(self, input_img, enc1, enc2):
        
        #print("CHOI")
        #print(input_img.size()) # 1/64/256/256
        #print(enc1.size())      # 1/64/256/256
        #print(enc2.size())      # 1/64/256/256

        enc1 = F.upsample(enc1, size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')
        

        out = self.block(torch.cat([input_img, enc1, enc2],1))


        return out


class UnetEnc_deep2_Front(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep2_Front, self).__init__()

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

    def forward(self, input_img, enc1, enc2):

        enc1 = F.upsample(enc1, size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (input_img.size(2), input_img.size(3)), mode = 'bilinear')


        #print("CHOI2")
        #print(input_img.size()) # 1/64/256/256
        #print(enc1.size())      # 1/64/256/256
        #print(enc2.size())      # 1/64/256/256
        #print(torch.cat([input_img, enc1, enc2],1).size())

        out = self.block(torch.cat([input_img, enc1, enc2],1))
        #print(out.size())

        return out






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

        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size = (target_size1, target_size2), mode = 'bilinear')
        enc1 = F.upsample(enc1,  size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2,  size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size = (target_size1, target_size2), mode = 'bilinear')

        #print('FollowMe')
        #print(input1.size()) # 1/512/8/8
        #print(input2.size()) # 1/512/8/8
        #print(enc1.size())   # 1/ 72/8/8
        #print(enc2.size())   # 1/ 72/8/8


        out = self.block(torch.cat([input1, input2, enc1, enc2],1))

        return out


class UnetDec_deep1_noidt(nn.Module):
    def __init__(self, input_nc, output_nc,use_tanh,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep1_noidt, self).__init__()

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

        #if use_tanh:
        #    block_full += [nn.Tanh()]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc2, target_size1, target_size2):
        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size= (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size= (target_size1, target_size2), mode = 'bilinear')


        out = self.block(torch.cat([input1, input2,  enc2],1))
        return out

class UnetDec_deep2_noidt(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep2_noidt, self).__init__()

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

    def forward(self, input1, input2, enc2, target_size1, target_size2):

        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2,  size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size = (target_size1, target_size2), mode = 'bilinear')

        #print('FollowMe')
        #print(input1.size()) # 1/512/8/8
        #print(input2.size()) # 1/512/8/8
        #print(enc1.size())   # 1/ 72/8/8
        #print(enc2.size())   # 1/ 72/8/8


        out = self.block(torch.cat([input1, input2, enc2],1))

        return out


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

        #if use_tanh:
        #    block_full += [nn.Tanh()]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2, target_size1, target_size2):
        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size= (target_size1, target_size2), mode = 'bilinear')
        enc1 = F.upsample(enc1, size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size= (target_size1, target_size2), mode = 'bilinear')


        out = self.block(torch.cat([input1, input2, enc1, enc2],1))
        return out

class UnetDec_deep2_NoHist(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep2_NoHist, self).__init__()

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

    def forward(self, input1, input2, target_size1, target_size2):

        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size = (target_size1, target_size2), mode = 'bilinear')
        #enc1 = F.upsample(enc1,  size = (target_size1, target_size2), mode = 'bilinear')
        #enc2 = F.upsample(enc2,  size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size = (target_size1, target_size2), mode = 'bilinear')

        #print('FollowMe')
        #print(input1.size()) # 1/512/8/8
        #print(input2.size()) # 1/512/8/8
        #print(enc1.size())   # 1/ 72/8/8
        #print(enc2.size())   # 1/ 72/8/8


        out = self.block(torch.cat([input1, input2],1))

        return out


class UnetDec_deep1_NoHist(nn.Module):
    def __init__(self, input_nc, output_nc,use_tanh,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep1_NoHist, self).__init__()

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

        #if use_tanh:
        #    block_full += [nn.Tanh()]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, target_size1, target_size2):
        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size= (target_size1, target_size2), mode = 'bilinear')
        #enc1 = F.upsample(enc1, size = (target_size1, target_size2), mode = 'bilinear')
        #enc2 = F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')
        input2=F.upsample(input2,size= (target_size1, target_size2), mode = 'bilinear')


        out = self.block(torch.cat([input1, input2],1))
        return out

#################################################################################################################################################################################################################################################################################################

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
        nn.InstanceNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),

        #2
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),

        #2 Down
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),
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
        nn.InstanceNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),

        #2
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),

        #3
        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
        nn.InstanceNorm2d(output_nc),
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
                        nn.InstanceNorm2d(output_nc),
                        nn.ReLU(True),

                        # 2
                        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(output_nc),
                        nn.ReLU(True),

                        # 3
                        nn.Conv2d(output_nc, output_nc, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(output_nc),
                        nn.ReLU(True),

                        ]

        return nn.Sequential(*block_full)

    def forward(self, input_prev, input_skip, enc1, enc2, target_size1, target_size2):


        input_prev = self.up(input_prev)

        #print('FollowMe')
        #print(input_prev.size()) # 1/512/8/8
        #print(input_skip.size()) # 1/512/8/8
        #print(enc1.size())   # 1/ 72/8/8
        #print(enc2.size())   # 1/ 72/8/8
        #print(target_size1)
        #print(target_size2)

        input_prev= F.upsample(input_prev, size = (target_size1, target_size2), mode = 'bilinear')
        enc1      = F.upsample(enc1,       size = (target_size1, target_size2), mode = 'bilinear')
        enc2      = F.upsample(enc2,       size = (target_size1, target_size2), mode = 'bilinear')
        input2    = F.upsample(input_skip, size = (target_size1, target_size2), mode = 'bilinear')




        #out = self.block(torch.cat([input1, input2, enc1, enc2],1))
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
                        nn.InstanceNorm2d(128),
                        nn.ReLU(True),

                        # 2
                        nn.Conv2d(128, 64, kernel_size=3,stride=1, padding=1),
                        nn.InstanceNorm2d(64),
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

        #print('FollowMe2')
        #print(input_prev.size()) 
        #print(input_skip.size()) 
        #print(enc1.size())   
        #print(enc2.size())   
        #print(target_size1)
        #print(target_size2)

        input_prev= F.upsample(input_prev,size= (target_size1, target_size2), mode = 'bilinear')
        enc1 =      F.upsample(enc1, size = (target_size1, target_size2), mode = 'bilinear')
        enc2 =      F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')
        input_skip= F.upsample(input_skip,size= (target_size1, target_size2), mode = 'bilinear')


        #out = self.block(torch.cat([input1, input2, enc1, enc2],1))
        out = torch.cat([input_prev, input_skip, enc1, enc2],1)
        out = self.block(out)

        #print(out.size())   

        return out





#################################################################################################################################################################################################################################################################################################








class UnetEnc(nn.Module):
    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(UnetEnc, self).__init__()

        self.netEnc1 = UnetEnc1(input_channel,ngf,norm_layer)
        self.netEnc2 = UnetEnc2(ngf * 1, ngf * 2, norm_layer) #64, 128
        self.netEnc3 = UnetEnc2(ngf * 2, ngf * 4, norm_layer) #128, 256
        self.netEnc4 = UnetEnc2(ngf * 4, ngf * 8, norm_layer) #256, 512
        self.netEnc5 = UnetEnc2(ngf * 8, ngf * 8, norm_layer) #512, 512

    def forward(self, input):
        output1 = self.netEnc1.forward(input)  # 256/256/64 -> 128/128/64
        output2 = self.netEnc2.forward(output1) # 128/128/64 -> 64/64/128
        output3 = self.netEnc3.forward(output2) # 64/64/128 -> 32/32/256
        output4 = self.netEnc4.forward(output3) # 32/32/256 -> 16/16/512
        output5 = self.netEnc5.forward(output4) # 16/16/512 -> 8/8/512

        return output1, output2, output3, output4, output5

class UnetEnc1(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc1, self).__init__()
        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [nn.Conv2d(input_nc, output_nc, kernel_size=4,stride=2, padding=1)]

        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)


class UnetEnc2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc2, self).__init__()

        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [nn.LeakyReLU(0.2, True),
                        nn.Conv2d(input_nc, output_nc, kernel_size=4,stride=2, padding=1),
                        nn.InstanceNorm2d(output_nc)]
        return nn.Sequential(*block_full)

    def forward(self, input_img):
        return self.block(input_img)

class UnetDec2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetDec2, self).__init__()

        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [  nn.ReLU(True),
                        # Changed Upsampling
                        nn.Upsample(scale_factor=2, mode = 'bilinear'),
                        nn.Conv2d(input_nc,output_nc,kernel_size=3,stride=1,padding=1, bias=use_bias),
                        #nn.Conv2d(output_nc,output_nc,kernel_size=1,stride=1,padding=0, bias=use_bias),

                        # Original
                        #nn.ConvTranspose2d(input_nc, output_nc,kernel_size=4, stride=2,padding=1),

                        # Always is
                        nn.InstanceNorm2d(output_nc)]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2):

        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        enc1 = F.upsample(enc1,  size = (input1.size(2), input1.size(3)), mode = 'bilinear')
        enc2 = F.upsample(enc2,  size = (input1.size(2), input1.size(3)), mode = 'bilinear')
        input2=F.upsample(input2,size = (input1.size(2), input1.size(3)), mode = 'bilinear')

        print('FollowMe')
        print(input1.size()) # 1/512/8/8
        print(input2.size()) # 1/512/8/8
        print(enc1.size())   # 1/ 72/8/8
        print(enc2.size())   # 1/ 72/8/8


        out = self.block(torch.cat([input1, input2, enc1, enc2],1))

        return out


class UnetDec1(nn.Module):
    def __init__(self, input_nc, output_nc,use_tanh,norm_layer=nn.BatchNorm2d):
        super(UnetDec1, self).__init__()

        self.block = self.build_block(input_nc,output_nc,use_tanh,norm_layer=norm_layer)

    def build_block(self,input_nc,output_nc,use_tanh,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [nn.ReLU(True),

                        # Changed
                        nn.Upsample(scale_factor=2, mode = 'bilinear'),
                        nn.Conv2d(input_nc,output_nc,kernel_size=3,stride=1,padding=1, bias=use_bias),
                        #nn.Conv2d(outer_nc,outer_nc,kernel_size=1,stride=1,padding=0, bias=use_bias),
                        
                        # Original
                        #nn.ConvTranspose2d(input_nc, output_nc,kernel_size=4, stride=2,padding=1),

                        #nn.Tanh()
                        ]

        if use_tanh:
            block_full += [nn.Tanh()]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2):
        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        enc1 = F.upsample(enc1, size = (input1.size(2), input1.size(3)), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (input1.size(2), input1.size(3)), mode = 'bilinear')
        input2=F.upsample(input2,size= (input1.size(2), input1.size(3)), mode = 'bilinear')


        out = self.block(torch.cat([input1, input2, enc1, enc2],1))
        return out
        





# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
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
        #print('****')
        #print(enc1.shape)
        #print(enc2.shape)
        #print(x.shape)
        enc1 = F.upsample(enc1,size = (x.size(2), x.size(3)), mode = 'bilinear')
        enc2 = F.upsample(enc2,size = (x.size(2), x.size(3)), mode = 'bilinear')
        #print(x.size())
        #print(enc1.size())
        #print(enc2.size())



        x_cat = torch.cat((x,enc1,enc2),1) # 64/64/64/
        #print(x_cat.shape)
        out = x + self.conv_block(x_cat) # 192 -> 64        
        return out

class ENCResnetBlock_noidt(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ENCResnetBlock_noidt, self).__init__()
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

        conv_block += [nn.Conv2d( dim * 2, dim * 2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim * 2),
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
        conv_block += [nn.Conv2d(dim * 2, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, enc2): #64/64/64
        #print('****')
        #print(x.shape)
        #print(enc2.shape)


        x_cat = torch.cat((x,enc2),1) # 64/64/64/
        #print(x_cat.shape)
        out = x + self.conv_block(x_cat) # 192 -> 64        
        return out

class ENCResnetBlock_res(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ENCResnetBlock_res, self).__init__()
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

        conv_block += [nn.Conv2d( dim * 2, dim * 2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim * 2),
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
        conv_block += [nn.Conv2d(dim * 2, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, enc1, enc2): #64/64/64
        enc_res = enc1 - enc2 #
        x_cat = torch.cat((x,enc_res),1) # 64/64/
        out = x + self.conv_block(x_cat) # 192 -> 64        
        return out
















#############################################

class HISTUnet3_Res2(nn.Module): 
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
        super(HISTUnet3_Res2, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        enc_nc = 64

        #Init
        self.block_init  = self.build_init_block(input_nc,enc_nc,padding_type, use_bias)
        #self.block_init2 = self.build_init_block(input_nc,enc_nc,padding_type, use_bias)

        #Enc
        self.HistUnetEnc = UnetEnc_deep_2(input_nc, 64, norm_layer)

        #Dec
        self.HistUnetDec2 = UnetDec_deep2_2(512 + 512 + enc_nc + enc_nc,  256, norm_layer)
        self.HistUnetDec3 = UnetDec_deep2_2(256 + 256 + enc_nc + enc_nc,  128, norm_layer)
        self.HistUnetDec4 = UnetDec_deep2_2(128 + 128 + enc_nc + enc_nc,   64, norm_layer)
        self.HistUnetDec5 = UnetDec_deep1_2( 64 +  64 + enc_nc + enc_nc,   64, norm_layer)

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
        print(mid_img1.size(), mid_img2.size(), mid_img3.size(), mid_img4.size(), mid_img5.size())

        # Dec
        out_img2 = self.HistUnetDec2(mid_img5, mid_img4, hist1_enc, hist2_enc, mid_img4.size(2), mid_img4.size(3)) # 
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc, mid_img3.size(2), mid_img3.size(3)) # 
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc, mid_img2.size(2), mid_img2.size(3)) # 
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc, mid_img1.size(2), mid_img1.size(3)) #
        print('out_img5: ', out_img5.size())

        ##### SMAE STRUCTURE
        res = out_img5 + self.block_init(input_img)
        x = self.ENC_Block1.forward(res, hist1_enc, hist2_enc)
        x = self.ENC_Block2.forward(res + x, hist1_enc, hist2_enc)
        out_img  = self.ENC_Block3.forward(res + x, hist1_enc, hist2_enc)
        
        # Out
        out_img1 = self.InterOut1(mid_img5)
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)

        # Residual
        out_img    = self.block_last(out_img + self.block_init(input_img))

        return out_img1, out_img2, out_img3, out_img4, out_img

class UnetEnc_deep_2(nn.Module):
    def __init__(self, input_channel, ngf, norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep_2, self).__init__()

        self.netEnc1 = UnetEnc_deep1_2(input_channel,ngf,norm_layer)
        self.netEnc2 = UnetEnc_deep2_2(ngf * 1, ngf * 2, norm_layer) #64, 128
        self.netEnc3 = UnetEnc_deep2_2(ngf * 2, ngf * 4, norm_layer) #128, 256
        self.netEnc4 = UnetEnc_deep2_2(ngf * 4, ngf * 8, norm_layer) #256, 512
        self.netEnc5 = UnetEnc_deep2_2(ngf * 8, ngf * 8, norm_layer) #512, 512

    def forward(self, input):
        output1 = self.netEnc1.forward(input)  # 256/256/64 -> 128/128/64
        output2 = self.netEnc2.forward(output1) # 128/128/64 -> 64/64/128
        output3 = self.netEnc3.forward(output2) # 64/64/128 -> 32/32/256
        output4 = self.netEnc4.forward(output3) # 32/32/256 -> 16/16/512
        output5 = self.netEnc5.forward(output4) # 16/16/512 -> 8/8/512

        return output1, output2, output3, output4, output5

class UnetEnc_deep1_2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep1_2, self).__init__()
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


class UnetEnc_deep2_2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetEnc_deep2_2, self).__init__()

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

class UnetDec_deep2_2(nn.Module):
    def __init__(self, input_nc, output_nc,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep2_2, self).__init__()

        self.block = self.build_block(input_nc,output_nc, norm_layer=norm_layer)


    def build_block(self,input_nc,output_nc,norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        block_full = [  
                        # 1
                        nn.ReLU(True),
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

        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size = (target_size1, target_size2), mode = 'bilinear')
        enc1 = F.upsample(enc1,  size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2,  size = (target_size1, target_size2), mode = 'bilinear')


        #print('FollowMe')
        #print(input1.size()) # 1/512/8/8
        #print(input2.size()) # 1/512/8/8
        #print(enc1.size())   # 1/ 72/8/8
        #print(enc2.size())   # 1/ 72/8/8


        out = self.block(torch.cat([input1, input2, enc1, enc2],1))

        return out

class UnetDec_deep1_2(nn.Module):
    def __init__(self, input_nc, output_nc,use_tanh,norm_layer=nn.BatchNorm2d):
        super(UnetDec_deep1_2, self).__init__()

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

        #if use_tanh:
        #    block_full += [nn.Tanh()]

        return nn.Sequential(*block_full)

    def forward(self, input1, input2, enc1, enc2, target_size1, target_size2):
        #enc1 = enc1.repeat(1,1,input1.size(2),input1.size(3))
        #enc2 = enc2.repeat(1,1,input2.size(2),input2.size(3))
        input1=F.upsample(input1,size= (target_size1, target_size2), mode = 'bilinear')
        enc1 = F.upsample(enc1, size = (target_size1, target_size2), mode = 'bilinear')
        enc2 = F.upsample(enc2, size = (target_size1, target_size2), mode = 'bilinear')


        out = self.block(torch.cat([input1, input2, enc1, enc2],1))
        return out

