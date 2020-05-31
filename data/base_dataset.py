import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy
from skimage import color

#Added
import time
from math import exp
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform_filter_sat(opt):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"random",1,1,1,1,time.time())))
    transform_list.append(transforms.Lambda(lambda img: RGB2LAB(numpy.array(img))))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_filter_red(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"original",0.8,1.3,1,1,time.time())))
    transform_list.append(transforms.Lambda(lambda img: RGB2LAB(numpy.array(img))))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_filter_blue(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"original",0.8,1,1,1.3,time.time())))
    transform_list.append(transforms.Lambda(lambda img: RGB2LAB(numpy.array(img))))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def Filter_syn(I,mode,w,x,y,z,start_time):

    # Preprocessing
    this_time=time.time()
    elapsed_time = this_time - start_time

    if mode == "random":
        w = ((elapsed_time*1000)%1000)*(1.5)/1000 # above 1
        if w > 1.3:
            w = 1.3
        if w <0.4:
            w = 0.4


    # Change Saturation
    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * (w)
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0

    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    # Change Color
    rgb[:,:,0] = rgb[:,:,0] * x
    rgb[:,:,1] = rgb[:,:,1] * y
    rgb[:,:,2] = rgb[:,:,2] * z


    # Post-processing
    rgb[:,:,0] = numpy.clip(rgb[:,:,0],0,1)
    rgb[:,:,1] = numpy.clip(rgb[:,:,1],0,1)
    rgb[:,:,2] = numpy.clip(rgb[:,:,2],0,1)

    return rgb

def no_transform(opt):
    transform_list = []
    #osize = [opt.loadSize, opt.loadSize]
    #transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    #transform_list.append(transforms.CenterCrop(opt.fineSize))
    #transform_list.append(transforms.RandomCrop(opt.fineSize))
    #transform_list.append(transforms.Resize(interpolation=0.5))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_hsv(opt):
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: RGB2HSV(numpy.array(img))))
    transform_list.append(transforms.Lambda(lambda img: (numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_lab(opt): # Now we are using
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: RGB2LAB(numpy.array(img))))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_hueshiftlab(opt): # Now we are using
    start_time = time.time()

    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV_shift_LAB(numpy.array(img),start_time)))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_hueshiftlab2(opt): # Now we are using
    start_time = time.time()

    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV_shift_LAB2(numpy.array(img),start_time)))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



def get_transform_hueshiftlab_randomcrop(opt): # Now we are using
    start_time = time.time()

    transform_list = []
    transform_list.append(transforms.Lambda(lambda img, i, j, h, w: TF.crop(img, i, j, h, w)))
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV_shift_LAB(numpy.array(img),start_time)))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_hueshiftlab2_randomcrop(opt): # Now we are using
    start_time = time.time()

    transform_list = []
    transform_list.append(transforms.Lambda(lambda img, i, j, h, w: TF.crop(img, i, j, h, w)))
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV_shift_LAB2(numpy.array(img),start_time)))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)









def get_transform_saturation(opt):
    transform_list = []

    #transform_list.append(transforms.Resize(interpolation=0.5))
    #osize = [opt.loadSize, opt.loadSize]
    #transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    #transform_list.append(transforms.CenterCrop(opt.fineSize))
    #transform_list.append(transforms.RandomCrop(opt.fineSize))

    start_time = time.time()
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV2RGB(numpy.array(img),"random",start_time)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_grayish(opt):
    transform_list = []

    start_time = time.time()
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV2RGB_Gray(numpy.array(img),"random",start_time)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
#def get_transform_lab2(opt):
#    transform_list = []
#
#    transform_list.append(transforms.Lambda(lambda img: RGB2HSV(numpy.array(img))))
#    transform_list.append(transforms.Lambda(lambda img: (numpy.array(img))))
#
#    transform_list += [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
#    return transforms.Compose(transform_list)


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def RGB2LAB(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0) #* 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -127 to 127
    #l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    #a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    #b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return numpy.dstack([l, a, b])

def LAB2RGB(I):
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] / 255.0 * (94.4781222765 + 107.857300207) - 107.857300207

    rgb = color.lab2rgb(numpy.dstack([l, a, b]).astype(numpy.float64))
    return rgb

def RGB2HSV(I):
    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    return numpy.dstack([h, s, v])

def RGB2HSV_shift_LAB(I,start_time): # shift value0 ~1
    this_time=time.time()
    elapsed_time=this_time-start_time
    shift = ((elapsed_time*1000)%1000)*(1.0)/1000 # above 1
    shift2 = ((elapsed_time*10000)%10000)*(1.2)/10000 # above 1 # Added at 'colorhistogram_noGAN_lr00005_lab_hueshift_histenc_histloss_satadded' 
    if shift2 < 0.3:
        shift2 = 0.3
    # Get Original L in LAB, shift H in HSV

    # Get Original LAB
    lab_original = color.rgb2lab(I)
    l_original = (lab_original[:, :, 0] / 100.0)
    
    # Shift HSV
    hsv = color.rgb2hsv(I)
    h = ((hsv[:, :, 0] + shift))
    s = (hsv[:, :, 1]) * shift2
    v = (hsv[:, :, 2])
    hsv2 = color.hsv2rgb(numpy.dstack([h, s, v]).astype(numpy.float64))

    # Merge (Original LAB, Shifted HSV)
    lab = color.rgb2lab(hsv2)
    l = l_original
    #l = (lab[:, :, 0] / 100.0)
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -127 to 127

    return numpy.dstack([l, a, b])

def RGB2HSV_shift_LAB2(I,start_time): # shift value0 ~1
    this_time=time.time()
    elapsed_time=this_time-start_time
    shift = ((elapsed_time*10000)%10000)*(1.0)/10000 # above 1
    shift2 = ((elapsed_time*10000)%10000)*(1.2)/10000 # above 1 # Added at 'colorhistogram_noGAN_lr00005_lab_hueshift_histenc_histloss_satadded'
    if shift2 < 0.3:
        shift2 = 0.3

    # Get Original L in LAB, shift H in HSV

    # Get Original LAB
    lab_original = color.rgb2lab(I)
    l_original = (lab_original[:, :, 0] / 100.0)
    
    # Shift HSV
    hsv = color.rgb2hsv(I)
    h = ((hsv[:, :, 0] + shift))
    s = (hsv[:, :, 1]) * shift2
    v = (hsv[:, :, 2])
    hsv2 = color.hsv2rgb(numpy.dstack([h, s, v]).astype(numpy.float64))

    # Merge (Original LAB, Shifted HSV)
    lab = color.rgb2lab(hsv2)
    l = l_original
    #l = (lab[:, :, 0] / 100.0)
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -127 to 127

    return numpy.dstack([l, a, b])

def RGB2HSV2RGB(I,mode,start_time):

    this_time=time.time()
    elapsed_time=this_time-start_time


    if mode == "original":
        alpha = 1
    elif mode == "random":
        ###### Gray Added
        alpha = ((elapsed_time*1000)%1000)*(1.5)/1000 # above 1

        if alpha > 1.0:
            alpha = 1.0
        if alpha <0.2:
            alpha = 0.2

    elif mode == "gray":
        alpha = 0.1
    elif mode == "decay":
        alpha = 1/(1+exp(0.0003*(elapsed_time-20000)))*0.9 + 0.1
    else:
        alpha = 1

    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * alpha
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0


    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    return rgb



def RGB2HSV2RGB_Gray(I,mode,start_time):


    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * 0.4
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0


    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    return rgb