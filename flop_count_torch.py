import torch
import numpy as np

from ptflops import get_model_complexity_info

import importlib
import argparse

from options.test_options import TestOptions
from models.models import create_model

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : Number of params: {}'.format(model._get_name(), para))
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = {}
    for k,v in input.items():
        print(k)
        input_[k] = v.clone()
        input_[k].requires_grad_(requires_grad=False)

    PV = input_['PV']
    img = torch.cat((input_['I_prev'], input_['I_curr'], input_['I_next']), axis = 1)

    out_sizes = []

    print('#####################################################')
    mods = list(model.modules())
    for i in range(len(mods)):
        print(mods[i])
        print('#####################################################')

    print('')

    out_sizes.append(np.array(PV.size()))
    out_sizes.append(np.array(img.size()))
    for i in range(2, len(mods)):
        m = mods[i]
        # print('!!', m)
        if isinstance(m, torch.nn.ReLU):
            # if m.inplace:
            continue

        if isinstance(m, torch.nn.Sequential) or isinstance(m, torch.nn.ModuleList):
            continue

        if i == 2:
            out = m(PV)
            out_PV = out.clone()
        elif i == 7:
            out = m(img)
            out_img = out.clone()
        elif i == 11:
            out = m(torch.cat((out_PV, out_img), axis = 1))
            out_sizes.append(np.array(torch.cat((out_PV, out_img), axis = 1).size()))
        else:
            out = m(input_)

        print('!!!!!!!', m)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # print('Model {} : Number of intermedite variables without backward: {}'.format(model._get_name(), total_nums))
    # print('Model {} : Number of intermedite variables with backward: {}'.format(model._get_name(), total_nums*2))
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1024 ** 2))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

#forward(self, I_prev, I_curr, I_next, I_prev_deblurred, gt_prev, gt_curr, h = None, w = None, is_train = False):
def input_constructor(res):
    b, c, h, w = res[:]

    input_A = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()
    input_B = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()
    input_C = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()
    input_A_Map = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()
    input_B_Map = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()

    # BIM/PVD
    return {'A': input_A, 'B': input_B, 'C': input_C, 'A_Map': input_A_Map, 'B_Map': input_B_Map}

with torch.no_grad():

    net = model.network
    shape = (1, 3, 256, 256)
    shape = (1, 3, 720, 1280)

    # modelsize(net, input_constructor(shape), type_size = 4)
    # exit()
    # net(**input_constructor(shape))
    # print('')
    # net(**input_constructor(shape))
    # print('')
    # net(**input_constructor(shape))
    # print('')
    # net(**input_constructor(shape))
    # print('')
    # net(**input_constructor(shape))
    # print('')
    # net(**input_constructor(shape))
    # print('')
    # exit()


# BIM/PVD
with torch.no_grad():
    flops,params = get_model_complexity_info(net, (1, 3, 720, 1280), input_constructor = input_constructor, as_strings=False,print_per_layer_stat=True)
# print(torch.cuda.memory_summary())
print('{:<30}  {:<8} B'.format('Computational complexity (Flops): ', flops / 1000 ** 3 ))
print('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
