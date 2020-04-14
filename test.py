import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import torchvision.utils as vutils
from util import util
from math import log10


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
opt.is_psnr = True

summary_dir = os.path.join(opt.results_dir, 'result_{}'.format(opt.test_dir), opt.mode)
util.mkdirs([summary_dir])

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input_test(data)
    loss1, loss2 =  model.test()

    print('\nimage: ', i, '/', len(dataset), ': ', psnr)

    visuals = model.get_current_visuals_test()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    for key, val in visuals.items():
        print(key, ': ', val.min(), ', ', val.max())
        vutils.save_image(val, '{}/{}_{}.png'.format(summary_dir, i, key), nrow=1, padding = 0, normalize = False)
