import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import torchvision.utils as vutils
from util import util


opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
opt.is_psnr = True

summary_dir = opt.results_dir
util.mkdirs([summary_dir])

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()

    print('\nimage: ', i, '/', len(dataset))

    visuals = model.get_current_visuals()
    print('%04d: process image... ' % (i))
    for key, val in visuals.items():
        vutils.save_image(val, '{}/{}_{}.png'.format(summary_dir, i, key), nrow=1, padding = 0, normalize = False)
