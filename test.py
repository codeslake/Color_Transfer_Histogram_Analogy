import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import torchvision.utils as vutils
from util import util
from math import log10


# from util.visualizer import Visualizer
# from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
print(opt.results_dir)
print(opt.name)
opt.is_psnr = True
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(dataset)
print("TEST_OUT")
summary_dir = os.path.join(opt.checkpoints_dir, 'result_{}'.format(opt.test_dir), opt.mode)
util.mkdirs([summary_dir])
print(summary_dir)

j = 0
avg_psnr = 0.
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input_test(data)
    loss1, loss2 =  model.test()

    # if opt.is_psnr:
    psnr1 = 10 * log10(1 / loss1)
    psnr2 = 10 * log10(1 / loss2)
    psnr = (psnr1 + psnr2)/2.
    print('\nimage: ', i, '/', len(dataset), ': ', psnr)
    avg_psnr += psnr

    visuals = model.get_current_visuals_test()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    for key, val in visuals.items():
        print(key, ': ', val.min(), ', ', val.max())
        vutils.save_image(val, '{}/{}_{}.png'.format(summary_dir, i, key), nrow=1, padding = 0, normalize = False)
    j += 1

# if opt.is_psnr:
print(avg_psnr / j)

# webpage.save()
