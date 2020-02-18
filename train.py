import time
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from tensorboardX import SummaryWriter
from util import util

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)

summary_dir = os.path.join(opt.checkpoints_dir, 'logs')
# util.remove_file_end_with(summary_dir, '*.cglabmark3')
util.mkdirs([summary_dir])
summary = SummaryWriter(summary_dir)

total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        # visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            # save_result = total_steps % opt.update_html_freq == 0
            # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            for key, val in model.get_current_visuals().items():
                summary.add_image('image/{}'.format(key), val, total_steps)

        if total_steps % opt.print_freq == 0:
            for key, val in model.get_current_errors().items():
                summary.add_scalar('loss/{}'.format(key), val, total_steps)
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # if opt.display_id > 0:
                # visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        print('[', epoch, ']', i, ' ', opt.name)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    # if epoch % opt.delete_log_epoch_freq == 0:
    #     util.remove_file_end_with(summary_dir, '*.cglabmark3')

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
