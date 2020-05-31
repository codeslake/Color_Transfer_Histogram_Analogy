import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt.checkpoints_dir

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def load_network(self, network, network_label, epoch_label=None, ckpt_name=None):
        if ckpt_name is None:
            save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        elif ckpt_name is not None:
            save_filename = '%s_net_%s.pth' % (ckpt_name, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        network.load_state_dict(torch.load(save_path))
