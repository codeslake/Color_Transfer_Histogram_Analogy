
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'colorenhance':
        from .colorenhance_model import ColorEnhance_Model
        model = ColorEnhance_Model()
    elif opt.model == 'colorize':
        assert(opt.dataset_mode == 'aligned_seg')
        from .colorize_model import ColorizeModel
        model = ColorizeModel()
    elif opt.model == 'colorize_fcycle':
        assert(opt.dataset_mode == 'aligned_seg')
        from .colorize_fcycle_model import Colorize_fcycle_Model
        model = Colorize_fcycle_Model()
    elif opt.model == 'color_histogram':
        from .colorhistogram_model import ColorHistogram_Model
        model = ColorHistogram_Model()        
    elif opt.model == 'color_histogram_test':
        from .colorhistogram_model_test import ColorHistogram_Model
        model = ColorHistogram_Model()        
    elif opt.model == 'cvpr_model':
        from .cvpr_model import ColorHistogram_Model
        model = ColorHistogram_Model()        
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
