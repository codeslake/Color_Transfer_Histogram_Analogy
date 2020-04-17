
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'color_histogram':
        from .colorhistogram_model import ColorHistogram_Model
        model = ColorHistogram_Model()        
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    return model
