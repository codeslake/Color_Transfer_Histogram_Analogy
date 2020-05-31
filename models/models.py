
def create_model(opt):
    from .colorhistogram_model import ColorHistogram_Model
    model = ColorHistogram_Model()        
    model.initialize(opt)
    return model
