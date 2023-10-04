from .xdecoder_head import *
from .build import *

def build_xdecoder_head(config, *args, **kwargs):
    model_name = config['MODEL']['HEAD']
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    body = model_entrypoints(model_name)(config, *args, **kwargs)
    return body