from .vlpencoder import *
from .build import *

def build_language_encoder(config, **kwargs):
    model_name = config['MODEL']['TEXT']['ARCH']

    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, **kwargs)