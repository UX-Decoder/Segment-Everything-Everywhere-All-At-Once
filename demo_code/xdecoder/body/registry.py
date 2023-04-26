_model_entrypoints = {}


def register_body(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _model_entrypoints[model_name] = fn
    return fn

def model_entrypoints(model_name):
    return _model_entrypoints[model_name]

def is_model(model_name):
    return model_name in _model_entrypoints