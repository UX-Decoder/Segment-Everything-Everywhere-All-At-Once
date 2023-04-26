_lang_encoders = {}


def register_lang_encoder(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]

    _lang_encoders[model_name] = fn

    return fn


def lang_encoders(model_name):
    return _lang_encoders[model_name]


def is_lang_encoder(model_name):
    return model_name in _lang_encoders
