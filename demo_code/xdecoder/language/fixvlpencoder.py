from importlib.metadata import requires
import torch
import torch.nn as nn

from .registry import register_model
from .vlpencoder import LanguageEncoder

class FixLanguageEncoder(LanguageEncoder):

    def __init__(
        self,
        *args, **kwargs):
        super(FixLanguageEncoder, self).__init__(*args, **kwargs)
        self.logit_scale = nn.Parameter(torch.ones([]), requires_grad=False)

    @torch.no_grad()
    def get_text_embeddings(self, *args, **kwargs):
        return super().get_text_embeddings(*args, **kwargs)

    @torch.no_grad()
    def get_text_token_embeddings(self, *args, **kwargs):
        return super().get_text_token_embeddings(*args, **kwargs)

    @torch.no_grad()
    def forward_language(self, *args, **kwargs):
        return super().forward_language(*args, **kwargs)

    @torch.no_grad()
    def forward_language_token(self, *args, **kwargs):
        return super().forward_language_token(*args, **kwargs)


@register_model
def get_language_model(cfg, **kwargs):
    return FixLanguageEncoder(cfg)        