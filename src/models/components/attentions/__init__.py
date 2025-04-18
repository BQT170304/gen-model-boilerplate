from .attn import AttnBlock
from .attention import AttentionBlock
from .multi_head_attention import SABlock, CABlock
from .spatial_transfomer import SpatialTransformer

Attentions = {
    "Attn": AttnBlock, # test for cifar10
    "Attention": AttentionBlock,
    "SelfAttention": SABlock,
    "Transformer": SpatialTransformer,
    "CrossAttention": CABlock,
}


def init_attention(name):
    """Initializes attention"""
    avai_attentions = list(Attentions.keys())
    if name not in avai_attentions:
        raise ValueError('Invalid attention name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_attentions))
    return Attentions[name]


def get_all_attentions():
    return tuple(Attentions.values())