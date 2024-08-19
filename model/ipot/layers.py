from functools import wraps
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


# Helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = dict()

    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result

    return cached_fn


# Helper classes
class PreNorm(nn.Module):
    def __init__(self, channel, fn, context_channel=None):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(channel)
        self.norm_context = nn.LayerNorm(context_channel) if exists(context_channel) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, channel, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, channel * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(channel * mult, channel)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self, query_channel, context_channel=None, output_channel=None,
            heads_num=8, heads_channel=64, dropout=0.):
        super(Attention, self).__init__()
        inner_channel = heads_channel * heads_num
        context_dim = default(context_channel, query_channel)
        output_dim = default(output_channel, query_channel)

        self.scale = heads_channel ** -0.5
        self.heads = heads_num

        self.to_q = nn.Linear(query_channel, inner_channel, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_channel * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_channel, output_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
