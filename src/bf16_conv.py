# Copyright (C) 2026 Silvet.icu
# SPDX-License-Identifier: AGPL-3.0-or-later

"""bf16 convolution on sm_70, where cuDNN has no bf16 kernel.

Usage:
    bf16_conv.install()  /  bf16_conv.uninstall()
    bf16_conv.set_mode('fp32' | 'fp16_scaled' | 'fp16')
"""
import math

import torch
import torch.nn.functional as F

__all__ = ['install', 'uninstall', 'is_installed', 'set_mode', 'get_mode', 'fallbacks',
           'skipped']

# name -> position of stride and groups among the positionals after bias
_SPECS = {
    'conv1d': (0, 3), 'conv2d': (0, 3), 'conv3d': (0, 3),
    'conv_transpose1d': (0, 3), 'conv_transpose2d': (0, 3), 'conv_transpose3d': (0, 3),
}
_ORIG = {name: getattr(F, name) for name in _SPECS}
_installed = False

# How bf16 enters cuDNN.
_MODES = ('fp32', 'fp16_scaled', 'fp16')
_mode = 'fp32'

# Operand magnitude band fp16 stores exactly.
_MAX_OPERAND = 65504.0
_MIN_OPERAND = 2.0 ** -8

# Multiply-accumulate gates: below PASS keep bf16, below LANE use fp32.
_MIN_MACS_PASS = 4.0e8
_MIN_MACS_LANE = 4.0e9

_fallbacks = 0
_skipped = 0


def set_mode(mode):
    """Pick how bf16 convolution runs: 'fp32', 'fp16_scaled' or 'fp16'."""
    global _mode
    if mode not in _MODES:
        raise ValueError('mode must be one of %s' % (_MODES,))
    _mode = mode


def get_mode():
    return _mode


def fallbacks():
    """How many calls fell back to fp32 because fp16 could not hold them."""
    return _fallbacks


def skipped():
    """How many calls were left to the native bf16 kernel as too small to gain."""
    return _skipped


def _pow2_scale(mx, peak=1.0):
    """Power of two that maps |max| into [peak, 2*peak)."""
    if not (mx > 0.0) or not math.isfinite(mx):
        return 1.0
    return 2.0 ** math.floor(math.log2(peak / mx))


def _extremes(t):
    """(min, max) of a tensor, as device scalars."""
    return torch.aminmax(t)


def _magnitude(lo, hi):
    """Largest |.| covered by the extremes (lo, hi)."""
    return max(-lo, hi)


def _prod(t):
    """Product of an int or a tuple of ints."""
    t = (t,) if isinstance(t, int) else tuple(t)
    out = 1
    for x in t:
        out *= x
    return out


def _terms(weight):
    """Multiplications behind one output element, per group."""
    n = weight.shape[1]
    for d in weight.shape[2:]:
        n *= d
    return n


def _peak(weight):
    """Largest power-of-two operand peak that keeps the fp16 sum under 65504."""
    return 2.0 ** math.floor(math.log2(math.sqrt(65504.0 / (4.0 * _terms(weight)))))


def _pick(args, kwargs, i, name, default):
    """Argument i from a positional list, else kwargs[name], else default."""
    if len(args) > i:
        return args[i]
    v = kwargs.get(name)
    return default if v is None else v


def _macs(spec, name, input, weight, args, kwargs):
    """Multiply-accumulate count of the conv."""
    stride = _pick(args, kwargs, spec[0], 'stride', 1)
    groups = _pick(args, kwargs, spec[1], 'groups', 1)
    per = input.numel() // (input.shape[0] * input.shape[1])   # spatial, one channel
    k = _terms(weight)
    if name.startswith('conv_transpose'):
        return input.shape[0] * weight.shape[1] * groups * per * _prod(stride) * k
    return input.shape[0] * weight.shape[0] * (per // _prod(stride)) * k


def _fp32(name, input, weight, bias, args, kwargs):
    """Conv with fp32 operands, output back in the input dtype."""
    return _ORIG[name](input.float(), weight.float(),
                       None if bias is None else bias.float(),
                       *args, **kwargs).to(input.dtype)


def _restore(out, bias, scale, spatial_dims):
    """Divide out the operand scale and add the bias, both in fp32."""
    out = out.float().mul_(1.0 / scale)
    if bias is not None:
        out.add_(bias.view(-1, *([1] * spatial_dims)))
    return out.to(torch.bfloat16)


def _direct(name, input, weight, bias, args, kwargs, spatial_dims):
    """Conv on plain fp16 casts."""
    out = _ORIG[name](input.half(), weight.half(), None, *args, **kwargs)
    return _restore(out, bias, 1.0, spatial_dims)


def _scaled(spec, name, input, weight, bias, args, kwargs):
    """Conv on fp16 casts, with power-of-two operand scales where needed."""
    global _fallbacks
    extremes = torch.stack([v for t in (input, weight) for v in _extremes(t)]).tolist()
    mxi, mxw = _magnitude(extremes[0], extremes[1]), _magnitude(extremes[2], extremes[3])
    if not (math.isfinite(mxi) and math.isfinite(mxw)):
        _fallbacks += 1
        return _fp32(name, input, weight, bias, args, kwargs)

    dims = input.dim() - 2
    terms = _terms(weight)
    if (_MIN_OPERAND <= mxi <= _MAX_OPERAND and _MIN_OPERAND <= mxw <= _MAX_OPERAND
            and mxi * mxw * terms <= _MAX_OPERAND):
        return _direct(name, input, weight, bias, args, kwargs, dims)

    peak = _peak(weight)
    si, sw = _pow2_scale(mxi, peak), _pow2_scale(mxw, peak)
    clamped = si < _MIN_OPERAND or sw < _MIN_OPERAND
    si, sw = max(si, _MIN_OPERAND), max(sw, _MIN_OPERAND)
    out = _ORIG[name]((input * si).half(), (weight * sw).half(), None, *args, **kwargs)
    if clamped and not torch.isfinite(out).all():
        _fallbacks += 1
        return _fp32(name, input, weight, bias, args, kwargs)
    return _restore(out, bias, si * sw, dims)


def _make(spec, name):
    """Build one patched conv: bf16 routed by size and mode, other dtypes as-is."""
    def conv(input, weight, bias=None, *args, **kwargs):
        if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return _ORIG[name](input, weight, bias, *args, **kwargs)
        global _skipped
        macs = _macs(spec, name, input, weight, args, kwargs)
        if macs < _MIN_MACS_PASS:
            _skipped += 1
            return _ORIG[name](input, weight, bias, *args, **kwargs)
        if _mode == 'fp32':
            return _fp32(name, input, weight, bias, args, kwargs)
        if _mode == 'fp16':
            return _ORIG[name](input.half(), weight.half(),
                               None if bias is None else bias.half(),
                               *args, **kwargs).to(input.dtype)
        if macs < _MIN_MACS_LANE:
            _skipped += 1
            return _fp32(name, input, weight, bias, args, kwargs)
        return _scaled(spec, name, input, weight, bias, args, kwargs)

    conv.__name__ = name
    conv.__doc__ = 'bf16 detour for F.%s.' % name
    return conv


def install():
    """Patch F.conv1d/2d/3d and their transposed siblings. Idempotent."""
    global _installed
    if not _installed:
        for name, spec in _SPECS.items():
            setattr(F, name, _make(spec, name))
        _installed = True


def uninstall():
    """Put the original conv functions back."""
    global _installed
    for name in _SPECS:
        setattr(F, name, _ORIG[name])
    _installed = False


def is_installed():
    return _installed
