# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import scipy.signal
import scipy.optimize
from typing import Tuple

import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act

from .networks_stylegan3 import FullyConnectedLayer, MappingNetwork

from util.utilgan import fix_size, multimask

#---------------------------------------------------------------------------
# Batched version of make_transform
def make_transform_batch(shifts, angles, scales, invert=False):
    """
    shifts: tensor-like of shape [B, 2]
    angles: tensor-like of shape [B] (in degrees)
    scales: tensor-like that should contain 2 values per sample.
            It can have shape [B], [B,1], [B,2], or higher dims; in the latter case, the first two values are used.
    Returns:
      transforms: [B, 3, 3]
    """
    # 入力を torch.Tensor に変換（dtype は float32）
    shifts = torch.as_tensor(shifts, dtype=torch.float32)
    angles = torch.as_tensor(angles, dtype=torch.float32)
    scales = torch.as_tensor(scales, dtype=torch.float32)

    # shifts が 1 次元の場合、[B] -> [B,2] にリシェイプ
    if shifts.dim() == 1:
        shifts = shifts.view(-1, 2)
    # scales の次元をバッチサイズ以外にまとめる
    if scales.dim() > 2:
        scales = scales.contiguous().view(scales.size(0), -1)
    elif scales.dim() == 1:
        scales = scales.view(-1, 1)
    # 各サンプルで 2 つの値がなければ、2 つに拡張；余分な次元があれば先頭2個を採用
    if scales.size(1) < 2:
        scales = scales.repeat(1, 2)
    elif scales.size(1) > 2:
        scales = scales[:, :2]

    B = shifts.shape[0]
    device = shifts.device
    # 単位行列を作成し、clone() で独立したメモリにする
    m = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3).clone()  # [B,3,3]

    pi = np.pi
    s = torch.sin(angles / 360.0 * pi * 2)  # [B]
    c = torch.cos(angles / 360.0 * pi * 2)  # [B]
    sy = scales[:, 0]
    sx = scales[:, 1]
    m[:, 0, 0] = sx * c
    m[:, 0, 1] = sx * s
    m[:, 0, 2] = shifts[:, 1]
    m[:, 1, 0] = -sy * s
    m[:, 1, 1] = sy * c
    m[:, 1, 2] = shifts[:, 0]
    if invert:
        m = torch.inverse(m)
    return m



#---------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                  # [batch_size, in_channels, H, W]
    w,                  # [out_channels, in_channels, kh, kw]
    s,                  # [batch_size, in_channels]
    latmask,                      # mask for split-frame latents blending
    countHW         = [1,1],      # frame split count by height,width
    splitfine       = 0.,         # frame split edge fineness
    splitmax        = None,       # max count of latents for frame splits
    size            = None,       # custom size
    scale_type      = None,       # scaling type
    demodulate  = True,
    padding     = 0,
    input_gain  = None,
):
    with misc.suppress_tracer_warnings():
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    misc.assert_shape(s, [batch_size, in_channels])

    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    w = w.unsqueeze(0)  # [1, O, I, kh, kw]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [B, O, I, kh, kw]

    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt()
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)

    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])

    if size is not None:
        x = fix_size(x, [s + padding for s in size], scale_type)
    if countHW != [1,1] or latmask is not None:
        x = multimask(x, x.shape[-2:], latmask, countHW, splitfine, splitmax)

    return x

#---------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])  # [channels, 2]
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5  # [channels]

        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
        self.register_buffer('transform', torch.eye(3, 3))
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w, trans_param=None):
        """
        w: [B, w_dim]
        trans_param: either None or a tuple (shifts, angles, scales) OR a list of such tuples for each sample.
            - shifts: [B, 2]
            - angles: [B]        (in degrees)
            - scales: [B, 2]
        """
        B = w.shape[0]
        if trans_param is None:
            transforms = self.transform.unsqueeze(0).expand(B, -1, -1)
        else:
            # if trans_param is provided as a list (or tuple) of tuples, stack them along batch dimension.
            if isinstance(trans_param, (list, tuple)) and isinstance(trans_param[0], (list, tuple)):
                shifts = torch.stack([torch.as_tensor(tp[0], device=w.device) for tp in trans_param], dim=0)
                angles = torch.stack([torch.as_tensor(tp[1], device=w.device) for tp in trans_param], dim=0)
                scales = torch.stack([torch.as_tensor(tp[2], device=w.device) for tp in trans_param], dim=0)
                trans_param = (shifts, angles, scales)
            # Now assume trans_param is a tuple of tensors of shape [B, ...]
            shifts, angles, scales = trans_param
            transforms = make_transform_batch(shifts, angles, scales, invert=True)
        # Expand freqs and phases to batch dimension.
        freqs = self.freqs.unsqueeze(0).expand(B, -1, -1)      # [B, channels, 2]
        phases = self.phases.unsqueeze(0).expand(B, -1)          # [B, channels]

        # Batched affine transformation.
        t = self.affine(w)   # [B, 4]
        t = t / t[:, :2].norm(dim=1, keepdim=True)  # Normalize rotation components.

        # Compute batched rotation and translation matrices.
        m_r = torch.eye(3, device=w.device).unsqueeze(0).expand(B, -1, -1)
        m_r[:, 0, 0] = t[:, 0]
        m_r[:, 0, 1] = -t[:, 1]
        m_r[:, 1, 0] = t[:, 1]
        m_r[:, 1, 1] = t[:, 0]
        m_t = torch.eye(3, device=w.device).unsqueeze(0).expand(B, -1, -1)
        m_t[:, 0, 2] = -t[:, 2]
        m_t[:, 1, 2] = -t[:, 3]
        transform_ = m_r.bmm(m_t).bmm(transforms)  # [B, 3, 3]

        # Transform frequencies.
        phases_ = phases + torch.bmm(freqs, transform_[:, :2, 2].unsqueeze(2)).squeeze(2)  # [B, channels]
        freqs_ = torch.bmm(freqs, transform_[:, :2, :2])  # [B, channels, 2]

        amplitudes = (1 - (freqs_.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid (same for each sample in the batch).
        theta = torch.eye(2, 3, device=w.device).unsqueeze(0).expand(B, -1, -1)
        theta[:, 0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[:, 1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta, [B, 1, int(self.size[1]), int(self.size[0])], align_corners=False)

        # Compute Fourier features in batch.
        x = torch.matmul(grids.unsqueeze(3), freqs_.permute(0, 2, 1).unsqueeze(1))
        x = x.squeeze(3)  # [B, H, W, channels]
        x = x + phases_.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)
        weight = self.weight / np.sqrt(self.channels)
        x = torch.matmul(x, weight.t())
        x = x.permute(0, 3, 1, 2)  # [B, channels, H, W]
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

#---------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,
        is_torgb,
        is_critically_sampled,
        use_fp16,
        in_channels,
        out_channels,
        in_size,
        out_size,
        in_sampling_rate,
        out_sampling_rate,
        in_cutoff,
        out_cutoff,
        in_half_width,
        out_half_width,
        countHW         = [1,1],
        splitfine       = 0.,
        splitmax        = None,
        size            = None,
        scale_type      = None,
        conv_kernel         = 3,
        filter_size         = 6,
        lrelu_upsampling    = 2,
        use_radial_filters  = False,
        conv_clamp          = 256,
        magnitude_ema_beta  = 0.999,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.countHW = countHW
        self.splitfine = splitfine
        self.splitmax = splitmax
        self.size = size
        self.scale_type = scale_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        self.affine = FullyConnectedLayer(w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        pad_total = (self.out_size - 1) * self.down_factor + 1
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor
        pad_total += self.up_taps + self.down_taps - 2
        pad_lo = (pad_total + self.up_factor) // 2
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, latmask, noise_mode='random', force_fp32=False, update_emas=False):
        misc.assert_shape(w, [x.shape[0], self.w_dim])
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            latmask=latmask, countHW=self.countHW, splitfine=self.splitfine, splitmax=self.splitmax, size=self.size, scale_type=self.scale_type,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        x = x.to(torch.float32)
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1
        if numtaps == 1:
            return None
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#---------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,
        img_resolution,
        img_channels,
        size            = None,
        scale_type      = None,
        channel_base        = 32768,
        channel_max         = 512,
        num_layers          = 14,
        num_critical        = 2,
        first_cutoff        = 2,
        first_stopband      = 2**2.1,
        last_stopband_rel   = 2**0.3,
        margin_size         = 10,
        output_scale        = 0.25,
        num_fp16_res        = 4,
        verbose         = False,
        **layer_kwargs,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.size = size
        self.scale_type = scale_type
        self.res_log2 = int(np.log2(img_resolution))
        self.channel_base = channel_base
        self.channel_max = channel_max
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res

        last_cutoff = self.img_resolution / 2
        last_stopband = last_cutoff * last_stopband_rel
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents

        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution))))
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        if size is None:
            sizes_custom = [None for i in range(len(sizes))]
        else:
            scale = np.array(size) / self.img_resolution
            sizes_custom = [[int(si) + self.margin_size * 2 for si in list(s * scale)] for s in sampling_rates]

        self.input = SynthesisInput(
            w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                size=sizes_custom[prev], scale_type=scale_type,
                in_channels=int(channels[prev]), out_channels=int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f"L{idx}_{layer.out_size[0]}_{layer.out_channels}"
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, latmask, trans_param=None, dconst=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)
        x = self.input(ws[0], trans_param=trans_param)
        if dconst is not None:
            x = x + dconst
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, latmask, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        if self.size is not None:
            if 'side' in self.scale_type.lower():
                x = x[:, :, :self.size[0], :self.size[1]]
            else:
                x = x[:, :, self.margin_size:self.size[0]+self.margin_size, self.margin_size:self.size[1]+self.margin_size]
        x = x.to(torch.float32)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
            f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])

#---------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        mapping_kwargs = {},
        **synthesis_kwargs,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.res = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, latmask, trans_param, dconst, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, latmask, trans_param, dconst, update_emas=update_emas, **synthesis_kwargs)
        return img
