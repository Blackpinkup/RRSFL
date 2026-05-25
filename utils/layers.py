import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch import nn

try:
    from .amp_utils import process as cython_process
except Exception:
    cython_process = None


def _extract_amp(img_np):
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    return np.abs(fft)


def _mutate(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    h, w = a_src.shape[-2:]
    b = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))
    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1
    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    return np.fft.ifftshift(a_src, axes=(-2, -1))


def _normalize(src_img, amp_trg, L=0.1):
    fft_src = np.fft.fft2(src_img, axes=(-2, -1))
    amp_src = np.abs(fft_src)
    pha_src = np.angle(fft_src)
    amp_src = _mutate(amp_src, amp_trg, L=L)
    fft_src = amp_src * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src, axes=(-2, -1))
    return np.real(src_in_trg)


def python_process(x, running_amp, momentum, fix_amp):
    batch = x.shape[0]
    if not fix_amp:
        amp_list = np.zeros_like(x, dtype=np.float32)
        for idx in range(batch):
            amp_list[idx] = _extract_amp(x[idx])
        amp_avg = np.mean(amp_list, axis=0)
        if np.sum(running_amp) == 0:
            running_amp = amp_avg
        else:
            running_amp = running_amp * (1.0 - momentum) + amp_avg * momentum

    for idx in range(batch):
        x[idx] = _normalize(x[idx], running_amp[:3, ...], L=0)
    return x, running_amp


def process(x, running_amp, momentum, fix_amp):
    if cython_process is not None:
        try:
            return cython_process(x, running_amp, momentum, fix_amp)
        except TypeError:
            return python_process(x, running_amp, momentum, fix_amp)
    return python_process(x, running_amp, momentum, fix_amp)


class AmpNorm(nn.Module):
    def __init__(self, input_shape, momentum=0.1):
        super(AmpNorm, self).__init__()
        self.register_buffer("running_amp", torch.zeros(input_shape))
        self.momentum = momentum
        self.fix_amp = False

    def forward(self, x):
        device = x.device
        x_np = x.detach().cpu().numpy().astype(np.float32)
        running_amp_np = self.running_amp.detach().cpu().numpy().astype(np.float32)
        x_np, amp_np = process(x_np, running_amp_np, float(self.momentum), bool(self.fix_amp))
        amp = torch.from_numpy(amp_np).to(device=self.running_amp.device, dtype=self.running_amp.dtype)
        self.running_amp.copy_(amp)
        return torch.from_numpy(x_np).to(device=device, dtype=x.dtype)


if __name__ == "__main__":
    exit()
