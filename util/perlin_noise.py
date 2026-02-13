import torch
import torch.nn.functional as F


def generate_perlin_noise(
    shape, *, iterations=0, device='cpu'
):
    width = shape[-1]
    height = shape[-2]
    remaining_dims = shape[:-2]
    nvecs = 1
    for dim in remaining_dims:
        nvecs *= dim

    _seed = torch.randn(nvecs, 2, height, width, device=device)
    _noise = torch.zeros(nvecs, height, width, device=device)
    _buffer = torch.zeros(
        nvecs,
        height // 2,
        width // 2,
        device=device
    )
    c_width = 2
    c_height = 2
    _n = 0

    while (
            c_width < width and
            c_height < height and
            (_n < iterations or iterations == 0)
    ):
        _n += 1
        _step_buffer = _buffer[:, :c_height*2, :c_width*2].zero_()
        iterations -= 1
        _step_vecs = F.interpolate(
            _seed,
            size=(c_height + 1, c_width + 1),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)

        _step_vecs = _step_vecs / (_step_vecs.norm(dim=1, keepdim=True) + 1e-8)
        _up_vecs = _step_vecs[:, :-1, 1:]
        _left_vecs = _step_vecs[:, 1:, :-1]
        _down_vecs = _step_vecs[:, 1:, 1:]
        _right_vecs = _step_vecs[:, :-1, :-1]
        _step_buffer[:, 0::2, 0::2] += (_down_vecs * _right_vecs).sum(dim=-1)
        _step_buffer[:, 0::2, 1::2] += (_down_vecs * _left_vecs).sum(dim=-1)
        _step_buffer[:, 1::2, 0::2] += (_up_vecs * _right_vecs).sum(dim=-1)
        _step_buffer[:, 1::2, 1::2] += (_up_vecs * _left_vecs).sum(dim=-1)

        _noise += F.interpolate(
            _step_buffer.unsqueeze(1),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        c_width = c_width * 2
        c_height = c_height * 2
    if _n > 1:
        _noise = _noise / _n

    return _noise.reshape(*remaining_dims, height, width)
