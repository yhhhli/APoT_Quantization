import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# this function construct an additive pot quantization levels set, with clipping threshold = 1,
def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, power=True):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        # xout = (xhard - x).detach() + x
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            if power:
                input_q = power_quant(input_abs, grids).mul(sign)  # project to Q^a(alpha, B)
            else:
                input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        self.power = power if w_bit>2 else False
        self.grids = build_power_value(self.w_bit, additive=True)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, power=self.power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)      # weights normalization
            weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q


def act_quantization(b, grid, power=True):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)
            if power:
                input_q = power_quant(input_c, grid)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit = 4
        self.weight_quant = weight_quantize_fn(w_bit=self.bit, power=True)
        self.act_grid = build_power_value(self.bit, additive=True)
        self.act_alq = act_quantization(self.bit, self.act_grid, power=True)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        x = self.act_alq(x, self.act_alpha)
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)
