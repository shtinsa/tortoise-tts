import torch
import torch.nn as nn
import torch.nn.functional as F
import tvm
from tvm import relax
from tvm import nd
import numpy as np
import os

MAX_WAV_VALUE = 32768.0

class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions'''

    def __init__(
            self,
            cond_channels,
            conv_in_channels,
            conv_out_channels,
            conv_layers,
            conv_kernel_size=3,
            kpnet_hidden_channels=64,
            kpnet_conv_size=3,
            kpnet_dropout=0.0,
            kpnet_nonlinear_activation="LeakyReLU",
            kpnet_nonlinear_activation_params={"negative_slope": 0.1},
    ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int): number of layers
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers  # l_w
        kpnet_bias_channels = conv_out_channels * conv_layers  # l_b

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True)),
            getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_convs = nn.ModuleList()
        padding = (kpnet_conv_size - 1) // 2
        for _ in range(3):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Dropout(kpnet_dropout),
                    nn.utils.weight_norm(
                        nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding,
                                  bias=True)),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                    nn.utils.weight_norm(
                        nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding,
                                  bias=True)),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                )
            )
        self.kernel_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True))
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True))

    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        '''
        batch, _, cond_length = c.shape
        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            residual_conv.to(c.device)
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length,
        )
        bias = b.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_out_channels,
            cond_length,
        )

        return kernels, bias

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv[0])
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)
        for block in self.residual_convs:
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])


class LVCBlock(torch.nn.Module):
    '''the location-variable convolutions'''

    def __init__(
            self,
            in_channels,
            cond_channels,
            stride,
            dilations=[1, 3, 9, 27],
            lReLU_slope=0.2,
            conv_kernel_size=3,
            cond_hop_length=256,
            kpnet_hidden_channels=64,
            kpnet_conv_size=3,
            kpnet_dropout=0.0,
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope}
        )

        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(nn.ConvTranspose1d(in_channels, in_channels, 2 * stride, stride=stride,
                                                    padding=stride // 2 + stride % 2, output_padding=stride % 2)),
        )

        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.weight_norm(nn.Conv1d(in_channels, in_channels, conv_kernel_size,
                                                   padding=dilation * (conv_kernel_size - 1) // 2, dilation=dilation)),
                    nn.LeakyReLU(lReLU_slope),
                )
            )

    def forward(self, x, c):
        ''' forward propagation of the location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        '''
        _, in_channels, _ = x.shape  # (B, c_g, L')

        x = self.convt_pre(x)  # (B, c_g, stride * L')
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)  # (B, c_g, stride * L')

            k = kernels[:, i, :, :, :, :]  # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]  # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(output, k, b,
                                                        hop_size=self.cond_hop_length)  # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[:, :in_channels, :]) * torch.tanh(
                output[:, in_channels:, :])  # (B, c_g, stride * L'): GAU

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation,
                     dilation)  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self):
        self.kernel_predictor.remove_weight_norm()
        nn.utils.remove_weight_norm(self.convt_pre[1])
        for block in self.conv_blocks:
            nn.utils.remove_weight_norm(block[1])


class UnivNetGenerator(nn.Module):
    """
    UnivNet Generator
    
    Originally from https://github.com/mindslab-ai/univnet/blob/master/model/generator.py.
    """

    def __init__(self, noise_dim=64, channel_size=32, dilations=[1,3,9,27], strides=[8,8,4], lReLU_slope=.2, kpnet_conv_size=3,
                 # Below are MEL configurations options that this generator requires.
                 hop_length=256, n_mel_channels=100):
        super(UnivNetGenerator, self).__init__()
        self.mel_channel = n_mel_channels
        self.noise_dim = noise_dim
        self.hop_length = hop_length
        channel_size = channel_size
        kpnet_conv_size = kpnet_conv_size
        if 'USE_TVM_MODEL'  in os.environ:
            models_path = os.environ['TVM_MODELS_DIR']
            target = tvm.target.Target("nvidia/geforce-rtx-3070", host="llvm")
            self.dev_ = tvm.device(target.kind.name, 0)
            self.MAX_PAD = 12
            self.CHUNK = 160
            self.HOP_LENGTH = 256
            self.GROUP_SIZE = (self.CHUNK + 2 * self.MAX_PAD)
            lib = tvm.runtime.load_module(f'{models_path}/univnet.so')
            self.vm_ = relax.VirtualMachine(lib, self.dev_)
            self.buffer_ = np.zeros((1, 100, self.GROUP_SIZE), dtype=np.float16)
            self.noise_buffer = np.zeros((1, 64, self.GROUP_SIZE), dtype=np.float16)
        else:
            self.res_stack = nn.ModuleList()
            hop_length = 1
            for stride in strides:
                hop_length = stride * hop_length
                self.res_stack.append(
                    LVCBlock(
                        channel_size,
                        n_mel_channels,
                        stride=stride,
                        dilations=dilations,
                        lReLU_slope=lReLU_slope,
                        cond_hop_length=hop_length,
                        kpnet_conv_size=kpnet_conv_size
                    )
                )

            self.conv_pre = \
                nn.utils.weight_norm(nn.Conv1d(noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

            self.conv_post = nn.Sequential(
                nn.LeakyReLU(lReLU_slope),
                nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
                nn.Tanh(),
            )

    def forward(self, c, z):
        '''
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)
            z (Tensor): the noise sequence (batch, noise_dim, in_length)

        '''
        z = self.conv_pre(z)  # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)  # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)  # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(UnivNetGenerator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()
            
    def run_tvm(self, c, z):
        x_t = c

        print("!!!!!!!!!!!!!")

        output_arr = np.zeros((1, self.HOP_LENGTH * x_t.shape[2]), dtype=np.float16)
        left = 184#min(self.CHUNK + self.MAX_PAD, x_t.shape[1])
        xx = x_t.to(torch.float16).cpu().numpy()
        zz = z.to(torch.float16).cpu().numpy()
        self.buffer_[0, :, 0:left] = xx[0, :, :left]
        self.noise_buffer[0, :, 0:left] = zz[0, :, :left]

        i_g = nd.array(self.noise_buffer, device=self.dev_)
        i_x = nd.array(self.buffer_, device=self.dev_)
        outs = []
        # first iter
        output_b = self.vm_["vocoder_184"](i_x, i_g)
        
        real_len = min(x_t.shape[2], self.CHUNK)
        outs.append([output_b.numpy(), real_len * self.HOP_LENGTH, real_len])
        for pos in range(self.CHUNK, x_t.shape[2], self.CHUNK):
            left = min(self.MAX_PAD + x_t.shape[2] - pos, self.GROUP_SIZE)
            self.buffer_[0, :, 0:left] = xx[0, :, pos - self.MAX_PAD: pos - self.MAX_PAD + left]
            self.noise_buffer[0, :, 0:left] = zz[0, :, pos - self.MAX_PAD: pos - self.MAX_PAD + left]
            if left < self.GROUP_SIZE:
                self.buffer_[0, :, left:] = 0
                self.noise_buffer[0, :, left:] = 0
            i_x = nd.array(self.buffer_, device=self.dev_)
            i_g = nd.array(self.noise_buffer, device=self.dev_)
            output_b1 = self.vm_["vocoder_184"](i_x, i_g)
            real_len = min(left - self.MAX_PAD, self.CHUNK)
            new_len = (pos + real_len) * self.HOP_LENGTH
            outs.append([output_b1.numpy(), new_len, real_len])
        pos = self.CHUNK
        b = outs[0][0]#.numpy()
        real_len = outs[0][1]
        output_arr[0, :real_len] = b[0, 0, :real_len]
        for i in range(1, len(outs)):
            val = outs[i]
            b = val[0]#.numpy()
            new_len = val[1]
            real_len = val[2]
            output_arr[0, pos*self.HOP_LENGTH:new_len] = b[0, 0, self.MAX_PAD * self.HOP_LENGTH:(real_len + self.MAX_PAD)*self.HOP_LENGTH]
            pos += self.CHUNK
        output_arr = torch.tensor(output_arr, dtype=torch.float32)
        return output_arr

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((c.shape[0], self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)

        if z is None:
            z = torch.randn(c.shape[0], self.noise_dim, mel.size(2)).to(mel.device)

        if 'USE_TVM_MODEL' in os.environ:
            audio = self.run_tvm(mel, z)
            audio = audio[:, :-(self.hop_length * 10)]
            audio = audio.clamp(min=-1, max=1)
        else:
            audio = self.forward(mel, z)
            audio = audio[:, :, :-(self.hop_length * 10)]
            audio = audio.clamp(min=-1, max=1)      
        
        return torch.tensor(audio, dtype=torch.float32)


if __name__ == '__main__':
    model = UnivNetGenerator()

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
