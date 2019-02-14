import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional


class ConvGRU1DCell(nn.Module):
    
    def __init__(self, input_channels: int, hidden_channels: int, 
                 kernel_size: int, stride: int=1, padding: int=0,
                 recurrent_kernel_size: int=3):
        super(ConvGRU1DCell, self).__init__()
        
        self.conv_ih = nn.Conv1d(input_channels, hidden_channels * 3, 
                                 kernel_size, stride=stride, padding=padding)
        self.conv_hh = nn.Conv1d(hidden_channels, hidden_channels * 3, 
                                 recurrent_kernel_size, stride=1, 
                                 padding=recurrent_kernel_size // 2)
        self.h_channels = hidden_channels
    
        self.reset_parameters()


    def reset_parameters(self):
        init.orthogonal_(self.conv_hh.weight)
        init.xavier_uniform_(self.conv_ih.weight)
        init.zeros_(self.conv_hh.bias)
        init.zeros_(self.conv_ih.bias)
    
    def forward(self, input, hx=None):
        output_size = \
            ((input.size(-1) - self.conv_ih.kernel_size + 
              2 * self.conv_ih.padding) / self.conv_ih.stride) + 1
        if hx is None:
            hx = torch.zeros(input.size(0), self.h_channels, output_size)
        # Run the input->hidden and hidden->hidden convolution kernels
        ih_conv_output = self.conv_ih(input)
        hh_conv_output = self.conv_ih(hx)
        z = functional.sigmoid(ih_conv_output[:, :self.h_channels, :] + 
                               hh_conv_output[:, :self.h_channels, :])
        r = functional.sigmoid(ih_conv_output[:, self.h_channels:2*self.h_channels, :] + 
                               hh_conv_output[:, self.h_channels:2*self.h_channels, :])
        n = functional.tanh(ih_conv_output[:, 2*self.h_channels:, :] + 
                            r * hh_conv_output[:, 2*self.h_channels:, :])
        return (1 - z) * n + z * hx
