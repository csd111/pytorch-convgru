import unittest
import torch
from convgru import ConvGRU1DCell, ConvGRU1D


class ConvGRU1DCellTest(unittest.TestCase):

    def test_convgru1dcell(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = 3
        padding = 1
        stride = 1
        data =  1 + torch.randn(10, channels, 128)
        hidden_state = 2 + torch.randn(10, channels, 128)
        dirac_weight = torch.zeros(channels, channels, kernel_size)
        torch.nn.init.dirac_(dirac_weight)
        dirac_weight = dirac_weight.repeat(3, 1, 1)
        # ----------------------------------------------------------------------
        # Initialize layer to get output=tanh(input) and run it 
        # ----------------------------------------------------------------------
        cell = ConvGRU1DCell(channels, channels, kernel_size, 
                             stride=stride, padding=padding)
        cell.conv_ih.weight.data = dirac_weight
        torch.nn.init.constant_(cell.conv_hh.weight, -10**5)
        output_data = cell(data, hidden_state)
        self.assertLessEqual(
            torch.max((torch.tanh(data) - output_data).abs()), 
             10**(-12))
        # ----------------------------------------------------------------------
        # Initialize layer to get output=hidden and run it 
        # ----------------------------------------------------------------------
        torch.nn.init.constant_(cell.conv_ih.weight, 10**5)
        torch.nn.init.constant_(cell.conv_hh.weight, 1)
        output_data = cell(data, hx=hidden_state)
        self.assertLessEqual(
            torch.max((hidden_state - output_data).abs()), 10**(-12))
