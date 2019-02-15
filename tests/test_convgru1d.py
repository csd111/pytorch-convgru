import unittest
import torch
from convgru import ConvGRU1DCell, ConvGRU1D


class ConvGRU1DTest(unittest.TestCase):

    def test_convgru1d_cell(self):
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
        # Initialize cell to get output=tanh(input) and run it 
        # ----------------------------------------------------------------------
        cell = ConvGRU1DCell(channels, channels, kernel_size, 
                             stride=stride, padding=padding)
        # Init weights to force z=0 and n=tanh(input)
        cell.conv_ih.weight.data = dirac_weight
        torch.nn.init.constant_(cell.conv_hh.weight, -10**5)
        # Run forward pass
        output_data = cell(data, hidden_state)
        self.assertLessEqual(
            torch.max((torch.tanh(data) - output_data).abs()), 10**(-12))
        # ----------------------------------------------------------------------
        # Initialize cell to get output=hidden and run it 
        # ----------------------------------------------------------------------
        # Init weights to force z=1
        torch.nn.init.constant_(cell.conv_ih.weight, 10**5)
        torch.nn.init.constant_(cell.conv_hh.weight, 1)
        # Run forward pass
        output_data = cell(data, hx=hidden_state)
        self.assertLessEqual(
            torch.max((hidden_state - output_data).abs()), 10**(-12))
        # ----------------------------------------------------------------------
        # Call reset_parameters() and check output != input
        # ----------------------------------------------------------------------
        cell.reset_parameters()
        output_data = cell(data, hidden_state)
        self.assertIs(bool(torch.all(torch.eq(data, output_data))), False)

    def test_convgru1d(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = 3
        padding = 1
        stride = 1
        data =  2 + torch.randn(10, channels, 128, 128)
        hidden_state = 2 + torch.randn(10, channels, 128)
        dirac_weight = torch.zeros(channels, channels, kernel_size)
        torch.nn.init.dirac_(dirac_weight)
        dirac_weight = dirac_weight.repeat(3, 1, 1)
        # ----------------------------------------------------------------------
        # Check the batch_first option works fine
        # ----------------------------------------------------------------------
        cgru1 = ConvGRU1D(1, channels, channels, kernel_size, 
                          stride=stride, padding=padding, batch_first=True)
        cgru2 = ConvGRU1D(1, channels, channels, kernel_size, 
                          stride=stride, padding=padding, batch_first=False)
        # transfer weights
        cgru2.cell.conv_hh.weight.data = cgru1.cell.conv_hh.weight.data[:]
        cgru2.cell.conv_ih.weight.data = cgru1.cell.conv_ih.weight.data[:]
        # Compute outputs
        output1 = cgru1(data)[0]
        output2 = cgru2(data.permute(2, 0, 1, 3))[0].permute(1, 2, 0, 3)
        self.assertIs(bool(torch.all(torch.eq(output1, output2))), True)
        # ----------------------------------------------------------------------
        # Initialize the layer's cell to get output=tanh(input) and try forward
        # ----------------------------------------------------------------------
        cgru1.cell.conv_ih.weight.data = dirac_weight[:]
        torch.nn.init.constant_(cgru1.cell.conv_hh.weight, -10**5)
        output = cgru1(data, hx=hidden_state)[0]
        self.assertLessEqual(
            torch.max((output - torch.tanh(data)).abs()), 10**(-12))

    def test_convgru1d_training(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = 3
        padding = 1
        stride = 1
        data =  1 + torch.randn(10, channels, 128, 128)
        target = torch.tanh(data)
        # ----------------------------------------------------------------------
        # Instantiate model for training
        # ----------------------------------------------------------------------
        cgru1d = ConvGRU1D(1, channels, channels, kernel_size, 
                           stride=stride, padding=padding, batch_first=True)
        optimizer = torch.optim.Adam(cgru1d.parameters(), 0.005)
        loss = torch.nn.MSELoss()
        weights_ih_bf_train = cgru1d.cell.conv_ih.weight.data.clone()
        weights_hh_bf_train = cgru1d.cell.conv_hh.weight.data.clone()
        # ----------------------------------------------------------------------
        # Train the model and check
        # ----------------------------------------------------------------------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cgru1d.to(device)
        for index in range(2000):
            optimizer.zero_grad()
            output, hidden = cgru1d(data.to(device))
            error = loss(output, target)
            error.backward()
            optimizer.step()
            if index % 100 == 0:
                print("Step {0} / 2000 - MSError is {1}".format(index, 
                                                                error.item()))
        weights_ih_af_train = cgru1d.cell.conv_ih.weight.data.cpu().clone()
        weights_hh_af_train = cgru1d.cell.conv_hh.weight.data.cpu().clone()
        self.assertIs(
            bool(torch.all(torch.eq(weights_ih_bf_train, weights_ih_af_train))), 
            False)
        self.assertIs(
            bool(torch.all(torch.eq(weights_hh_bf_train, weights_hh_af_train))), 
            False)
        self.assertLessEqual(error.item(), 10**(-4))
