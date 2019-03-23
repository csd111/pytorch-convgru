import unittest
import torch
from convgru import ConvGRU2DCell


class ConvGRU2DTest(unittest.TestCase):

    def test_convgru2d_cell(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = (3, 3)
        padding = (1, 1)
        stride = (1, 1)
        data =  1 + torch.randn(5, channels, 64, 64)
        hidden_state = 2 + torch.randn(5, channels, 64, 64)
        dirac_weight = torch.zeros(channels, channels, *kernel_size)
        torch.nn.init.dirac_(dirac_weight)
        dirac_weight = dirac_weight.repeat(3, 1, 1, 1)
        # ----------------------------------------------------------------------
        # Initialize cell to get output=tanh(input) and run it 
        # ----------------------------------------------------------------------
        cell = ConvGRU2DCell(channels, channels, kernel_size, 
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

    def test_convgru2d_training(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = (3, 3)
        padding = (1, 1)
        stride = (1, 1)
        batch_size = 10
        time_steps = 32
        features = (32, 32)
        data =  1 + torch.randn(time_steps, batch_size, channels, *features)
        # Perpare the target
        target = torch.tanh(data)
        for step in range(1, data.size(0)):
            target[step] = 0.5 * target[step] + 0.5 * target[step - 1]
        # ----------------------------------------------------------------------
        # Instantiate model for training
        # ----------------------------------------------------------------------
        cgru2d = ConvGRU2DCell(channels, channels, kernel_size, 
                               stride=stride, padding=padding)
        optimizer = torch.optim.RMSprop(cgru2d.parameters(), 0.001)
        loss = torch.nn.MSELoss()
        weights_ih_bf_train = cgru2d.conv_ih.weight.data.clone()
        weights_hh_bf_train = cgru2d.conv_hh.weight.data.clone()
        # ----------------------------------------------------------------------
        # Train the model and check
        # ----------------------------------------------------------------------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cgru2d.to(device)
        # Run a basic training procedure
        for index in range(1000):
            # Reset the gradients
            optimizer.zero_grad()
            data = data.to(device)
            # Loop over the time steps
            output = []
            hx = None
            for step in range(data.size(0)):
                hx = cgru2d(data[step], hx)
                output.append(hx)
            # Reshape the output appropriately
            output = torch.cat(output, 0).view(data.size(0), *output[0].size())
            # Compute the loss and backpropagate
            error = loss(output, target)
            error.backward()
            optimizer.step()
            if index % 100 == 0:
                print("Step {0} / 1000 - MSError is {1}".format(index, 
                                                                error.item()))
        # Make sure the weights have changed
        weights_ih_af_train = cgru2d.conv_ih.weight.data.cpu().clone()
        weights_hh_af_train = cgru2d.conv_hh.weight.data.cpu().clone()
        self.assertIs(
            bool(torch.all(torch.eq(weights_ih_bf_train, weights_ih_af_train))), 
            False)
        self.assertIs(
            bool(torch.all(torch.eq(weights_hh_bf_train, weights_hh_af_train))), 
            False)
        # Check the final error is low enough
        self.assertLessEqual(error.item(), 5*10**(-2))
