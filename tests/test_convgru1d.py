import unittest
import torch
from convgru import ConvGRU1DCell
from rnn_wrapper import RNNWrapper


class ConvGRU1DTest(unittest.TestCase):
    def test_convgru1d_cell(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = 3
        padding = 1
        stride = 1
        data = 1 + torch.randn(10, channels, 128)
        hidden_state = 2 + torch.randn(10, channels, 128)
        dirac_weight = torch.zeros(channels, channels, kernel_size)
        torch.nn.init.dirac_(dirac_weight)
        dirac_weight = dirac_weight.repeat(3, 1, 1)
        # ----------------------------------------------------------------------
        # Initialize cell to get output=tanh(input) and run it
        # ----------------------------------------------------------------------
        cell = ConvGRU1DCell(
            channels, channels, kernel_size, stride=stride, padding=padding
        )
        # Init weights to force z=0 and n=tanh(input)
        cell.weight_ih.data = dirac_weight
        torch.nn.init.constant_(cell.weight_hh, -(10 ** 5))
        # Run forward pass
        output_data = cell(data, hidden_state)
        self.assertLessEqual(
            torch.max((torch.tanh(data) - output_data).abs()), 10 ** (-12)
        )
        # ----------------------------------------------------------------------
        # Initialize cell to get output=hidden and run it
        # ----------------------------------------------------------------------
        # Init weights to force z=1
        torch.nn.init.constant_(cell.weight_ih, 10 ** 5)
        torch.nn.init.constant_(cell.weight_hh, 1)
        # Run forward pass
        output_data = cell(data, hx=hidden_state)
        self.assertLessEqual(
            torch.max((hidden_state - output_data).abs()), 10 ** (-12)
        )
        # ----------------------------------------------------------------------
        # Call reset_parameters() and check output != input
        # ----------------------------------------------------------------------
        cell.reset_parameters()
        output_data = cell(data, hidden_state)
        self.assertIs(bool(torch.all(torch.eq(data, output_data))), False)

    def test_convgru1d_training_rnn_wrapper(self):
        # ----------------------------------------------------------------------
        # Data preparation
        # ----------------------------------------------------------------------
        channels = 8
        kernel_size = 3
        padding = 1
        stride = 1
        time_steps = 64
        batch_size = 10
        nb_features = 128
        data = 1 + torch.randn(time_steps, batch_size, channels, nb_features)
        target = torch.tanh(data)
        # ----------------------------------------------------------------------
        # Instantiate model for training
        # ----------------------------------------------------------------------
        cgru1d_cell = ConvGRU1DCell(
            channels, channels, kernel_size, stride=stride, padding=padding
        )
        cgru = RNNWrapper(cgru1d_cell, truncation_steps=time_steps // 4)
        optimizer = torch.optim.RMSprop(cgru.parameters(), 0.005)
        loss = torch.nn.MSELoss()
        weights_ih_bf_train = cgru1d_cell.weight_ih.data.clone()
        weights_hh_bf_train = cgru1d_cell.weight_hh.data.clone()
        # ----------------------------------------------------------------------
        # Train the model and check
        # ----------------------------------------------------------------------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cgru.to(device)
        data = data.to(device)
        target = target.to(device)
        # Run a basic training procedure
        for index in range(500):
            # Reset the gradients
            optimizer.zero_grad()
            # Run the RNN wrapper
            output = cgru(data)
            # Compute the loss and backpropagate
            error = loss(output, target)
            error.backward()
            optimizer.step()
            if index % 100 == 0:
                print(
                    "Step {0} / 500 - MSError is {1}".format(index, error.item())
                )
        # Make sure the weights have changed
        weights_ih_af_train = cgru1d_cell.weight_ih.data.cpu().clone()
        weights_hh_af_train = cgru1d_cell.weight_hh.data.cpu().clone()
        self.assertIs(
            bool(torch.all(torch.eq(weights_ih_bf_train, weights_ih_af_train))),
            False,
        )
        self.assertIs(
            bool(torch.all(torch.eq(weights_hh_bf_train, weights_hh_af_train))),
            False,
        )
        # Check the final error is low enough
        self.assertLessEqual(error.item(), 5 * 10 ** (-3))
