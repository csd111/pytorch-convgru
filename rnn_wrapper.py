import torch
import torch.nn as nn


class RNNWrapper(nn.Module):
    def __init__(
        self,
        rnn_cell: nn.Module,
        truncation_steps: int = None,
    ):
        """
        Recurrent wrapper for use with any rnn cell that takes as input a tensor
        and a hidden state and returns an updated hidden state. This wrapper
        returns the full sequence of hidden states. It assumes the first
        dimension corresponds to the timesteps, and that the other dimensions
        are directly compatible with the given rnn cell.

        Implements a very basic truncated backpropagation through time
        corresponding to the case k1=k2 (see 'An Efficient Gradient-Based
        Algorithm for On-Line Training of Recurrent Network Trajectories',
        Ronald J. Williams and Jing Pen, Neural Computation, vol. 2,
        pp. 490-501, 1990).


        Args:
            rnn_cell (nn.Module): [The torch module that takes one timestep of
                the input tensor and the hidden state and returns a new hidden
                state]
            truncation_steps (int, optional): [The maximum number of timesteps
                to include in the backpropagation graph which to detach the
                hidden state from the graph. This can help speed up runtime on
                CPU and avoid vanishing gradient problems, however it is mostly
                useful for very long sequences]. Defaults to None.
        """
        super(RNNWrapper, self).__init__()

        self.rnn_cell = rnn_cell
        self.truncation_steps = truncation_steps

    def forward(self, input, hx=None):
        output = []
        for step in range(input.size(0)):
            # Compute current time-step
            hx = self.rnn_cell(input[step], hx)
            # Detach hidden state from graph if we reach truncation threshold
            if self.truncation_steps is not None and (
                step % self.truncation_steps == 0
            ):
                hx = hx.detach()
                hx.requires_grad = True
            output.append(hx)
        # Stack the list of output hidden states into a tensor
        output = torch.stack(output, 0)
        return output
