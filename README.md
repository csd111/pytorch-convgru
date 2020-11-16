# pytorch-convgru
PyTorch implementations of one- and two-dimensional Convolutional Gated Recurrent Units.

It features a grouping of the input-to-hidden and hidden-to-hidden 
kernels as in the cuDNN implementation of GRUs, plus some torchscript optmizations
and a basic RNN wrapper with simplified truncated backpropagation through time.

###Usage

See unit tests for basic usage
```
python -m unittest discover tests/
```


The gist of it is :
```
from convgru import ConvGRU1DCell
from rnn_wrapper import RNNWrapper
```
```
cell = ConvGRU1DCell(input_channels, hidden_channels, kernel_size, stride=stride, padding=padding)
conv_gru_1d = RNNWrapper(cgru1d_cell, truncation_steps=128)
```
### License

This program is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness for a 
particular purpose.  See the GNU General Public License for more details.
