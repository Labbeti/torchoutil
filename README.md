# torchoutil

<center>

<a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
</a>
<a href="https://black.readthedocs.io/en/stable/">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray">
</a>
<a href="https://github.com/Labbeti/torchoutil/actions">
    <img alt="Build" src="https://img.shields.io/github/actions/workflow/status/Labbeti/torchoutil/test.yaml?branch=main&style=for-the-badge&logo=github">
</a>
<a href='https://torchoutil.readthedocs.io/en/stable/?badge=stable'>
    <img src='https://readthedocs.org/projects/torchoutil/badge/?version=stable&style=for-the-badge' alt='Documentation Status' />
</a>

Collection of functions and modules to help development in PyTorch.

</center>


## Installation
```bash
pip install torchoutil
```

The only requirement is **PyTorch**.

To check if the package is installed and show the package version, you can use the following command:
```bash
torchoutil-info
```


## Examples

### Multilabel conversions
```python
import torch
from torchoutil import probs_to_name

probs = torch.as_tensor([[0.9, 0.1], [0.4, 0.6]])
names = probs_to_name(probs, idx_to_name={0: "Cat", 1: "Dog"})
# ["Cat", "Dog"]
```

```python
import torch
from torchoutil import multihot_to_indices

multihot = torch.as_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
indices = multihot_to_indices(multihot)
# [[0], [1, 2], []]
```

### Masked operations

```python
import torch
from torchoutil import lengths_to_non_pad_mask

x = torch.as_tensor([3, 1, 2])
mask = lengths_to_non_pad_mask(x, max_len=4)
# Each row i contains x[i] True values for non-padding mask
# tensor([[True, True, True, False],
#         [True, False, False, False],
#         [True, True, False, False]])
```

```python
import torch
from torchoutil import masked_mean

x = torch.as_tensor([1, 2, 3, 4])
mask = torch.as_tensor([True, True, False, False])
result = masked_mean(x, mask)
# result contains the mean of the values marked as True: 1.5
```

### Pre-compute datasets to pickle or HDF files

Here is an example of pre-computing spectrograms of torchaudio `SPEECHCOMMANDS` dataset, using `pack_to_custom` function:

```python
from torch import nn
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Spectrogram
from torchoutil.utils.pack import pack_to_custom

speech_commands_root = "path/to/speech_commands"
packed_root = "path/to/packed_dataset"

dataset = SPEECHCOMMANDS(speech_commands_root, download=True, subset="validation")
# dataset[0] is a tuple, contains waveform and other metadata

class MyTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram()

    def forward(self, item):
        waveform = item[0]
        spectrogram = self.spectrogram_extractor(waveform)
        return (spectrogram,) + item[1:]

pack_to_custom(dataset, packed_root, MyTransform())
```

Then you can load the pre-computed dataset using `PackedDataset`:
```python
from torchoutil.utils.pack import PackedDataset

packed_root = "path/to/packed_dataset"
packed_dataset = PackedDataset(packed_root)
packed_dataset[0]  # == first transformed item, i.e. transform(dataset[0])
```

### Other tensors manipulations!

```python
import torch
from torchoutil import insert_at_indices

x = torch.as_tensor([1, 2, 3, 4])
result = insert_at_indices(x, indices=[0, 2], values=5)
# result contains tensor with inserted values: tensor([5, 1, 2, 5, 3, 4])
```

```python
import torch
from torchoutil import get_inverse_perm

perm = torch.randperm(10)
inv_perm = get_inverse_perm(perm)

x1 = torch.rand(10)
x2 = x1[perm]
x3 = x2[inv_perm]
# inv_perm are indices that allow us to get x3 from x2, i.e. x1 == x3 here
```

## Extras requirements
`torchoutil` also provides additional modules when some specific package are already installed in your environment.
All extras can be installed with `pip install torchoutil[extras]`

- If `tensorboard` is installed, the function `load_event_file` can be used. It is useful to load manually all data contained in an tensorboard event file.
- If `numpy` is installed, the classes `NumpyToTensor` and  `ToNumpy` can be used and their related function. It is meant to be used to compose dynamic transforms into `Sequential` module.
- If `h5py` is installed, the function `pack_to_hdf` and class `HDFDataset` can be used. Can be used to pack/read dataset to HDF files, and supports variable-length sequences of data.


## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
