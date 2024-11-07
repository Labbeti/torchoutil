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

The main requirement is **[PyTorch](https://pytorch.org/)**.

To check if the package is installed and show the package version, you can use the following command:
```bash
torchoutil-info
```

## Examples

`torchoutil` functions and modules can be used like `torch` ones. The default acronym for `torchoutil` is `to`.

### Label conversions
Supports **multiclass** labels conversions between probabilities, classes indices, classes names and onehot encoding.

```python
import torchoutil as to

probs = to.as_tensor([[0.9, 0.1], [0.4, 0.6]])
names = to.probs_to_name(probs, idx_to_name={0: "Cat", 1: "Dog"})
# ["Cat", "Dog"]
```

This package also supports **multilabel** labels conversions between probabilities, classes multi-indices, classes multi-names and multihot encoding.

```python
import torchoutil as to

multihot = to.as_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
indices = to.multihot_to_indices(multihot)
# [[0], [1, 2], []]
```

### Typing

```python
from torchoutil import Tensor2D

x1 = torch.as_tensor([1, 2])
print(isinstance(x1, Tensor2D))  # False
x2 = torch.as_tensor([[1, 2], [3, 4]])
print(isinstance(x2, Tensor2D))  # True
```

```python
from torchoutil import SignedIntegerTensor

x1 = torch.as_tensor([1, 2], dtype=torch.int)
print(isinstance(x1, SignedIntegerTensor))  # True

x2 = torch.as_tensor([1, 2], dtype=torch.long)
print(isinstance(x2, SignedIntegerTensor))  # True

x3 = torch.as_tensor([1, 2], dtype=torch.float)
print(isinstance(x3, SignedIntegerTensor))  # False
```

### Padding

```python
import torchoutil as to

x1 = torch.rand(10, 3, 1)
x2 = to.pad_dim(x, target_length=5, dim=1, pad_value=-1)
# x2 has shape (10, 5, 1)
```

```python
import torchoutil as to

tensors = [torch.rand(10, 2), torch.rand(5, 3), torch.rand(0, 5)]
padded = to.pad_and_stack_rec(tensors, pad_value=0)
# padded has shape (10, 5)
```

### Masking

```python
import torchoutil as to

x = to.as_tensor([3, 1, 2])
mask = to.lengths_to_non_pad_mask(x, max_len=4)
# Each row i contains x[i] True values for non-padding mask
# tensor([[True, True, True, False],
#         [True, False, False, False],
#         [True, True, False, False]])
```

```python
import torchoutil as to

x = to.as_tensor([1, 2, 3, 4])
mask = to.as_tensor([True, True, False, False])
result = to.masked_mean(x, mask)
# result contains the mean of the values marked as True: 1.5
```

### Others tensors manipulations!

```python
import torchoutil as to

x = to.as_tensor([1, 2, 3, 4])
result = to.insert_at_indices(x, indices=[0, 2], values=5)
# result contains tensor with inserted values: tensor([5, 1, 2, 5, 3, 4])
```

```python
import torchoutil as to

perm = to.randperm(10)
inv_perm = to.get_inverse_perm(perm)

x1 = to.rand(10)
x2 = x1[perm]
x3 = x2[inv_perm]
# inv_perm are indices that allow us to get x3 from x2, i.e. x1 == x3 here
```

### Pre-compute datasets to pickle or HDF files

Here is an example of pre-computing spectrograms of torchaudio `SPEECHCOMMANDS` dataset, using `pack_dataset` function:

```python
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Spectrogram
from torchoutil import nn
from torchoutil.utils.pack import pack_dataset

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

pack_dataset(dataset, packed_root, MyTransform())
```

Then you can load the pre-computed dataset using `PackedDataset`:
```python
from torchoutil.utils.pack import PackedDataset

packed_root = "path/to/packed_dataset"
packed_dataset = PackedDataset(packed_root)
packed_dataset[0]  # == first transformed item, i.e. transform(dataset[0])
```

<!--
## Main modules

- [IndexToName](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.IndexToName): Convert multiclass indices to names.
- [IndexToOnehot](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.IndexToOnehot): Convert multiclass indices to onehot encoding.
- [NameToIndex](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.NameToIndex): Convert names to multiclass indices.
- [NameToOnehot](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.NameToOnehot): Convert names to onehot encoding.
- [OnehotToIndex](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.OnehotToIndex): Convert onehot encoding to multiclass indices.
- [OnehotToName](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.OnehotToName): Convert onehot encoding to names.
- [ProbsToIndex](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.ProbsToIndex): Convert probabilities to multiclass indices using a threshold.
- [ProbsToName](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.ProbsToName): Convert probabilities to names using a threshold.
- [ProbsToOnehot](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multiclass.html#torchoutil.nn.modules.multiclass.ProbsToOnehot): Convert probabilities to onehot encoding using a threshold.
- [IndicesToMultihot](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.IndicesToMultihot): Convert multilabel indices to names.
- [IndicesToNames](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.IndicesToNames): Convert multilabel indices to multihot encoding.
- [MultihotToIndices](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.MultihotToIndices): Convert multihot encoding to multilabel indices.
- [MultihotToNames](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.MultihotToNames): Convert multihot encoding to names.
- [NamesToIndices](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.NamesToIndices): Convert names to multilabel indices.
- [NamesToMultihot](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.NamesToMultihot): Convert names to multihot encoding.
- [ProbsToIndices](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.ProbsToIndices): Convert probabilities to multilabel indices using a threshold.
- [ProbsToMultihot](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.ProbsToMultihot): Convert probabilities to names using a threshold.
- [ProbsToNames](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.multilabel.html#torchoutil.nn.modules.multilabel.ProbsToNames): Convert probabilities to multihot encoding using a threshold.

- [LogSoftmaxMultidim](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.activation.html#torchoutil.nn.modules.activation.LogSoftmaxMultidim): Apply LogSoftmax along multiple dimensions.
- [SoftmaxMultidim](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.activation.html#torchoutil.nn.modules.activation.SoftmaxMultidim): Apply Softmax along multiple dimensions.
- [CropDim](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.crop.html#torchoutil.nn.modules.crop.CropDim): Crop a tensor along a single dimension.
- [CropDims](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.crop.html#torchoutil.nn.modules.crop.CropDims): Crop a tensor along multiple dimensions.
- [PositionalEncoding](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.layer.html#torchoutil.nn.modules.layer.PositionalEncoding): Positional encoding layer for vanilla transformers models.
- [MaskedMean](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.mask.html#torchoutil.nn.modules.mask.MaskedMean): Average non-masked element of a tensor.
- [MaskedSum](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.mask.html#torchoutil.nn.modules.mask.MaskedSum): Sum non-masked element of a tensor.
- [NumpyToTensor](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.numpy.html#torchoutil.nn.modules.numpy.NumpyToTensor): Convert numpy array or generic to tensor.
- [NumpyToTensor](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.numpy.html#torchoutil.nn.modules.numpy.TensorToNumpy): Convert tensor to numpy array.
- [NumpyToTensor](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.numpy.html#torchoutil.nn.modules.numpy.ToNumpy): Convert sequence to numpy array.
- [PadAndStackRec](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.pad.html#torchoutil.nn.modules.pad.PadAndStackRec): Pad and stack sequence to tensor.
- [PadDim](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.pad.html#torchoutil.nn.modules.pad.PadDim): Pad a tensor along a single dimension.
- [PadDims](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.pad.html#torchoutil.nn.modules.pad.PadDims): Pad a tensor along multiples dimensions.
- [RepeatInterleaveNd](https://torchoutil.readthedocs.io/en/latest/torchoutil.nn.modules.transform.html#torchoutil.nn.modules.transform.RepeatInterleaveNd): Repeat interleave a tensor with an arbitrary number of dimensions.

from .tensor import (
    FFT,
    IFFT,
    Abs,
    Angle,
    AsTensor,
    Exp,
    Exp2,
    Imag,
    Log,
    Log2,
    Log10,
    Max,
    Mean,
    Min,
    Normalize,
    Permute,
    Pow,
    Real,
    Repeat,
    RepeatInterleave,
    Reshape,
    Squeeze,
    TensorTo,
    ToItem,
    ToList,
    Transpose,
    Unsqueeze,
)
from .transform import (
    Flatten,
    Identity,
    PadAndCropDim,
    RepeatInterleaveNd,
    ResampleNearestFreqs,
    ResampleNearestRates,
    ResampleNearestSteps,
    Shuffled,
    TransformDrop,
)
-->


## Extras requirements
`torchoutil` also provides additional modules when some specific package are already installed in your environment.
All extras can be installed with `pip install torchoutil[extras]`

- If `tensorboard` is installed, the function `load_event_file` can be used. It is useful to load manually all data contained in an tensorboard event file.
- If `numpy` is installed, the classes `NumpyToTensor` and  `ToNumpy` can be used and their related function. It is meant to be used to compose dynamic transforms into `Sequential` module.
- If `h5py` is installed, the function `pack_to_hdf` and class `HDFDataset` can be used. Can be used to pack/read dataset to HDF files, and supports variable-length sequences of data.
- If `pyyaml` is installed, the functions `to_yaml` and `load_yaml` can be used.


## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
