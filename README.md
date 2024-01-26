# torchoutil

<center>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

Collection of functions and modules to simplify deep learning development.

</center>


## Installation
```bash
pip install torchoutil
```

The only requirements are `python>=3.8` and `torch>=1.10`.

## Usage

### Batch of padded sequences
```python
from torchoutil import masked_mean

x = torch.as_tensor([1, 2, 3, 4])
mask = torch.as_tensor([True, True, False, False])
result = masked_mean(x, mask)
# result contains the mean of the values marked as True: 1.5
```

```python
from torchoutil import lengths_to_non_pad_mask

x = torch.as_tensor([3, 1, 2])
pad_mask = lengths_to_non_pad_mask(x, max_len=4)
# tensor([[True, True, True, False],
#         [True, False, False, False],
#         [True, True, False, False]])
```

### Multilabel conversions
```python
from torchoutil import probs_to_names

probs = torch.as_tensor([[0.9, 0.1], [0.6, 0.9]])
names = probs_to_names(probs, threshold=0.5, idx_to_name={0: "Cat", 1: "Dog"})
# [["Cat"], ["Cat", "Dog"]]
```

```python
from torchoutil import multihot_to_indices

multihot = torch.as_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
indices = multihot_to_indices(multihot)
# [[0], [1, 2], []]
```

### ...and more tensor manipulations!

```python
from torchoutil import insert_at_indices

x = torch.as_tensor([1, 2, 3, 4])
result = insert_at_indices(x, indices=[0, 2], values=5)
# result contains tensor with inserted values: tensor([5, 1, 2, 5, 3, 4])
```

## Contact
Maintainer:
- Etienne Labbé "Labbeti": labbeti.pub@gmail.com
