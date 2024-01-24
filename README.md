# extentorch

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

Utilities for manipuling PyTorch tensors.


## Installation
```bash
pip install extentorch
```

## Usage
```python
from extentorch import masked_mean

x = torch.as_tensor([1, 2, 3, 4])
mask = torch.as_tensor([True, True, False, False])
result = masked_mean(x, mask)
# result contains the mean of the values marked as True: 1.5
```

## Contact
Maintainer:
- Etienne Labbé "Labbeti": labbeti.pub@gmail.com
