# extentorch

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
