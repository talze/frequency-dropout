# FrequencyDropout

A PyTorch module that implements dropout in the frequency domain for uncertainty estimation via Monte Carlo sampling. This specialized dropout layer operates by randomly dropping frequency components after FFT transformation, making it particularly useful for uncertainty estimation in signal and image processing tasks.

## Installation

```bash
pip install frequency-dropout
```

## Usage

```python
import torch
from frequency_dropout import FrequencyDropout

# Initialize the layer
freq_dropout = FrequencyDropout(
    p=0.1,                # dropout probability
    preserve_energy=True, # maintain signal energy
    preserve_dc=True     # retain DC component
)

# For uncertainty estimation
model.eval()  # Keep dropout active for MC sampling
predictions = []
for _ in range(num_samples):
    pred = model(input)  # Multiple forward passes
    predictions.append(pred)
uncertainty = torch.std(torch.stack(predictions), dim=0)
```

## Repository Structure
```
frequency-dropout/
├── README.md           # This file
├── setup.py            # Package configuration
├── frequency_dropout/  # Main package
    ├── __init__.py    # Package initialization
    └── module.py      # FrequencyDropout implementation
```

## Features

- Frequency-domain dropout for uncertainty estimation
- Optional energy preservation using Parseval's theorem
- DC component preservation option
- Support for 2D and 3D inputs
- Compatible with Monte Carlo dropout inference

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{zeevi2024monte,
  title={Monte-Carlo Frequency Dropout for Predictive Uncertainty Estimation in Deep Learning},
  author={Zeevi, T. and Venkataraman, R. and Staib, L. H. and Onofrey, J. A.},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```

## License

MIT
