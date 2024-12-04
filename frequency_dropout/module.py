import torch
import torch.nn as nn
from typing import Optional

class FrequencyDropout(nn.Module):
    """
    Applies dropout in the frequency domain for uncertainty estimation via Monte Carlo sampling.

    This layer is designed primarily for uncertainty estimation during inference using the 
    Monte Carlo dropout approach. It operates by:
    1. Transforming the input tensor to the frequency domain using FFT
    2. Randomly dropping frequency components
    3. Optionally scaling remaining frequencies to preserve signal energy
    4. Transforming back to the spatial domain using IFFT
    """
    
    def __init__(self, 
                 p: float = 0.1, 
                 preserve_energy: bool = False,
                 preserve_dc: bool = True,
                 eps: float = 1e-8) -> None:
        """Initialize FrequencyDropout module."""
        super().__init__()
        
        if not 0 <= p <= 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        
        self.p = p
        self.preserve_energy = preserve_energy
        self.preserve_dc = preserve_dc
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain dropout to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) or (N, C, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor of the same shape as input
        """
        if self.p == 0:
            return x
            
        # Transform to frequency domain
        x_freq = torch.fft.fftn(x, dim=(-2, -1))
        
        # Generate dropout mask
        mask = (torch.rand_like(x_freq.real) > self.p).to(x.device)
        
        if self.preserve_dc:
            # Ensure DC component is not dropped
            mask_slices = [slice(None)] * (x_freq.ndim - 2) + [0, 0]
            mask[tuple(mask_slices)] = 1
        
        if self.preserve_energy:
            # Calculate initial energy using Parseval's theorem
            initial_energy = torch.sum(
                torch.abs(x_freq)**2, 
                dim=(-2, -1), 
                keepdim=True
            )
            
            # Apply mask
            x_freq_dropped = x_freq * mask
            
            # Calculate energy after dropout
            final_energy = torch.sum(
                torch.abs(x_freq_dropped)**2, 
                dim=(-2, -1), 
                keepdim=True
            )
            
            # Scale to preserve energy
            scaling_factor = torch.sqrt(
                initial_energy / (final_energy + self.eps)
            )
            x_freq = x_freq_dropped * scaling_factor
        else:
            x_freq = x_freq * mask
        
        # Transform back to spatial domain
        x_spatial = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
        
        return x_spatial
        
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (f'p={self.p}, preserve_energy={self.preserve_energy}, '
                f'preserve_dc={self.preserve_dc}, eps={self.eps}')
