"""Shared fixed k-grid for emulator training and prediction."""
import numpy as np

# Fixed k-grid: 200 log-spaced points covering CMB to PBH scales
FIXED_K_GRID = np.logspace(0, 18, 200)
