#!/usr/bin/env python3
"""Find maximum temperature in T_no_velocity file at timestep 0.2s (folder 50001)."""

import numpy as np

# Load the temperature data
T_no_velocity = np.load('results/3_fine/050001/T_no_velocity.npy')

# Find the maximum temperature
max_temp = np.max(T_no_velocity)

# Maximum temperature at timestep 0.2s: 2050.51

print(f"Maximum temperature at timestep 0.2s: {max_temp}")
