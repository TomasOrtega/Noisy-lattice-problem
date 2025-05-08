# Noisy Lattice Denoising

This repository implements an algorithm to recover the underlying lattice structure from noisy 2D measurements. Given a set of points perturbed by Gaussian noise, the code estimates the lattice origin and basis vectors that minimize the overall mean squared error (MSE) and outputs the integer lattice coordinates for each point.

See the image below for an example. The circles are the given noisy measurements, which we are given initially.
We then compute the minimum square error lattice, plotted with dotted lines in the figure. The crosses are where the adjusted measurements fall in the lattice.

![Lattice Example](example.png)

An iterative procedure is used to obtain the solution, seen in the animation below.

![Lattice Animation](animated.gif)

## Features

* Automatic initialization of lattice origin and basis
* Iterative coordinate assignment and basis refinement
* Visualization of noisy measurements, recovered lattice points, and lattice grid lines
* Support for both batch processing and incremental (online) fitting
* Available in both Python and MATLAB implementations
* Comprehensive test suite for Python implementation

## Requirements

### Python Version
* Python 3.6 or later
* NumPy >= 1.19.0
* SciPy >= 1.7.0
* Matplotlib >= 3.3.0

Install dependencies:
```bash
pip install -r requirements.txt
```

### MATLAB Version
* MATLAB R2018b or later
* Optimization Toolbox (for `fminsearch`)
* Statistics and Machine Learning Toolbox (for `pdist` and `squareform`)

## Repository Structure

```
/denoise_lattice.py          Main Python function for batch lattice denoising
/test_denoise_lattice.py     Python test suite
/requirements.txt            Python package dependencies
/denoiseLattice.m            Main MATLAB function for batch lattice denoising
/legacy_matlab/              Legacy MATLAB scripts for baseline implementations
    noisyLattice.m           Simple batch version with heuristic initialization
    noisyLatticeIncremental.m Incremental version processing points one by one
/README.md                   Project documentation
/example.png                 Sample noisy measurements visualization
/animated.gif                Iterative lattice fitting animation
```

## Usage

### Python Version

```python
import numpy as np
from denoise_lattice import denoise_lattice

# Generate or load your noisy measurements
# noisy_mes should be a 2×N numpy array of noisy 2D points
[coords, origin, v1, v2] = denoise_lattice(noisy_mes)

# coords: 2×N integer coordinates in lattice space
# origin: 2×1 vector representing the lattice origin
# v1, v2: 2×1 basis vectors defining the lattice
```

The Python implementation includes an example usage in the main block that generates random noisy lattice points and demonstrates the algorithm.

### Running Tests

The Python implementation includes a comprehensive test suite. To run the tests:

```bash
python -m unittest test_denoise_lattice.py
```

The tests cover:
* Basic functionality
  - Output shapes and types
  - Integer coordinate validation
  - Basis vector properties
* Edge cases
  - Minimum number of points (4)
  - Different noise levels
  - Input validation
* Quality checks
  - Reconstruction error
  - Consistency across multiple runs

### MATLAB Version

```matlab
% Assume noisyMes is a 2×N matrix of noisy 2D points:
[coords, origin, v1, v2] = denoiseLattice(noisyMes);
% coords: 2×N integer coordinates in lattice space
% origin: 2×1 vector representing the lattice origin
% v1, v2: 2×1 basis vectors defining the lattice
```

### Incremental Mode (MATLAB only)

```matlab
% Legacy script with point-by-point fitting:
legacy_matlab/noisyLatticeIncremental.m
% Adjust parameters (e.g., data source, number of iterations) and run.
noisyLatticeIncremental
```

## Algorithm Overview

1. **Initialization**: Select an initial origin as the point with the smallest sum of distances to all other points. Initialize basis vectors using the nearest neighbor and its perpendicular.
2. **Coordinate Assignment**: For each measurement, compute floating-point lattice coordinates by solving:

   $$
     \lambda = [v_1\; v_2]^{-1}(p - origin),
   $$

   then round within a small search window to find the nearest integer lattice coordinates.
3. **Basis Refinement**: Optimize the origin and basis vectors by minimizing total squared reconstruction error:

   $$
     \min_{origin, v_1, v_2} \sum_k \|origin + \lambda_k^1 v_1 + \lambda_k^2 v_2 - p_k\|^2
   $$

   using optimization tools (`fminsearch` in MATLAB, `scipy.optimize.fmin` in Python).
4. **Iteration**: Alternate between coordinate assignment and basis refinement until convergence or maximum iterations.

![Example of Noisy Measurements and Recovered Lattice](example.png)

![Lattice Fitting Animation](animated.gif)

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and open a pull request for review. For Python contributions, please ensure all tests pass and add new tests for any new functionality.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research or projects, please cite:

> T. Ortega, "Noisy Lattice Denoising," GitHub repository, 2025. Available at [https://github.com/TomasOrtega/Noisy-lattice-problem](https://github.com/TomasOrtega/Noisy-lattice-problem)
