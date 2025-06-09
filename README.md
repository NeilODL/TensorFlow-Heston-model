# Heston ANN: Neural Network Implementation of the Heston Model

This project implements a neural network-based approach to solve the Heston stochastic volatility model, a popular model in quantitative finance for pricing options.

## Overview

The Heston model is a stochastic volatility model that assumes the volatility of the underlying asset follows a mean-reverting square root process. This implementation uses artificial neural networks to approximate the solution to the Heston partial differential equation (PDE).

## Features

- Neural network implementation of the Heston model
- Training and prediction of option prices
- Support for various option types (European, American)
- Visualization of results and model performance
- Integration with financial data sources

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heston-ann.git
cd heston-ann
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```python
from heston_ann import HestonANN

model = HestonANN()
model.train(epochs=1000, batch_size=32)
```

2. Make predictions:
```python
predictions = model.predict(test_data)
```

3. Visualize results:
```python
model.plot_results()
```

## Project Structure

```
heston-ann/
├── data/               # Data storage
├── models/            # Neural network models
├── utils/             # Utility functions
├── visualization/     # Plotting and visualization
├── tests/             # Unit tests
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Model Architecture

The neural network architecture consists of:
- Input layer: Market parameters (spot price, strike price, time to maturity, etc.)
- Hidden layers: Multiple fully connected layers with ReLU activation
- Output layer: Option price prediction

## Training

The model is trained using:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: Adaptive
- Batch size: Configurable

## Performance

The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared score
- Comparison with analytical solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Heston model paper: "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options" by Steven L. Heston
- PyTorch community for the deep learning framework
- Contributors and maintainers of the project

## Contact

For questions and support, please open an issue in the GitHub repository or contact the maintainers.

## Citation

If you use this code in your research, please cite:
```
@software{heston_ann,
  author = {Your Name},
  title = {Heston ANN: Neural Network Implementation of the Heston Model},
  year = {2024},
  url = {https://github.com/yourusername/heston-ann}
}
```
