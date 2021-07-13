# MLE for Exponential Hawkes Process

Source code for [Maximum likelihood estimation for self-regulating Gaussian Hawkes process](https://arxiv.org/abs/2103.05299).

It includes a class to simulate an univariate Hawkes process and functions to compute the log-likelihood presented in the aforementioned paper along with the approximated version from Lemonnier's work [[1]](#1).

## Example

We include two examples of application, one for simple plotting with the ```exp_thinning_hawkes``` class, shown below, and another one implementing the algorithm of estimation with the ```estimator_class```.

```python:./examples/example_simulation.py

```

This example must yield the following plot:

<img src="./examples/plot_simulation.png" width="500">

## Dependencies

This code was implemented using Python 3.8.5 and needs Numpy, Matplotlib and Scipy.

## Installation

Copy both files in the current directory.

## Author

Miguel Alejandro Martinez Herrera

## References

<a id="1">[1]</a>
R. Lemonnier, N. Vayatis, Nonparametric markovian learning of triggering kernels for mutually exciting and mutually inhibiting multivariate hawkes processes, in: Machine Learning and Knowledge Discovery in Databases, Springer Berlin Heidelberg, 2014, p. 161–176
