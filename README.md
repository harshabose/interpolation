# Header file for k-Nearest Points Average Interpolation Class
A powerful and rapid data interpolator.

interpolate.h is a C++ header file designed to simplify interpolation tasks in scientific computing, game development, or any field requiring numerical methods for data analysis or representation using k-nearest points average interpolation technique.

## Motivation

This file is part of a larger project aiming to provide specialised tool to automate pre-conceptual and conceptual design stages of an aircraft or Unmanned Aerial Vehicle (UAV). Checkout other tools in my profile and include tools for propulsion system layout analysis, parametric analysis, optimisation and parallel-computing techniques. This file is developed to aid rapid calculation of aerodynamic data of a airfoil at any given angle of attack, reynolds number and mach number. An example case is given below.

## Key Feature

- Optimised to work with large data sets and multi-variable data.
- Memory effecient impletation of data-set storing and cache management.
- Uses Eigen C++ library to perform mathematical complex operations (a copy of Eigen3 is provided)
- Offers multiple methods for adding training data:
  - Using Eigen vectors for function and variable data.
  - Using variadic arguments for variable data with various input formats (Eigen vectors, std::vector<float>, or std::array<float, N>).
- Provides different eval_at functions to evaluate the function at a specified point using k-nearest points average interpolation:
  - Input point as an Eigen vector.
  - nput point as a compatible container (like std::vector<float> or std::array<float, N>).
  - Input point provided directly as variadic arguments (must be floating-point values).
- Implements different algorithms for finding the mean of the k-nearest neighbors based on data size for efficiency

## How to Use?
It is a header file, so simply include the file in your project.
```cpp
#include "interpolate.h"
```
