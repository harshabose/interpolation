# Header file for k-Nearest Points Average Interpolation Class
A powerful and rapid data interpolator.

interpolate.h is a C++ header file designed to simplify interpolation tasks in scientific computing, game development, or any field requiring numerical methods for data analysis or representation using k-nearest points average interpolation technique.

## Motivation

This file is part of a larger project aiming to provide specialised tool to automate pre-conceptual and conceptual design stages of an aircraft or Unmanned Aerial Vehicle (UAV). Checkout other tools in my profile and include tools for propulsion system layout analysis, parametric analysis, optimisation and parallel-computing techniques. This file is developed to aid rapid calculation of aerodynamic data of a airfoil at any given angle of attack, reynolds number and mach number. An example case is given below.

## Key Feature

- Optimised to work with large data sets and multi-variable data.
- Memory effecient impletation of data-set storing and cache management.
- Uses Eigen C++ library to perform mathematical complex operations (a copy of Eigen 3.4 is provided)
- Offers multiple methods for adding training data:
  - Using Eigen vectors for function and variable data.
  - Using variadic arguments for variable data with various input formats (Eigen vectors, std::vector<float>, or std::array<float, N>).
- Provides different eval_at functions to evaluate the function at a specified point using k-nearest points average interpolation:
  - Input point as an Eigen vector.
  - nput point as a compatible container (like std::vector<float> or std::array<float, N>).
  - Input point provided directly as variadic arguments (must be floating-point values).
- Implements different algorithms for finding the mean of the k-nearest neighbors based on data size for efficiency

## How to Use?
- Clone the repository into your project directory
  ```bash
  git clone --recursive https://github.com/harshabose/interpolation.git
  ```
- It is a header file, so simply include the file in your project.
  
  ```cpp
  #include "interpolation/interpolate.h"
  ```
- Define the template parameters based on your specific needs.
- Create an instance of the interpolate class:
  
  ```cpp
  template <std::size_t dimension, std::size_t max_training_data_size, std::size_t mean_size> interpolate<dimension, max_training_data_size, mean_size> interpolator;
  ```
- Add training data to the interpolator using the appropriate add_training_data method.
- Use the eval_at function to evaluate the function at a specified point.

## Example
The following example creates a interpolator for CL (lift coeffecient) of an airfoil from angle of attack (-90deg to 90deg) and reynolds number (100,000 to 2,000,000).

<p align="center">
  <img src="[http://some_place.com/image.png](https://github.com/harshabose/interpolation/assets/127072856/39b58c95-0344-45c3-8060-a8a238461033)" />
</p>

Following is a 3D plot representing the training data collected from 5000 CFD simulations of NACA 65(2)-415 airfoil (included in the repo). They are in the form of JSON data. To parse this JSON data, we will use nlohmann/json library. This library is also included as a git submodule.

Following is the interpolator code which allows us to find accurate estimate of CL at a new angle of attack and reynolds number.

```cpp
#include <fstream>
#include <chrono>

#include "interpolation/interpolate.h"
#include "interpolation/json/include/nlohmann/json.hpp"

struct interpolate_fixture_ {
  interpolate_fixture_ () {
    std::ifstream operational_file(this->json_data_path, std::ios::ate);

    if (!operational_file.is_open() && operational_file.tellg() != 0) {
      throw std::runtime_error("Could not open file for reading or it is empty: " + this->json_data_path);
    }

    operational_file.seekg(0, std::ios::beg);

    const nlohmann::json json_data = nlohmann::json::parse(operational_file);
    try {
      this->CL = json_data.at("CL").get<std::vector<float>>();
      this->Re = json_data.at("Re").get<std::vector<float>>();
      this->alpha = json_data.at("alpha").get<std::vector<float>>();

      if (this->CL.size() != 5000) {
        std::cerr << "Current size is: " << this->CL.size() << std::endl;
        throw std::runtime_error("Expected size is 5000");
      }
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
      std::cerr << "ERROR while parsing JSON data" << std::endl;
      throw;
    }

    interpolator.add_training_data<5000>(this->CL, this->alpha, this->Re);
    this->CL.clear();
    this->Re.clear();
    this->alpha.clear();
  }

  float get_CL_at (const float new_alpha, const float new_reynolds_number) {
    auto start = std::chrono::high_resolution_clock::now();
    float result = interpolator.eval_at(new_alpha, new_reynolds_number);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CL: " << result << std::endl;
    std::cout << "Time taken to calc interpolation: " << duration.count() << " microseconds" << std::endl;
    return result;
    }

  std::vector<float> CL, Re, alpha;
  const std::string json_data_path = "assets/airfoil_data/naca652415_training.json";
  CONCEPTUAL_INTERPOLATE_H::interpolate<2, 5000, 5> interpolator = CONCEPTUAL_INTERPOLATE_H::interpolate<2, 5000, 5>();
}

int main () {
  auto test = interpolate_fixture_();
  float new_alpha = 5.0f;
  float new_reynolds_number = 1000000.0f;
  test.get_CL_at(new_alpha, new_reynolds_number);
}
```

Following is the output of the above example:

```bash
CL: 0.892368
Time taken to calc interpolation: 35 microseconds
```
