/**
 * @file interpolate.h
 * @brief Header file defining the interpolate class for k-nearest points average interpolation.
 *
 * @author Harshavardhan Karnati
 * @date 07/03/2024
 */

#ifndef CONCEPTUAL_INTERPOLATE_H
#define CONCEPTUAL_INTERPOLATE_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <concepts>
#include <queue>
#include <utility>

#include "eigen/Eigen/Core"
#include "eigen/Eigen/Dense"

/**
 * @brief Compile-time check if the given type is a std::vector<float> or not.
 * @tparam T Type to check
 */
template <class... T>
inline constexpr bool check_vector = ((std::is_same_v<std::remove_cvref_t<T>, std::vector<float>>) && ...);

/**
 * @brief Compile-time check if the given type is a std::array<float, N> or not.
 * @tparam N Size of the array
 * @tparam T Type to check
 */
template <std::size_t N, class... T>
inline constexpr bool check_array = ((std::is_same_v<std::remove_cvref_t<T>, std::array<float, N>>) && ...);

/**
 * @brief Compile-time check if the given type is a (std::array<float, N> or std::vector<float>) or not.
 * @tparam N Size of the array
 * @tparam T Type to check
 */
template <std::size_t N, class... T>
inline constexpr bool check_vector_array = check_vector<T...> || check_array<N, T...>;

/**
 * @brief Compile-time check if the given type is a Eigen::Vector<float, N> or not.
 * @tparam N Size of the Eigen::Vector
 * @tparam T Type to check
 */
template <std::size_t N, class... T>
inline constexpr bool check_eigen = ((std::is_same_v<std::remove_cvref_t<T>, Eigen::Vector<float, N>>) && ...);

/**
 * @brief Compile-time check if the given type is a (Eigen::Vector<float, N> or std::array<float, N> or std::vector<float>)
 * or not
 * @tparam N Size of the Eigen::Vector or std::array
 * @tparam T Type to check
 */
template <std::size_t N, class... T>
inline constexpr bool check_all = check_eigen<N, T...> || check_vector<T...> || check_array<N, T...>;




/**
 * @class interpolate
 * @brief Interpolator class for k-nearest points average interpolation.
 *
 * @details The interpolate class provides functionality for k-nearest points average interpolation
 * (k-value is given by mean_size). It allows users to add training data, evaluate the function at
 * a specified point using various input formats, and includes methods optimised for different data sizes.
 *
 * @tparam dimension The dimensionality of the input space.
 * @tparam max_training_data_size The maximum size allocated for training data storage.
 * @tparam mean_size The size of the subset used to calculate the mean during interpolation.
 *
 * @note Uses Eigen library for efficient linear algebra operations. Ensure Eigen3 is properly installed
 * and included in the project for optimal performance. By default, the code assumes the path is local.
 */
template <std::size_t dimension, std::size_t max_training_data_size, std::size_t mean_size>
class interpolate {
public:

    /**
     * @brief Default constructor for the interpolate class.
     */
    interpolate () = default;


    /**
     * @brief Adds training data to the interpolator.
     *
     * @details This function adds training data, consisting of function values and corresponding variable values,
     * to the interpolator. The function values are represented by the Eigen::Vector @p IN_FUNC_DATA,
     * and the variable values are represented by the Eigen::Matrix @p IN_VAR_DATA. Both function and
     * variable data are added to the interpolator's internal storage.
     *
     * @tparam current_training_size The size of the function data vector and the number of rows in the variable data matrix.
     *
     * @param [in] IN_FUNC_DATA The function values to be added.
     * @param [in] IN_VAR_DATA The corresponding variable values to be added.
     *
     * @throw std::runtime_error Thrown if the total size of the existing data and the new data exceeds the maximum allocated size.
     *
     * @note This method updates the internal state of the interpolator and increases the total number of training points.
     */
    template<std::size_t current_training_size>
    void add_training_data (const Eigen::Vector<float, current_training_size> &IN_FUNC_DATA,
                            const Eigen::Matrix<float, current_training_size, dimension> &IN_VAR_DATA) {

        // check if max training size is reach
        if (this->interpolated_data_size + current_training_size > max_training_data_size) {
            std::cerr << "Current size: " << this->interpolated_data_size
                      << ", Attempted size: " << this->interpolated_data_size + current_training_size
                      << ", Maximum size: " << max_training_data_size << std::endl;
            throw std::runtime_error("Input Training data size exceeds max size");
        }

        this->func_data.segment<current_training_size>(this->interpolated_data_size) = IN_FUNC_DATA;
        this->var_data.block<current_training_size, dimension>(this->interpolated_data_size, 0) = IN_VAR_DATA;

        this->interpolated_data_size += current_training_size;
        if (this->interpolated_data_size == max_training_data_size)
            std::cout << "max training data size reached. No more points are accepted" << std::endl;
    }


    /**
     * @brief Adds training data to the interpolator with variadic input for variable values.
     *
     * @details This function adds training data to the interpolator, where the function values are
     * represented by the Eigen::Vector @p IN_FUNC_DATA, and the variable values are represented by the
     * variadic parameter pack @p IN_VAR_DATAs. The number of variable values must match the specified
     * dimension, and each variable value in the pack must be of type Eigen::Vector<float, current_training_size>
     *
     * @tparam current_training_size The size of the function data vector.
     * @tparam var_type Variadic template parameters enforcing Eigen::Vector<float, current_training_size>
     *
     * @param [in] IN_FUNC_DATA The function values to be added.
     * @param [in] IN_VAR_DATAs Variadic input representing the variable values to be added.
     *
     * @throw std::runtime_error Thrown if the total size of the existing data and the new data exceeds the maximum
     * allocated size.
     *
     * @note This method updates the internal state of the interpolator and increases the total number of training points.
     * @note This method copies the values. It is a good idea to clear the data in the higher scope or use mapped version
     */
    template<std::size_t current_training_size, class func_type, class... var_type>
    requires (check_eigen<current_training_size, func_type, var_type...>)
    void add_training_data (const func_type &IN_FUNC_DATA, const var_type&... IN_VAR_DATAs) {
        // check if max training size is reach
        if (this->interpolated_data_size + current_training_size > max_training_data_size) {
            std::cerr << "Current size: " << this->interpolated_data_size
                      << ", Attempted size: " << this->interpolated_data_size + current_training_size
                      << ", Maximum size: " << max_training_data_size << std::endl;
            throw std::runtime_error("Input Training data size exceeds max size");
        }

        this->func_data.template segment<current_training_size>(this->interpolated_data_size) = IN_FUNC_DATA;

        auto add = [this] <std::size_t... i>(const var_type&... IN_VAR_DATAs_, std::index_sequence<i...>) {
            auto add_at = [this] <std::size_t i_> (const Eigen::Vector<float, current_training_size> &IN_VAR_DATA_) {
                this->var_data.template block<current_training_size, 1>(this->interpolated_data_size, i_) = IN_VAR_DATA_;
            };
            (add_at.template operator()<i>(IN_VAR_DATAs_),...);
        };

        add(IN_VAR_DATAs..., std::make_index_sequence<dimension>{});

        this->interpolated_data_size += current_training_size;
        if (this->interpolated_data_size == max_training_data_size)
            std::cout << "max training data size reached. No more points are accepted" << std::endl;
    }


    /**
     * @brief Adds training data to the interpolator with variadic input for variable values.
     *
     * @details This function adds training data to the interpolator, where the function values are
     * represented by the Eigen::Vector @p IN_FUNC_DATA, and the variable values are represented by the
     * variadic parameter pack @p IN_VAR_DATAs. The number of variable values must match the specified
     * dimension, and each variable value in the pack must be of type  or std::vector or
     * std::array<float, current_training_size>.
     *
     * @tparam current_training_size The size of the function data vector.
     * @tparam var_type Variadic template parameters enforcing types std::vector or
     * std::array<float, current_training_size>.
     *
     * @param [in] IN_FUNC_DATA The function values to be added.
     * @param [in] IN_VAR_DATAs Variadic input representing the variable values to be added.
     *
     * @throw std::runtime_error Thrown if the total size of the existing data and the new data exceeds the maximum
     * allocated size.
     *
     * @note This method updates the internal state of the interpolator and increases the total number of training points.
     * @note This method copies the values. It is a good idea to clear the data in the higher scope or use mapped version
     */
    template<std::size_t current_training_size, class func_type, class... var_type>
    requires ((sizeof...(var_type) == dimension) && (check_vector_array<current_training_size, func_type, var_type...>))
    void add_training_data (func_type &IN_FUNC_DATA, var_type&... IN_VAR_DATAs) {

        auto ensure_eigen_format = [] <class T> (const T& container) -> decltype(auto) {
            if constexpr (check_vector_array<current_training_size, T>) {
                if (container.size() != current_training_size) {
                    std::cerr << "Required size: " << dimension << ", Given size: " << container.size() << std::endl;
                    throw std::invalid_argument("Container must have a size of 'dimension");
                }
                auto* ptr = const_cast<float*>(container.data());
                return static_cast<Eigen::Vector<float, current_training_size>>(Eigen::Map<Eigen::Vector<float, current_training_size>>(ptr));
            } else if constexpr (check_eigen<current_training_size, T>){
                return container;
            } else {
                std::cerr << "Only works with std::vector or std::array. Unknown data-type given" << std::endl;
                throw std::invalid_argument("Only works with std::vector or std::array. Unknown data-type given");
            }
        };

        this->add_training_data<current_training_size>(ensure_eigen_format(IN_FUNC_DATA), ensure_eigen_format(IN_VAR_DATAs)...);
    }



    /**
     * @brief Evaluates the function at a specified point using k-nearest points average interpolation.
     *
     * @details This function calculates the interpolated value of the function at the given point
     * @p IN_POINT using k-nearest points average interpolation technique. The distance between
     * @p IN_POINT and training data points is computed, and the mean of the k-nearest values is
     * returned. The choice of k depends on the size of the training data, and different algorithms
     * are used for efficiency.
     *
     * @param [in] IN_POINT The input point to interpolate at, represented by an Eigen::Vector<float, dimension>.
     * @return Interpolated value at the specified point.
     *
     * @throw None
     *
     * @note This method is marked as `noexcept` for better performance optimisations.
     */
    float eval_at (const Eigen::Vector<float, dimension> &IN_POINT) noexcept {
        Eigen::Vector<float, max_training_data_size> distance_vector = this->calculate_distance_vector(IN_POINT, this->var_data);
        if constexpr (max_training_data_size < 32) {
            return this->find_mean_of_first_k_minCoeff(distance_vector);
        }
        return this->find_mean_of_first_minCoeff_using_nth(distance_vector, this->func_data);    // can be replaced with find_mean_of_first_minCoeff_using_que
    }

    /**
     * @brief Evaluates the function at a specified point using k-nearest points average interpolation.
     *
     * @details This overload of the `eval_at` function allows users to provide the input point @p IN_POINT
     * as a container, either std::vector<float> or std::array<float, dimension>. It checks if the
     * size of the container matches the specified dimension and then internally maps the container
     * data to an Eigen::Vector for further interpolation avoiding additional copies with minimal performance
     * overhead.
     *
     * @tparam type The type of container, enforced at compile-time to be std::vector<float> or std::array<float, dimension>
     * @param [in] IN_POINT The input point to interpolate at, represented by a compatible container.
     * @return Interpolated value at the specified point.
     *
     * @throw std::invalid_argument Thrown if the size of the container does not match the specified dimension.
     *
     * @note This method is **NOT** marked as `noexcept`. `IN_POINT` as a std::vector has no compile-time resource to
     * get size. Thus, this overload is not marked as 'noexcept' and might be inferior in terms of performance and
     * optimisation compared to other overloads.
     */
    template <class type>
    requires (std::is_same_v<std::remove_cvref_t<type>, std::vector<float>> ||
              std::is_same_v<std::remove_cvref_t<type>,std::array<float, dimension>>)
    float eval_at (const type &IN_POINT) {
        // checks if input size matches dimension
        if (IN_POINT.size() != dimension) {
            std::cerr << "Required size: " << dimension << ", Given size: " << IN_POINT.size() << std::endl;
            throw std::invalid_argument("IN_POINT must have a size of 'dimension");
        }
        auto* ptr = const_cast<float*>(IN_POINT.data());
        return this->eval_at(static_cast<Eigen::Vector<float, dimension>>(Eigen::Map<Eigen::Vector<float, dimension>>(ptr)));
//        return this->eval_at();
    }


    /**
     * @brief Evaluates the function at a specified point using k-nearest points average interpolation with variadic arguments.
     *
     * @details This overload of the `eval_at` function allows users to directly input the variables as variadic arguments.
     * The arguments are expected to be floating-point values, and their number must match the specified dimension.
     * The function internally forwards the arguments to an Eigen::Vector for further interpolation avoiding additional
     * copies with minimal performance overhead.
     *
     * @tparam types Variadic template for IN_POINT. Compile-time enforced to be of size `dimension` and floating-point type.
     * @param [in] IN_POINT Perfect forwarding referenced variables. Input point to interpolate at.
     * @return Interpolated value at the specified point.
     *
     * @throw None
     *
     * @note This method is marked as `noexcept` for better performance optimisations.
     */
    template <class... types>
    requires ((sizeof...(types) == dimension) && (std::is_floating_point_v<std::remove_cvref_t<types>> && ...))
    float eval_at (types&&... IN_POINT) noexcept {
        return this->eval_at(std::array<float, dimension>{std::forward<types>(IN_POINT)...});
    }

private:
    /** @brief Tracks number of training data inputted in `interpolate::func_data` and `interpolate::var_data` */
    std::size_t interpolated_data_size = 0;

    /** @brief Stores function values at the `interpolate::var_data` */
    Eigen::Vector<float, max_training_data_size> func_data;

    /** @brief Stores training data points */
    Eigen::Matrix<float, max_training_data_size, dimension> var_data;


    /**
     * @brief Calculates the squared Euclidean distance vector between a point and a matrix of points.
     *
     * @details This function computes the squared Euclidean distance vector between a specified point
     * @p IN_POINT and a matrix of points @p IN_MATRIX. The result is a column vector where each element
     * represents the squared distance between the input point and the corresponding row in the matrix.
     *
     * @param [in] IN_POINT The input point for distance calculation, represented by an Eigen::Vector<float, dimension>.
     * Needs to be within the bounds of hypercube [min(training_data), max(training_data)]
     * @param [in] IN_MATRIX The matrix of points for distance comparison, represented by an
     * Eigen::Matrix<float, max_training_data_size, dimension>. All points need to be within the bounds of hypercube
     * [min(training_data), max(training_data)]. `IN_MATRIX` is usually interpolate::var_data.
     *
     * @return a Vector of size `max_training_data_size` with distance value
     *
     * @throw None
     *
     * @note This method is marked as `noexcept` for better performance optimisations.
     */
    Eigen::Matrix<float, max_training_data_size, 1>
            calculate_distance_vector (const Eigen::Vector<float, dimension> &IN_POINT,
                                       const Eigen::Matrix<float, max_training_data_size, dimension> &IN_MATRIX) noexcept {
        return (((IN_POINT.transpose()).template replicate<max_training_data_size, 1>()) - IN_MATRIX).rowwise().squaredNorm();
    }


    /**
     * @brief Finds the mean of first `mean_size` minimum values in `IN_VALS` using Eigen::Vector::minCoeff().
     * Other versions of this method exists.
     *
     * @details This function calculates the mean of the first `mean_size` minimum values in the input vector @p IN_VALS.
     * The implementation is optimised for a smaller data size (max_training_data_size), providing efficient
     * computation with minimal memory overhead from other implementations.
     *
     * @param [in, out] IN_VALS The input vector of values to find the mean of. The size should `max_training_data_size` and needs
     * to available in compile time. The IN_VALS is modified (injects NaN in places of first `mean_size` minimum values.
     *
     * @return mean of the first `mean_size` minimum values in the input vector.
     *
     * @throw None
     *
     * @note This method is marked as `noexcept` for better performance optimisations.
     * @note This version is optimized for smaller max_training_data_size. Different versions exist for other data sizes.
     */
    float find_mean_of_first_k_minCoeff (Eigen::Vector<float, max_training_data_size> &IN_VALS) noexcept {
        float sum = 0.0f;
        Eigen::Index index_;
        for (std::size_t i = 0; i < mean_size; i++) {
            sum += IN_VALS.minCoeff(&index_);
            IN_VALS(index_) = std::numeric_limits<float>::quiet_NaN();
        }
        return sum / static_cast<float>(mean_size);
    }


    /**
     * @brief Finds the mean of first `mean_size` minimum values in `IN_VALS` using `std::priority_queue`.
     *
     * @details This function calculates the mean of the first `mean_size` minimum values in the input vector @p IN_VALS
     * using a priority queue for efficient retrieval of the minimum values. The implementation is tailored
     * for a large data sizes (max_training_data_size).
     *
     * @param [in] IN_VALS The input vector of values to find the mean of. The size should `max_training_data_size` and needs
     * to available in compile time.
     *
     * @return mean of the first `mean_size` minimum values in the input vector.
     *
     * @throw None
     *
     * @note This method is marked as `noexcept` for better performance optimisations.
     * @note This version utilises a priority queue for efficient retrieval of minimum values from a large Eigen::Vector.
     */
    float find_mean_of_first_k_minCoeff_using_que (Eigen::Vector<float, max_training_data_size> &IN_VALS) noexcept {
        float sum = 0.0f;
        std::size_t i = 0;
        std::priority_queue que(IN_VALS.begin(), IN_VALS.end(), std::greater<>());
        while (i++ < mean_size) {
            sum += que.top();
            que.pop();
        }
        return sum / static_cast<float>(mean_size);
    }


    /**
     * @brief Finds the mean of first `mean_size` minimum values in `IN_VALS` using `std::nth_element`
     *
     * @details This function calculates the mean of the first `mean_size` minimum values in the input vector @p IN_VALS
     * using a std::nth_element method for efficient retrieval of the minimum values. The implementation is tailored
     * for a large data sizes (max_training_data_size).
     *
     * @param [in] IN_VALS The input vector of values to find the mean of. The size should `max_training_data_size` and needs
     * to available in compile time.
     *
     * @return mean of the first `mean_size` minimum values in the input vector.
     *
     * @throw None
     *
     * @note This method is marked as `noexcept` for better performance optimisations.
     * @note This version utilises a std::nth_element for efficient retrieval of minimum values from a large Eigen::Vector.
     */
    float find_mean_of_first_minCoeff_using_nth (Eigen::Vector<float, max_training_data_size> &IN_VALS, Eigen::Vector<float, max_training_data_size> &IN_FUNC) noexcept {
        std::vector<std::size_t> indices(IN_VALS.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::nth_element(indices.begin(), indices.begin() + mean_size, indices.end(),
                         [&IN_VALS] (const std::size_t i , const std::size_t j) -> bool {return IN_VALS(i) < IN_VALS(j);});

        return IN_FUNC(std::vector<std::size_t>(indices.begin(), indices.begin() + mean_size)).sum() / static_cast<float>(mean_size);
    }
};


#endif //CONCEPTUAL_INTERPOLATE_H
