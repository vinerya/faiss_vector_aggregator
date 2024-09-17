
# Faiss Embeddings Aggregation Library

This Python library provides a suite of advanced methods for aggregating multiple embeddings associated with a single document or entity into a single representative embedding. It supports a wide range of aggregation techniques, from simple averaging to sophisticated methods like PCA and Attentive Pooling.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Example 1: Simple Average Aggregation](#example-1-simple-average-aggregation)
  - [Example 2: Weighted Average Aggregation](#example-2-weighted-average-aggregation)
  - [Example 3: Principal Component Analysis (PCA) Aggregation](#example-3-principal-component-analysis-pca-aggregation)
  - [Example 4: Centroid Aggregation (K-Means)](#example-4-centroid-aggregation-k-means)
  - [Example 5: Attentive Pooling Aggregation](#example-5-attentive-pooling-aggregation)
- [Aggregation Methods](#aggregation-methods)
- [Parameters](#parameters)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Simple Average**: Compute the arithmetic mean of embeddings.
- **Weighted Average**: Compute a weighted average of embeddings.
- **Geometric Mean**: Compute the geometric mean across embeddings (for positive values).
- **Harmonic Mean**: Compute the harmonic mean across embeddings (for positive values).
- **Centroid (K-Means)**: Use K-Means clustering to find the centroid of the embeddings.
- **Principal Component Analysis (PCA)**: Use PCA to reduce embeddings to a single representative vector.
- **Median**: Compute the element-wise median of embeddings.
- **Trimmed Mean**: Compute the mean after trimming outliers.
- **Max-Pooling**: Take the maximum value for each dimension across embeddings.
- **Min-Pooling**: Take the minimum value for each dimension across embeddings.
- **Entropy-Weighted Average**: Weight embeddings by their entropy (information content).
- **Attentive Pooling**: Use an attention mechanism to learn the weights for combining embeddings.
- **Tukey's Biweight**: A robust method to down-weight outliers.
- **Exemplar**: Select the embedding that best represents the group by minimizing average distance.

## Installation

To install the package, you can use pip:

```bash
pip install faiss_vector_aggregator
```

## Usage

Below are examples demonstrating how to use the library to aggregate embeddings using different methods.

### Example 1: Simple Average Aggregation

Suppose you have a collection of embeddings stored in a FAISS index, and you want to aggregate them by their associated document IDs using simple averaging.

```python
from faiss_vector_aggregator import aggregate_embeddings

# Aggregate embeddings using simple averaging
aggregate_embeddings(
    input_folder="data/input",
    column_name="id",
    output_folder="data/output",
    method="average"
)
```

- **Parameters:**
  - `input_folder`: Path to the folder containing the input FAISS index and metadata.
  - `column_name`: The metadata field by which to aggregate embeddings (e.g., `'id'`).
  - `output_folder`: Path where the output FAISS index and metadata will be saved.
  - `method="average"`: Specifies the aggregation method.

### Example 2: Weighted Average Aggregation

If you have different weights for the embeddings, you can apply a weighted average to give more importance to certain embeddings.

```python
from faiss_vector_aggregator import aggregate_embeddings

# Example weights for the embeddings
weights = [0.1, 0.3, 0.6]

# Aggregate embeddings using weighted averaging
aggregate_embeddings(
    input_folder="data/input",
    column_name="id",
    output_folder="data/output",
    method="weighted_average",
    weights=weights
)
```

- **Parameters:**
  - `weights`: A list or array of weights corresponding to each embedding.
  - `method="weighted_average"`: Specifies the weighted average method.

### Example 3: Principal Component Analysis (PCA) Aggregation

To reduce high-dimensional embeddings to a single representative vector using PCA:

```python
from faiss_vector_aggregator import aggregate_embeddings

# Aggregate embeddings using PCA
aggregate_embeddings(
    input_folder="data/input",
    column_name="id",
    output_folder="data/output",
    method="pca"
)
```

- **Parameters:**
  - `method="pca"`: Specifies that PCA should be used for aggregation.

### Example 4: Centroid Aggregation (K-Means)

Use K-Means clustering to find the centroid of embeddings for each document ID.

```python
from faiss_vector_aggregator import aggregate_embeddings

# Aggregate embeddings using K-Means clustering to find the centroid
aggregate_embeddings(
    input_folder="data/input",
    column_name="id",
    output_folder="data/output",
    method="centroid"
)
```

- **Parameters:**
  - `method="centroid"`: Specifies that K-Means clustering should be used.

### Example 5: Attentive Pooling Aggregation

To use an attention mechanism for aggregating embeddings:

```python
from faiss_vector_aggregator import aggregate_embeddings

# Aggregate embeddings using Attentive Pooling
aggregate_embeddings(
    input_folder="data/input",
    column_name="id",
    output_folder="data/output",
    method="attentive_pooling"
)
```

- **Parameters:**
  - `method="attentive_pooling"`: Specifies the attentive pooling method.

## Aggregation Methods

Below is a detailed description of each aggregation method supported by the library:

- **average**: Compute the arithmetic mean of embeddings.
- **weighted_average**: Compute a weighted average of embeddings. Requires `weights`.
- **geometric_mean**: Compute the geometric mean across embeddings. Only for positive values.
- **harmonic_mean**: Compute the harmonic mean across embeddings. Only for positive values.
- **median**: Compute the element-wise median of embeddings.
- **trimmed_mean**: Compute the mean after trimming a percentage of outliers. Use `trim_percentage` parameter.
- **centroid**: Use K-Means clustering to find the centroid of the embeddings.
- **pca**: Use Principal Component Analysis to project embeddings onto the first principal component.
- **exemplar**: Select the embedding that minimizes the average cosine distance to others.
- **max_pooling**: Take the maximum value for each dimension across embeddings.
- **min_pooling**: Take the minimum value for each dimension across embeddings.
- **entropy_weighted_average**: Weight embeddings by their entropy (information content).
- **attentive_pooling**: Use an attention mechanism based on similarity to aggregate embeddings.
- **tukeys_biweight**: A robust method to down-weight outliers in the embeddings.

## Parameters

- `input_folder` (str): Path to the folder containing the input FAISS index (`index.faiss`) and metadata (`index.pkl`).
- `column_name` (str): The metadata field by which to aggregate embeddings (e.g., `'id'`).
- `output_folder` (str): Path where the output FAISS index and metadata will be saved.
- `method` (str): The aggregation method to use. Options include:
  - `'average'`, `'weighted_average'`, `'geometric_mean'`, `'harmonic_mean'`, `'centroid'`, `'pca'`, `'median'`, `'trimmed_mean'`, `'max_pooling'`, `'min_pooling'`, `'entropy_weighted_average'`, `'attentive_pooling'`, `'tukeys_biweight'`, `'exemplar'`.
- `weights` (list or np.ndarray, optional): Weights for the `weighted_average` method.
- `trim_percentage` (float, optional): Fraction to trim from each end for `trimmed_mean`. Should be between 0 and less than 0.5.
- `weights` (list or np.ndarray, optional): Weights for the `weighted_average` method.

## Dependencies

Ensure you have the following packages installed:

- **faiss**: For handling FAISS indexes.
- **numpy**: For numerical computations.
- **scipy**: For statistical functions.
- **scikit-learn**: For PCA and K-Means clustering.
- **langchain**: For handling document stores and vector stores.

You can install the dependencies using:

```bash
pip install faiss-cpu numpy scipy scikit-learn langchain
```

*Note:* Replace `faiss-cpu` with `faiss-gpu` if you prefer to use the GPU version of FAISS.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on the [GitHub repository](https://github.com/vinerya/faiss_vector_aggregator).

When contributing, please ensure that your code adheres to the following guidelines:

- Follow PEP 8 coding standards.
- Include docstrings and comments where necessary.
- Write unit tests for new features or bug fixes.
- Update the documentation to reflect changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Additional Notes

- **Usage with LangChain:**
  - This library is compatible with LangChain's `FAISS` vector store. Ensure that your embeddings and indexes are handled consistently when integrating with LangChain.