import numpy as np
from scipy.stats import gmean, hmean
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def calculate_embedding(embeddings, method, weights=None, trim_percentage=0.1):
    """
    Aggregate embeddings using the specified method.

    Parameters:
        embeddings (np.ndarray): Array of embeddings with shape (n_samples, n_dimensions).
        method (str): Aggregation method to use.
        weights (np.ndarray, optional): Weights for weighted methods. Defaults to None.
        trim_percentage (float, optional): Fraction to trim from each end for trimmed mean. Defaults to 0.1.

    Returns:
        np.ndarray: Aggregated embedding vector with shape (n_dimensions,).
    """
    if method == "average":
        return np.mean(embeddings, axis=0)
    elif method == "weighted_average":
        if weights is not None:
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError("Weights must be provided for weighted average.")
    elif method == "median":
        return np.median(embeddings, axis=0)
    elif method == "geometric_mean":
        return calculate_geometric_mean(embeddings)
    elif method == "harmonic_mean":
        return calculate_harmonic_mean(embeddings)
    elif method == "trimmed_mean":
        return calculate_trimmed_mean(embeddings, trim_percentage)
    elif method == "centroid":
        return calculate_centroid(embeddings)
    elif method == "pca":
        return calculate_pca(embeddings)
    elif method == "exemplar":
        return calculate_exemplar(embeddings)
    elif method == "max_pooling":
        return np.max(embeddings, axis=0)
    elif method == "min_pooling":
        return np.min(embeddings, axis=0)
    elif method == "entropy_weighted_average":
        return calculate_entropy_weighted_average(embeddings)
    elif method == "attentive_pooling":
        return calculate_attentive_pooling(embeddings)
    elif method == "tukeys_biweight":
        return calculate_tukeys_biweight(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_geometric_mean(embeddings):
    """
    Calculate the geometric mean of embeddings.

    Note:
        Geometric mean is only defined for positive numbers.
        This method will raise an error if embeddings contain non-positive values.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Geometric mean of embeddings.
    """
    if np.any(embeddings <= 0):
        raise ValueError("Geometric mean is only defined for positive numbers.")
    return gmean(embeddings, axis=0)

def calculate_harmonic_mean(embeddings):
    """
    Calculate the harmonic mean of embeddings.

    Note:
        Harmonic mean is only defined for positive numbers.
        This method will raise an error if embeddings contain non-positive values.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Harmonic mean of embeddings.
    """
    if np.any(embeddings <= 0):
        raise ValueError("Harmonic mean is only defined for positive numbers.")
    return hmean(embeddings, axis=0)

def calculate_trimmed_mean(embeddings, trim_percentage):
    """
    Calculate the trimmed mean of embeddings.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.
        trim_percentage (float): Fraction to trim from each end (0 <= trim_percentage < 0.5).

    Returns:
        np.ndarray: Trimmed mean of embeddings.
    """
    if not 0 <= trim_percentage < 0.5:
        raise ValueError("trim_percentage must be between 0 and less than 0.5.")
    n = embeddings.shape[0]
    lower = int(n * trim_percentage)
    upper = n - lower
    if lower >= upper:
        raise ValueError("Not enough data points to trim with the given trim_percentage.")
    sorted_embeddings = np.sort(embeddings, axis=0)
    trimmed_embeddings = sorted_embeddings[lower:upper]
    return np.mean(trimmed_embeddings, axis=0)

def calculate_centroid(embeddings):
    """
    Calculate the centroid of embeddings using KMeans clustering with one cluster.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Centroid of the embeddings.
    """
    kmeans = KMeans(n_clusters=1, random_state=0, n_init='auto').fit(embeddings)
    return kmeans.cluster_centers_[0]

def calculate_pca(embeddings):
    """
    Aggregate embeddings using PCA by projecting onto the first principal component.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Aggregated embedding vector reconstructed from the first principal component.
    """
    pca = PCA(n_components=1)
    pca.fit(embeddings)
    pc1 = pca.components_[0]  # First principal component
    projections = embeddings @ pc1  # Project embeddings onto pc1
    mean_projection = projections.mean()
    aggregated_embedding = mean_projection * pc1  # Reconstruct the aggregated embedding
    return aggregated_embedding

def calculate_exemplar(embeddings):
    """
    Select the exemplar embedding that minimizes the average cosine distance to other embeddings.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Exemplar embedding.
    """
    distances = cdist(embeddings, embeddings, metric='cosine')
    mean_distances = distances.mean(axis=1)
    idx = np.argmin(mean_distances)
    return embeddings[idx]

def calculate_entropy_weighted_average(embeddings):
    """
    Calculate the entropy-weighted average of embeddings.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Aggregated embedding.
    """
    # Shift embeddings to positive values
    min_val = embeddings.min()
    shifted_embeddings = embeddings - min_val + 1e-6  # Ensure all values are positive
    # Compute entropy along the feature dimension
    entropies = scipy_entropy(shifted_embeddings.T)
    # Normalize entropies to sum to 1
    weights = entropies / entropies.sum()
    return np.average(embeddings, axis=0, weights=weights)

def calculate_attentive_pooling(embeddings):
    """
    Calculate the attentive pooling of embeddings.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Aggregated embedding.
    """
    mean_embedding = np.mean(embeddings, axis=0)
    similarities = embeddings @ mean_embedding
    exp_similarities = np.exp(similarities - np.max(similarities))  # For numerical stability
    attention_weights = exp_similarities / exp_similarities.sum()
    return np.sum(embeddings * attention_weights[:, np.newaxis], axis=0)

def calculate_tukeys_biweight(embeddings):
    """
    Calculate Tukey's biweight of embeddings.

    Parameters:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        np.ndarray: Aggregated embedding.
    """
    median_embedding = np.median(embeddings, axis=0)
    diff = embeddings - median_embedding
    mad = np.median(np.abs(diff), axis=0)
    mad[mad == 0] = 1e-6  # Avoid division by zero
    u = diff / (9 * mad)
    u2 = u ** 2
    mask = u2 < 1
    weights = (1 - u2) ** 2
    weights[~mask] = 0
    numerator = np.sum(embeddings * weights[:, np.newaxis], axis=0)
    denominator = np.sum(weights)
    if denominator == 0:
        return median_embedding
    return numerator / denominator
