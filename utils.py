import numpy as np

def remove_invalid_samples(features, labels):
    valid_indices = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    return features[valid_indices], labels[valid_indices]
