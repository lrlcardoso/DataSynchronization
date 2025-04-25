"""
Signal processing utilities: resampling, normalization, correlation.
"""

def resample_signal(signal, orig_freq, target_freq):
    """
    Resample signal to target frequency.
    """
    # TODO: Implement resampling (e.g., using scipy.interpolate or pandas)
    pass

def normalize_signal(signal, method):
    """
    Normalize signal ('zscore' or 'minmax').
    """
    # TODO: Implement normalization
    pass

def compute_cross_correlation(sig1, sig2, fs, lag_range):
    """
    Compute cross-correlation and find the optimal lag.
    Returns (lag_samples, max_corr, similarity_curve)
    """
    # TODO: Implement cross-correlation over allowed lags
    pass
