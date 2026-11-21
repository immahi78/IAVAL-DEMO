"""Instrument-aware modelling of pixel counts: remove the part explained by known systematics."""
import numpy as np
from sklearn.linear_model import PoissonRegressor


def fit_systematics(counts, templates, mask=None, alpha=1e-6):
    """Poisson GLM of counts on systematics templates (depth, seeing, ...).

    Returns expected counts mu and the fitted model. `mask` excludes pixels
    from the fit (e.g. already-suspect pixels, in a second robust pass).
    """
    X = np.column_stack(templates)
    fit_mask = np.ones(len(counts), bool) if mask is None else mask
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(X[fit_mask], counts[fit_mask])
    return model.predict(X), model


def robust_fit(counts, templates, n_iter=3, clip=3.0):
    """Iteratively exclude high-residual pixels so real structure does not bias the systematics fit."""
    mask = np.ones(len(counts), bool)
    for _ in range(n_iter):
        mu, model = fit_systematics(counts, templates, mask)
        z = (counts - mu) / np.sqrt(mu)
        mask = np.abs(z) < clip
    return mu, model, mask
