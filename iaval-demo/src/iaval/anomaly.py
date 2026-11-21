"""Anomaly scores: naive vs instrument- and uncertainty-aware."""
import numpy as np
from scipy import stats


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def naive_pixel_score(counts):
    """z-score of counts against the sky mean, ignoring instrument effects."""
    return (counts - counts.mean()) / counts.std()


def aware_pixel_score(counts, mu):
    """Pearson residual against the systematics model, rescaled by an empirical
    (robust) dispersion so that extra scatter is propagated as uncertainty."""
    z = (counts - mu) / np.sqrt(mu)
    return z / mad_sigma(z)


def smooth_score(score, nside, radius_deg=5.0):
    """Matched-filter style disc average, so extended structures add up (score / sqrt(N))."""
    import healpy as hp
    out = np.empty_like(score)
    r = np.deg2rad(radius_deg)
    for p in range(len(score)):
        disc = hp.query_disc(nside, hp.pix2vec(nside, p), r)
        out[p] = score[disc].sum() / np.sqrt(len(disc))
    return out


def naive_object_score(colours):
    """Mahalanobis distance with the sample covariance (ignores per-object errors)."""
    mu = colours.mean(0)
    C = np.cov(colours.T)
    d = colours - mu
    return np.einsum("ij,jk,ik->i", d, np.linalg.inv(C), d)


def aware_object_score(colours, sigma):
    """Error-aware chi-square: d^T (C_int + E_i)^-1 d, with C_int estimated by subtracting the mean error covariance.
    Uses medians/MAD-like robust centre so the injected tail does not bias the estimate."""
    mu = np.median(colours, axis=0)
    d = colours - mu
    C_obs = np.cov(colours.T)
    E_mean = np.diag((sigma ** 2).mean(0))
    C_int = C_obs - E_mean
    w, v = np.linalg.eigh(C_int)
    C_int = (v * np.clip(w, 1e-4, None)) @ v.T
    chi2 = np.empty(len(colours))
    for i0 in range(0, len(colours), 20000):
        sl = slice(i0, i0 + 20000)
        C = C_int[None] + np.einsum("ni,ij->nij", sigma[sl] ** 2, np.eye(3))
        Ci = np.linalg.inv(C)
        chi2[sl] = np.einsum("ni,nij,nj->n", d[sl], Ci, d[sl])
    return chi2


def chi2_to_sigma(chi2, dof=3):
    p = stats.chi2.sf(chi2, dof)
    return stats.norm.isf(np.clip(p, 1e-300, 0.5))
