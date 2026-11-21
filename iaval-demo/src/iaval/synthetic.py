"""Synthetic DESI-like survey with known systematics and known injected anomalies.

Everything here is simulated so the pipeline can be tested against ground truth.
The same functions that consume this data take any catalogue with the same columns,
so a real DESI DR1 / TOPCAT export can be dropped in (see README).
"""
import numpy as np
import healpy as hp


def smooth_field(nside, rng, slope=-2.0, amp=1.0):
    """Gaussian random field on the sphere with a power-law spectrum."""
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    cl = np.zeros_like(ell, dtype=float)
    cl[1:] = ell[1:] ** slope
    m = hp.synfast(cl, nside, lmax=lmax, pol=False, verbose=False) if "verbose" in hp.synfast.__code__.co_varnames else hp.synfast(cl, nside, lmax=lmax, pol=False)
    m = (m - m.mean()) / m.std()
    return amp * m


def make_survey(nside=32, base_density=300.0, n_true_clusters=6, seed=0):
    """Return dict with pixel-level maps, truth masks and an object catalogue."""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    depth = smooth_field(nside, rng, slope=-2.0, amp=0.35)      # imaging depth offset (mag-like)
    seeing = smooth_field(nside, rng, slope=-1.5, amp=0.25)     # seeing offset
    # real large-scale structure: a few compact overdensities (the "discoveries")
    true_mask = np.zeros(npix, bool)
    boost = np.ones(npix)
    centres = rng.choice(npix, n_true_clusters, replace=False)
    for c in centres:
        vec = hp.pix2vec(nside, c)
        disc = hp.query_disc(nside, vec, radius=np.deg2rad(5.0))
        boost[disc] *= 1.6
        true_mask[disc] = True
    # instrument response: more galaxies recovered where the imaging is deeper, fewer in poor seeing
    beta_d, beta_s = 0.9, -0.5
    lam = base_density * np.exp(beta_d * depth + beta_s * seeing) * boost
    counts = rng.poisson(lam)
    return dict(nside=nside, depth=depth, seeing=seeing, counts=counts, lam=lam,
                true_mask=true_mask, centres=centres, rng=rng)


def make_catalogue(survey, n_obj=150_000, n_anom=250, seed=1):
    """Object-level colours with heteroscedastic errors that grow where the imaging is shallow."""
    rng = np.random.default_rng(seed)
    nside = survey["nside"]
    npix = hp.nside2npix(nside)
    pix = rng.integers(0, npix, n_obj)
    depth = survey["depth"][pix]
    # intrinsic colour distribution (g-r, r-z, z-W1): correlated population
    mean = np.array([0.8, 0.9, 1.4])
    A = np.array([[0.10, 0.06, 0.04], [0.06, 0.12, 0.05], [0.04, 0.05, 0.15]])
    intrinsic = rng.multivariate_normal(mean, A, n_obj)
    sig = 0.05 * np.exp(-1.6 * depth)[:, None] * np.array([1.0, 1.2, 1.6])[None, :]   # shallow = noisier
    obs = intrinsic + rng.normal(size=(n_obj, 3)) * sig
    is_anom = np.zeros(n_obj, bool)
    idx = rng.choice(n_obj, n_anom, replace=False)
    shift = rng.choice([-1, 1], size=(n_anom, 3)) * rng.uniform(0.9, 1.4, size=(n_anom, 3))
    obs[idx] += shift
    is_anom[idx] = True
    return dict(pix=pix, colours=obs, sigma=sig, is_anom=is_anom)
