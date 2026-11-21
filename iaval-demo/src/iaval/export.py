"""TOPCAT-ready exports: CSV with columns a human can inspect and a plain-language reason."""
import numpy as np
import pandas as pd
import healpy as hp


def pixel_table(nside, counts, mu, naive, aware, depth, seeing, truth=None):
    ra, dec = hp.pix2ang(nside, np.arange(len(counts)), lonlat=True)
    df = pd.DataFrame(dict(healpix=np.arange(len(counts)), ra=ra, dec=dec, counts=counts,
                           expected_from_systematics=mu, depth_offset=depth, seeing_offset=seeing,
                           naive_score=naive, aware_score=aware))
    df["why"] = np.where(
        (df.naive_score > 3) & (df.aware_score <= 3), "explained by imaging depth/seeing (naive false positive)",
        np.where(df.aware_score > 3, "excess beyond systematics model", ""))
    if truth is not None:
        df["is_injected"] = truth
    return df.sort_values("aware_score", ascending=False)


def object_table(cat, naive_chi2, aware_sigma, top=500):
    df = pd.DataFrame(dict(healpix=cat["pix"], g_r=cat["colours"][:, 0], r_z=cat["colours"][:, 1],
                           z_w1=cat["colours"][:, 2], err_g_r=cat["sigma"][:, 0], err_r_z=cat["sigma"][:, 1],
                           err_z_w1=cat["sigma"][:, 2], naive_chi2=naive_chi2, aware_sigma=aware_sigma))
    return df.sort_values("aware_sigma", ascending=False).head(top)
