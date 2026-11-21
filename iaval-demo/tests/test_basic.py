import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from iaval import synthetic, systematics, anomaly


def test_systematics_model_recovers_expected_counts():
    s = synthetic.make_survey(16, seed=3, n_true_clusters=0)
    mu, _, _ = systematics.robust_fit(s["counts"], [s["depth"], s["seeing"]])
    z = (s["counts"] - mu) / np.sqrt(mu)
    assert abs(z.mean()) < 0.2 and 0.8 < z.std() < 1.3


def test_aware_object_score_is_null_like_for_normal_objects():
    s = synthetic.make_survey(16, seed=3)
    cat = synthetic.make_catalogue(s, n_obj=40000, n_anom=0)
    sig = anomaly.chi2_to_sigma(anomaly.aware_object_score(cat["colours"], cat["sigma"]))
    assert (sig > 4).mean() < 5e-4


def test_aware_beats_naive_in_shallow_regions():
    s = synthetic.make_survey(16, seed=3)
    cat = synthetic.make_catalogue(s, n_obj=60000, n_anom=100)
    naive = anomaly.chi2_to_sigma(anomaly.naive_object_score(cat["colours"]))
    aware = anomaly.chi2_to_sigma(anomaly.aware_object_score(cat["colours"], cat["sigma"]))
    fp = lambda x: ((x > 4) & ~cat["is_anom"]).sum()
    assert fp(aware) <= fp(naive)
