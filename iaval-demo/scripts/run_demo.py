"""End-to-end demo: simulate, detect (naive vs aware), score against truth, plot, export."""
import os, sys
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from iaval import synthetic, systematics, anomaly, export

OUT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(OUT, "docs", "figures")
NSIDE = 32
THR = 3.0


def main():
    s = synthetic.make_survey(NSIDE, seed=0)
    counts, truth = s["counts"], s["true_mask"]

    # --- pixel level -------------------------------------------------------
    naive = anomaly.smooth_score(anomaly.naive_pixel_score(counts), NSIDE)
    mu, model, _ = systematics.robust_fit(counts, [s["depth"], s["seeing"]])
    aware_raw = anomaly.aware_pixel_score(counts, mu)
    aware = anomaly.smooth_score(aware_raw, NSIDE)
    naive /= naive.std(); aware /= aware.std()   # compare on a common footing: sigma units of the score distribution

    # pixels within the matched-filter radius of a true cluster count as "near truth" (smoothing halo), not false alarms
    near = np.zeros(len(truth), bool)
    for p_ in np.where(truth)[0]:
        near[hp.query_disc(NSIDE, hp.pix2vec(NSIDE, p_), np.deg2rad(5.0))] = True

    def stats(score):
        flag = score > THR
        tp = (flag & truth).sum(); fp = (flag & ~near).sum()
        return dict(flagged=int(flag.sum()), true_pos_pixels=int(tp), halo_pixels=int((flag & near & ~truth).sum()),
                    false_pos_pixels=int(fp), recall=round(float(tp / truth.sum()), 2))
    pn, pa = stats(naive), stats(aware)

    # --- object level ------------------------------------------------------
    cat = synthetic.make_catalogue(s)
    c_naive = anomaly.naive_object_score(cat["colours"])
    c_aware = anomaly.aware_object_score(cat["colours"], cat["sigma"])
    s_naive = anomaly.chi2_to_sigma(c_naive); s_aware = anomaly.chi2_to_sigma(c_aware)
    shallow = s["depth"][cat["pix"]] < np.quantile(s["depth"], 0.25)
    def ostats(sig, thr=4.0):
        flag = sig > thr
        return dict(flagged=int(flag.sum()), true_pos=int((flag & cat["is_anom"]).sum()),
                    false_pos=int((flag & ~cat["is_anom"]).sum()),
                    false_pos_in_shallowest_quartile=int((flag & ~cat["is_anom"] & shallow).sum()))
    on, oa = ostats(s_naive), ostats(s_aware)

    report = ["IAVAL demo results (synthetic survey, threshold %.1f sigma)" % THR, "",
              "Pixel level (6 injected overdensities, %d true pixels)" % truth.sum(),
              "  naive : %s" % pn, "  aware : %s" % pa, "",
              "Object level (%d injected anomalies in %d objects, threshold 4 sigma)" % (cat["is_anom"].sum(), len(cat["pix"])),
              "  naive : %s" % on, "  aware : %s" % oa]
    txt = "\n".join(report)
    print(txt)
    open(os.path.join(OUT, "outputs", "results.txt"), "w").write(txt + "\n")

    # --- exports for TOPCAT --------------------------------------------------
    export.pixel_table(NSIDE, counts, mu, naive, aware, s["depth"], s["seeing"], truth).to_csv(
        os.path.join(OUT, "outputs", "pixel_candidates_topcat.csv"), index=False)
    export.object_table(cat, c_naive, s_aware).to_csv(
        os.path.join(OUT, "outputs", "object_candidates_topcat.csv"), index=False)

    # --- figures -------------------------------------------------------------
    def mollview(m, title, fname, **kw):
        plt.figure(figsize=(7, 4))
        hp.mollview(m, title=title, hold=True, **kw)
        plt.savefig(os.path.join(FIG, fname), dpi=130, bbox_inches="tight"); plt.close()
    mollview(s["depth"], "Imaging depth offset (systematic)", "depth.png")
    mollview(counts, "Observed galaxy counts per pixel", "counts.png")
    mollview(naive, "Naive anomaly score", "naive_score.png", min=-4, max=6)
    mollview(aware, "Instrument-aware anomaly score", "aware_score.png", min=-4, max=6)
    mollview(truth.astype(float), "Injected true overdensities", "truth.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 12, 60)
    ax[0].hist(s_naive[~cat["is_anom"]], bins, alpha=.6, label="naive", density=True)
    ax[0].hist(s_aware[~cat["is_anom"]], bins, alpha=.6, label="aware", density=True)
    ax[0].set_yscale("log"); ax[0].set_xlabel("significance (sigma), normal objects"); ax[0].legend()
    ax[0].set_title("Normal objects: aware scores follow the null")
    for sig, lab in [(s_naive, "naive"), (s_aware, "aware")]:
        ths = np.linspace(2, 8, 30)
        ax[1].plot(ths, [((sig > t) & ~cat["is_anom"] & shallow).sum() for t in ths], label=lab)
    ax[1].set_yscale("symlog"); ax[1].set_xlabel("threshold (sigma)"); ax[1].set_ylabel("false positives, shallowest quartile")
    ax[1].legend(); ax[1].set_title("False alarms in shallow imaging")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "object_level.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
