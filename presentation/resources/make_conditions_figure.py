"""Figure for the backup slide on predictive coherence: one valid update rule and
one rule violating each of Fong et al.'s two conditions.

Same model and data as martingale_posterior.html: y_i ~ N(theta, 1), theta ~ N(0, 1),
n = 10, thetaBar_n = 1.84. Every panel uses the same standard normal draws.

  (a) Valid (Bayesian):      thetaBar_{k} = thetaBar_{k-1} + (Y_k - thetaBar_{k-1}) / (k+1),
                             Y_k ~ N(thetaBar_{k-1}, 1 + 1/k)        -> converges, centred
  (b) Existence violated:    fixed learning rate a = 0.03 and fixed predictive variance
                             -> random walk, paths never settle (no P_inf)
  (c) Unbiasedness violated: Bayesian update, but Y_k ~ N(thetaBar_{k-1} + delta_k, ...)
                             with a fading shift delta_k = C_SHIFT / sqrt(k): the total expected
                             drift sum_k delta_k / (k+1) is finite, so the paths still settle
                             (Condition 1 holds), but the cloud ends away from thetaBar_n

    python3 make_conditions_figure.py   ->  conditions.svg
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#00883A"
ORANGE = "#D9822B"
FG = "#626468"
GREY = "#A9A7A0"
TEXT = "#2F2F2F"

N_OBS, K, B = 10, 1000, 150
THETA_N = 1.84
S2_N = 1 + 1 / (N_OBS + 1)
C_SHIFT = 0.7


def simulate(z, kind):
    paths = np.empty((B, K + 1))
    th = np.full(B, THETA_N)
    paths[:, 0] = th
    for j in range(K):
        k = N_OBS + j + 1                    # number of observations after this update
        if kind == "existence":
            y = th + np.sqrt(S2_N) * z[:, j]
            th = th + 0.03 * (y - th)
        else:
            s2 = 1 + 1 / k                   # posterior predictive variance before the update
            shift = C_SHIFT / np.sqrt(k) if kind == "unbiased" else 0.0
            y = th + shift + np.sqrt(s2) * z[:, j]
            th = th + (y - th) / (k + 1)
        paths[:, j + 1] = th
    return paths


def main():
    rng = np.random.default_rng(31542)
    z = rng.standard_normal((B, K))
    panels = [
        ("valid", "Valid (Bayesian update)", "$P_\\infty$ exists and $E[P_\\infty \\mid y_{1:n}] = P_n$", GREEN),
        ("existence", "Condition 1 (Existence) violated", "no limit $P_\\infty$", ORANGE),
        ("unbiased", "Condition 2 (Unbiasedness) violated", "$P_\\infty$ exists, but $E[P_\\infty \\mid y_{1:n}] \\neq P_n$", ORANGE),
    ]
    steps = np.arange(K + 1)
    x = N_OBS + steps

    plt.rcParams.update({"svg.fonttype": "path", "mathtext.fontset": "cm", "font.size": 17,
                         "font.family": "Arial"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, (kind, title, sub, colour) in zip(axes, panels):
        p = simulate(z, kind)
        for row in p:
            ax.plot(x, row, color=GREY, alpha=0.35, linewidth=0.7)
        ax.plot(x, p.mean(axis=0), color=colour, linewidth=2.6)
        ax.axhline(THETA_N, color=TEXT, linestyle=(0, (4, 4)), linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlim(N_OBS, N_OBS + K)
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1000"])
        ax.minorticks_off()
        ax.set_ylim(THETA_N - 1.6, THETA_N + 1.6)
        ax.set_title(title, color=TEXT, fontsize=20, fontweight="bold", loc="left", pad=28)
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, color=FG, fontsize=16, va="bottom")
        ax.set_xlabel("forward step $i$ (log scale)", color=FG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(FG)
        ax.tick_params(colors=FG)
    axes[0].set_yticks([THETA_N])
    axes[0].set_yticklabels([r"$\bar\theta_n$"], fontsize=22, color=TEXT)

    fig.tight_layout(w_pad=2.5)
    out = Path(__file__).with_name("conditions.svg")
    fig.savefig(out, transparent=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
