"""Figure for the "Characterizing Uncertainty" slide: a posterior for theta.

A standard normal density with a double arrow marking its spread
(+/- 2 sd), in the same LMU colours as the title figure.

    python3 make_posterior_figure.py   ->  posterior.svg
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#00883A"
FG = "#626468"
AXIS = "#9C9DA1"
TEXT = "#2F2F2F"   # slide body-text colour (near black)


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def main():
    x = np.linspace(-4.2, 4.2, 1200)
    dens = normal_pdf(x, 0.0, 1.0)


    plt.rcParams.update({
        "svg.fonttype": "path",
        "mathtext.fontset": "cm",
        "font.size": 22,
    })
    fig, ax = plt.subplots(figsize=(8, 4.2))

    ax.fill_between(x, dens, color=FG, alpha=0.16, linewidth=0)
    ax.plot(x, dens, color=FG, linewidth=3)

    # double arrow inside the density, close to the x-axis, not touching the curve
    y_arr = normal_pdf(2.4, 0.0, 1.0)
    ax.annotate("", xy=(-2.0, y_arr), xytext=(2.0, y_arr),
                arrowprops=dict(arrowstyle="<->", color=TEXT, lw=2.2,
                                shrinkA=0, shrinkB=0, mutation_scale=22))

    ax.text(1.6, dens.max() * 0.75, r"$\pi(\theta \mid y_{1:n})$", color=TEXT, fontsize=26)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, dens.max() * 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(x[-1], -dens.max() * 0.03, r"$\theta$", ha="right", va="top", color=TEXT, fontsize=26)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.5)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "posterior.svg"
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out)


if __name__ == "__main__":
    main()
