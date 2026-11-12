"""Figure for the "Characterizing Uncertainty" slide: a posterior for theta.

An arbitrary, slightly skewed density (mixture of two normals), in the
same LMU colours as the title figure.

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
    x = np.linspace(-3.2, 5.2, 1200)
    dens = 0.7 * normal_pdf(x, 0.8, 0.75) + 0.3 * normal_pdf(x, 2.3, 1.0)


    plt.rcParams.update({
        "svg.fonttype": "path",
        "mathtext.fontset": "cm",
        "font.size": 22,
    })
    fig, ax = plt.subplots(figsize=(8, 4.2))

    ax.fill_between(x, dens, color=FG, alpha=0.16, linewidth=0)
    ax.plot(x, dens, color=FG, linewidth=3)

    ax.text(3.0, dens.max() * 0.75, r"$\pi(\theta \mid y_{1:n})$", color=TEXT, fontsize=26)

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
