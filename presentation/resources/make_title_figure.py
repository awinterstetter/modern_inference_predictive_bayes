"""Title-slide figure: martingale posterior via predictive resampling.

Re-implements the simulation from martingale_posterior.html (same RNG,
seeds and update rule, so the paths are identical) and draws a stripped-down
version in LMU colours as an SVG with a transparent background.

    python3 make_title_figure.py   ->  title-figure.svg        (for white backgrounds)
                                       title-figure-white.svg  (for the green title slide)
"""
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- RNG (JS port)
M32 = 0xFFFFFFFF


def _imul(a, b):
    r = (a & M32) * (b & M32) & M32
    return r - (1 << 32) if r & 0x80000000 else r


def _i32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def mulberry32(a):
    state = [_i32(a)]

    def rnd():
        a = _i32(state[0] + 0x6D2B79F5)
        state[0] = a
        t = _imul(a ^ ((a & M32) >> 15), 1 | a)
        t = _i32(t + _imul(t ^ ((t & M32) >> 7), 61 | t)) ^ t
        return ((t ^ ((t & M32) >> 14)) & M32) / 4294967296

    return rnd


class Sim:
    n, K = 10, 1000
    N = n + K

    def __init__(self):
        self.rng = mulberry32(7)
        y = [2 + self.randn() for _ in range(self.n)]
        shift = (1.84 * (self.n + 1) - sum(y)) / self.n
        y = [v + shift for v in y]
        self.S0 = sum(y)
        self.theta_hat = self.S0 / (self.n + 1)
        self.sd_post = 1 / math.sqrt(self.n + 1)
        self.sd_theory = math.sqrt(1 / (self.n + 1) - 1 / (self.N + 1))

    def randn(self):
        u = 0.0
        while u == 0.0:
            u = self.rng()
        v = self.rng()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)

    def path(self):
        vals = np.empty(self.K + 1, dtype=np.float32)
        S, th = self.S0, self.theta_hat
        vals[0] = th
        for i in range(self.n, self.N):
            s2 = 1 / (i + 1)
            yn = th + math.sqrt(1 + s2) * self.randn()
            S += yn
            th = S / (i + 2)
            vals[i - self.n + 1] = th
        return vals

    def separated_path(self, prev):
        n, N, K = self.n, self.N, self.K
        k0 = round(n * ((N / n) ** 0.25 - 1))
        best, best_score = None, -math.inf
        for _ in range(400):
            v = self.path()
            if abs(v[K] - self.theta_hat) > 1.8 * self.sd_theory:
                continue
            gap = end_gap = math.inf
            for q in prev:
                end_gap = min(end_gap, abs(v[K] - q[K]))
                gap = min(gap, float(np.min(np.abs(v[k0:K + 1:5] - q[k0:K + 1:5]))))
            score = min(gap / 0.45, end_gap / 0.9) / self.sd_theory
            if score >= 1:
                return v
            if score > best_score:
                best, best_score = v, score
        return best if best is not None else self.path()


def simulate():
    sim = Sim()
    seed = 101
    hl = []
    for stage, s in enumerate([31542, 39769, None]):
        sim.rng = mulberry32(seed + stage * 7919)
        if s is not None:
            sim.rng = mulberry32(s)
            hl.append(sim.path())
        else:
            hl.append(sim.separated_path(hl))
    grey = []
    for i in range(997):
        sim.rng = mulberry32(seed * 31 + i * 104729)
        grey.append(sim.path())
    return sim, hl, np.array(grey)


# ---------------------------------------------------------------- drawing
# two colour schemes: "light" for a white background, "dark" (all white
# strokes) for drawing directly on the LMU-green title slide
PALETTES = {
    "light": dict(grey="#A9A7A0", grey_a=0.22, hl="#00883A", fg="#626468",
                  axis="#9C9DA1", fill="#00883A", fill_a=0.16),
    "dark":  dict(grey="#FFFFFF", grey_a=0.10, hl="#FFFFFF", fg="#FFFFFF",
                  axis="#FFFFFF", fill="#FFFFFF", fill_a=0.18),
}


def draw(out, scheme="light"):
    c = PALETTES[scheme]
    sim, hl, grey = simulate()
    K, th0, sd = sim.K, sim.theta_hat, sim.sd_post
    ylo, yhi = th0 - 4.2 * sd, th0 + 4.2 * sd

    # dense early, sparse late: keeps the SVG small without visible change
    ks = np.unique(np.concatenate([np.arange(0, 60), np.geomspace(60, K, 160).astype(int), [K]]))

    plt.rcParams.update({
        "svg.fonttype": "path",          # text as outlines: looks identical everywhere
        "mathtext.fontset": "cm",
        "font.size": 20,
    })
    fig = plt.figure(figsize=(8.3, 7.2))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.09, 0.08, 0.60, 0.84])
    axd = fig.add_axes([0.75, 0.08, 0.23, 0.84], sharey=ax)
    for a in (ax, axd):
        a.patch.set_alpha(0)
        a.set_ylim(ylo, yhi)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_xticks([])
        a.set_yticks([])

    # background paths
    for v in grey:
        ax.plot(ks, v[ks], color=c["grey"], lw=0.6, alpha=c["grey_a"], solid_capstyle="round",
                rasterized=True)   # one embedded 300-dpi image -> small SVG
    # highlighted paths + end markers
    ends = []
    for v in hl:
        ax.plot(ks, v[ks], color=c["hl"], lw=2.4, solid_capstyle="round", zorder=5)
        ax.plot([K], [v[K]], "o", ms=7, color=c["hl"], zorder=6)
        ends.append(float(v[K]))

    # axes: thin y-axis and an x-arrow
    ax.set_xlim(-8, K * 1.06)
    # both axes as ONE connected line, so the corner is closed
    ax.plot([0, 0, K * 1.05], [yhi, ylo, ylo], color=c["axis"], lw=1.4,
            solid_joinstyle="miter", solid_capstyle="butt", clip_on=False)
    # arrow heads only (no shrink, so they sit exactly on the line ends)
    head = dict(arrowstyle="-|>", color=c["axis"], lw=1.4, mutation_scale=16,
                shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(K * 1.05, ylo), xytext=(K * 1.0, ylo), arrowprops=head)
    ax.annotate("", xy=(0, yhi), xytext=(0, yhi - 0.05 * (yhi - ylo)), arrowprops=head,
                annotation_clip=False)
    # start value: a short tick across the y-axis (same style as the pi_N ticks)
    ax.plot([-K * 0.012, K * 0.012], [th0, th0], color=c["fg"], lw=2.4,
            solid_capstyle="butt", clip_on=False, zorder=7)
    ax.text(-30, th0, r"$\bar\theta_n$", ha="right", va="center", color=c["fg"], fontsize=32)
    # y-axis label at its top (like pi_N)
    ax.text(K * 0.035, yhi, r"$\bar\theta_i$", ha="left", va="top", color=c["fg"], fontsize=32)
    # name of the x-axis variable at the arrow tip: theta_bar_i is indexed by i
    ax.text(K * 1.075, ylo, r"$i$", ha="left", va="center", color=c["fg"], fontsize=30, clip_on=False)

    # density of the limits theta_bar_N (KDE of all 1000 end points)
    end_all = np.concatenate([grey[:, K], np.array(ends)])
    bw = 1.06 * end_all.std() * len(end_all) ** (-1 / 5)
    yy = np.linspace(ylo, yhi, 400)
    dens = np.exp(-0.5 * ((yy[:, None] - end_all[None, :]) / bw) ** 2).sum(1)
    dens /= dens.max()
    axd.set_xlim(0, 1.08)
    axd.fill_betweenx(yy, 0, dens, color=c["fill"], alpha=c["fill_a"], lw=0)
    axd.plot(dens, yy, color=c["hl"], lw=2.2)
    axd.plot([0, 0], [ylo, yhi], color=c["axis"], lw=1.4)
    axd.text(0.06, yhi, r"$\pi_N$", ha="left", va="top", color=c["fg"], fontsize=32)
    for e in ends:
        axd.plot([-0.05, 0.05], [e, e], color=c["hl"], lw=2.4, clip_on=False, zorder=4)

    # dotted connectors from the path ends to the density axis
    for e in ends:
        con = matplotlib.patches.ConnectionPatch(
            xyA=(K, e), coordsA=ax.transData, xyB=(0, e), coordsB=axd.transData,
            color=c["hl"], lw=1.2, ls=(0, (1, 3)), alpha=0.8)
        fig.add_artist(con)

    fig.savefig(out, transparent=True, dpi=300)
    plt.close(fig)
    print("wrote", out, f"({Path(out).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    here = Path(__file__).parent
    draw(here / "title-figure.svg", "light")         # for white backgrounds
    draw(here / "title-figure-white.svg", "dark")    # for the green title slide
