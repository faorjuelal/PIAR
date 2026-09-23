"""Publication style for all PIAR figures.

Rules enforced here (project requirement + dataviz checks):
  * Computer Modern text through LaTeX (text.usetex).
  * No top/right spines, thin marks, hairline axes, no gridlines by default.
  * Continuous maps: Crameri batlow (sequential, magnitude) and vik (diverging).
  * Categorical colours in a FIXED order, validated for colour-vision deficiency
    (adjacent CVD Delta E >= 8 on a white surface); low-contrast slots are always
    paired with a legend or a marker (secondary encoding).
  * A colour follows the entity (method/integrator), never its rank.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

# AIP maximum figure widths in inches (single column 8.5 cm, double column 17 cm)
COL1 = 3.37
COL2 = 6.69
# AIP minimum type size at the final printed width (8 pt); every text element is set at >= 8 pt
FS_MIN = 8.0

# Validated categorical order (dataviz validator, surface #ffffff): all checks pass.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK = "#222222"        # ground truth / analytic curves (neutral, not a series slot)
MUTED = "#8a8a8a"      # separatrices, references, recessive guides

# Entity -> colour (fixed everywhere in the paper)
C = {
    "truth": INK,
    "nsf_phys": CAT[0], "verlet": CAT[0],
    "gpr": CAT[1], "dop853": CAT[1],
    "nsf": CAT[2], "rk4": CAT[2],
    "diffusion": CAT[3],
    "cvae": CAT[4],
    "static": CAT[5], "implicit_midpoint": CAT[5],
}
MARK = {"nsf_phys": "o", "gpr": "s", "nsf": "^", "diffusion": "D", "cvae": "v",
        "static": "x", "verlet": "o", "dop853": "s"}
LABEL = {"nsf_phys": r"NSF + physics", "gpr": r"original FPRM (GPR)", "nsf": r"NSF",
         "diffusion": r"Diffusion", "cvae": r"cVAE", "static": r"Static ($E=1$)"}

SEQ = cmc.batlow           # sequential (magnitude)
SEQ_W = cmc.batlowW_r      # white -> dark: densities on a white page
DIV = cmc.vik              # diverging (signed quantities)


def set_style(usetex: bool = True) -> None:
    mpl.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": INK,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "image.cmap": "cmc.batlow",
        "pdf.fonttype": 42,
    })


def panel_label(ax, letter: str, x: float = 0.0, y: float = 1.03) -> None:
    ax.text(x, y, rf"({letter})", transform=ax.transAxes, fontsize=9.0,
            va="bottom", ha="left", color=INK)


def savefig(fig, stem: str, outdir: str = "figures") -> None:
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{stem}.pdf")
    fig.savefig(f"{outdir}/{stem}.png", dpi=600)
    plt.close(fig)
