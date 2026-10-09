"""Expected-abundant lipid species from fatty-acid composition.

The dominant species in each class are predictable: they're built from the most
common fatty acids (16:0, 18:0, 18:1, 18:2, 20:4, 22:6 …). Enumerating the likely
FA combinations for a class and summing to total carbon:double-bond gives the
species you'd expect to be most abundant - the most trustworthy annotations, and
therefore the best RT-grid anchors.

This is an approximate biological prior (mammalian-lipidomics defaults); the FA
pools and weights are easy to tune.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

# Common glycerolipid acyl chains: (carbons, double_bonds, relative_weight).
ACYL = [
    (16, 0, 1.00),   # palmitic
    (18, 1, 1.00),   # oleic
    (18, 2, 0.90),   # linoleic
    (18, 0, 0.70),   # stearic
    (20, 4, 0.70),   # arachidonic
    (22, 6, 0.55),   # DHA
    (16, 1, 0.40),   # palmitoleic
    (18, 3, 0.30),   # α/γ-linolenic
    (20, 3, 0.30),
    (22, 5, 0.25),
    (22, 4, 0.20),
    (14, 0, 0.20),
    (20, 1, 0.15),
    (20, 0, 0.12),
]

# Sphingolipid N-acyl chains (long, often saturated / monounsaturated).
SPH_NACYL = [
    (16, 0, 1.00),
    (24, 1, 0.90),
    (24, 0, 0.80),
    (22, 0, 0.60),
    (18, 0, 0.50),
    (23, 0, 0.40),
    (20, 0, 0.30),
    (26, 1, 0.25),
    (26, 0, 0.20),
    (22, 1, 0.20),
]

# Dominant sphingoid base by subclass marker (carbons, double_bonds).
SPH_BASE = {"d": (18, 1), "t": (18, 0), "m": (18, 1)}

# class -> (number of variable acyl chains, kind). kind: 'g' glycero, 's' sphingo.
RECIPE = {
    # diacyl glycerophospholipids + diacylglycerols
    "PC": (2, "g"), "PE": (2, "g"), "PS": (2, "g"), "PG": (2, "g"),
    "PI": (2, "g"), "PA": (2, "g"), "BMP": (2, "g"), "DG": (2, "g"),
    "SLBPA": (2, "g"), "MGDG": (2, "g"), "DGDG": (2, "g"), "CDP-DG": (2, "g"),
    "PGP": (2, "g"), "DLCL": (2, "g"),
    # lyso / single-chain
    "LPC": (1, "g"), "LPE": (1, "g"), "LPS": (1, "g"), "LPG": (1, "g"),
    "LPI": (1, "g"), "LPA": (1, "g"), "MG": (1, "g"), "CE": (1, "g"),
    "AC": (1, "g"), "FA": (1, "g"), "FA-OH": (1, "g"), "AC-OH": (1, "g"),
    # tri / tetra
    "TG": (3, "g"), "MLCL": (3, "g"), "CL": (4, "g"),
    # sphingolipids (sphingoid base + 1 N-acyl)
    "SM": (1, "s"), "Cer": (1, "s"), "HexCer": (1, "s"), "Hex2Cer": (1, "s"),
    "Hex3Cer": (1, "s"), "SM4": (1, "s"), "C1P": (1, "s"),
    "GM1": (1, "s"), "GM2": (1, "s"), "GM3": (1, "s"),
    "GD1": (1, "s"), "GD2": (1, "s"), "GD3": (1, "s"), "GT1": (1, "s"),
    "GA1": (1, "s"), "GA2": (1, "s"),
    "SPH": (0, "s"), "S1P": (0, "s"),
}

POSITIONAL_CLASS_ALIASES = {
    "1-acylLPC": "LPC",
    "1-acylLPE": "LPE",
}
GRID_CLASS_MARKERS = {"O", "P", "d", "t", "m"}


def _recipe_identity(lipid_class: str, base_marker=None):
    """Accept both raw classes and MATLAB/grid-class spellings."""
    class_name = POSITIONAL_CLASS_ALIASES.get(
        str(lipid_class), str(lipid_class)
    )
    marker = None if base_marker in (None, "") else str(base_marker)
    if marker is None and "-" in class_name:
        head, suffix = class_name.rsplit("-", 1)
        if suffix in GRID_CLASS_MARKERS:
            class_name, marker = head, suffix
    return class_name, marker


def expected_compositions(lipid_class: str, base_marker=None, top_k: int = 15):
    """Return the expected-abundant total compositions for a class.

    ``lipid_class`` is the raw class (e.g. 'PC', 'SM' - not the grid_class), and
    ``base_marker`` is the sphingoid/ether marker ('d','t','O','P' or None).
    Returns a dict ``{(carbon, db): weight}`` of the top_k compositions, or {} for
    classes we don't have a recipe for.
    """
    lipid_class, base_marker = _recipe_identity(lipid_class, base_marker)
    rec = RECIPE.get(lipid_class)
    if rec is None:
        return {}
    n_acyl, kind = rec
    if kind == "s":
        pool = SPH_NACYL
        base_c, base_d = SPH_BASE.get(base_marker, (18, 1))
    else:
        pool, base_c, base_d = ACYL, 0, 0

    weights: dict[tuple[int, int], float] = defaultdict(float)
    if n_acyl == 0:
        weights[(base_c, base_d)] = 1.0
    else:
        for combo in itertools.combinations_with_replacement(pool, n_acyl):
            c = base_c + sum(f[0] for f in combo)
            d = base_d + sum(f[1] for f in combo)
            w = 1.0
            for f in combo:
                w *= f[2]
            weights[(c, d)] += w

    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return dict(top)
