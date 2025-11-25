import os
from pathlib import Path

from ..utils.myutils import load_config

def _load_config(filename):
    return load_config(os.path.join(Path(__file__).parent.parent, "config", filename))

CFG_DRDF = _load_config("drdf.yaml")
CFG_AMORT = _load_config("amortizer.yaml")
CFG_FITTING = _load_config("fitting.yaml")

SYMBOLS = {
    "a": f"$a$",
    "mu": r"$\mu$",
    "T_er": r"$T_{\mathrm{er}}$",
    "eta": r"$\eta$",
    "s_z": r"$s_z$",
    "s_t": r"$s_t$",
    "sigma": r"$\sigma$",
    "step_size": r"$\tau$",
    "a_z_ratio": r"$a/z$"
}