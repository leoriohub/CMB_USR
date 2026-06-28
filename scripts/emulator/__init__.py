"""NN emulator for P_S(k) prediction from Ezquiaga CHI parameters."""

try:
    from .model import PSEmulator, train_emulator
except ImportError:
    PSEmulator = None
    train_emulator = None

from .postprocess import extract_usr_info

__all__ = ["PSEmulator", "train_emulator", "extract_usr_info"]
