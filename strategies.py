# strategies.py
from __future__ import annotations
from typing import Dict


def default_params(mode: str) -> Dict:
    """
    entry_score: 0..1 score threshold
    max_hold: max holding days
    atr_stop / atr_target: ATR multipliers
    """
    mode = (mode or "").upper()
    if mode == "MOMO":
        return {"entry_score": 0.72, "max_hold": 20, "atr_stop": 2.2, "atr_target": 5.0}
    if mode == "SQUEEZE":
        return {"entry_score": 0.70, "max_hold": 25, "atr_stop": 2.4, "atr_target": 6.0}
    # QUALITY
    return {"entry_score": 0.68, "max_hold": 60, "atr_stop": 2.0, "atr_target": 3.5}
