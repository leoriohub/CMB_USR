"""Test suite configuration for CMB Anomaly project."""
# NOTE: CAMB NumPy 2.x compat patch lives in scripts/camb_wrapper.py (lines 29-36).
# pytest picks up camb_wrapper.py's import-time patch automatically.
# DO NOT duplicate it here — double-patching causes TypeError in pytest mode.
