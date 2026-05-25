# Historical Bug Log

## 2026-05-25 — Ezquiaga `_d2Vdx2` `dD` derivative sign error

**File**: `models/ezquiaga_chi.py:160`

**Symptom**: Mukhanov-Sasaki solver gave P_S(k) ~ 10⁻⁴² instead of
~10⁻⁹. Ratio MS/SR was essentially zero at all k. The mode decayed
as a damped oscillator instead of growing as z = a√(2ε_H) after
horizon crossing.

**Root cause**: `_d2Vdx2()` had an extra `xs` factor in `dD`:

```python
# WRONG:
dD = self.c * xs * (2 * B + self.b * xs)
# CORRECT:
dD = self.c * xs * (2 * B + self.b)
```

**Math**: `D = 1 + c·B·x²` with `B = 1 + b·ln(x)`. The derivative is:
```
dD/dx = c·[2x·B + x²·dB/dx] = c·[2x·B + x²·b/x] = c·x·(2B + b)
```

The extra `xs` made `dD` artificially large at large x (>3 M_P),
causing `_d2Vdx2` to give +1.78 instead of −0.004 at χ=8.0 — 3
orders of magnitude off with the **wrong sign**. This propagated
through the MS effective mass `m2 = ... − v0·d²f/dχ²/S²`, making
it negative early in inflation.

**Fix**: Remove the extra `xs` factor (commit `e1957b5`).

**Verification**: After fix, `n_s = 0.952` at `k = 0.05` for
χ₀ = 8.0 — exact match to the published value in Ezquiaga et al.
(1705.04861). MS/SR ratio is flat ~1.14 (systematic normalization
offset, not shape error).

**Diagnostic**: `python -m scripts.ezquiaga_ps_check --chi0 8.0 --n-k 80`
