# Replay simulation — counterfactual on 132 May-4 FAIL traces

**Object:** u_orange
**Episodes replayed:** 69
**Hole xy used:** (34.1, -363.5) mm

## Headline
- 47/69 = **68.1%** of failures predicted to land in GOLD displacement band [1.2, 1.7] mm @ 1 s post-contact
- 69/69 = **100.0%** of failures predicted to end CLOSER to the hole vs actual

## Predicted vs actual displacement at t = 1 s post-contact (mm)
| | p25 | p50 | p75 | GOLD median |
|---|---|---|---|---|
| actual    | 0.13 | 0.20 | 0.57 | 1.2-1.7 |
| predicted | 1.49 | 1.60 | 1.82 | 1.2-1.7 |

## Caveats
- Linear admittance approximation; ignores rim-contact nonlinearity
- Single-step prediction (no Fz collapse cascade modeled)
- Treats cmd_fx/cmd_fy actual as 0 (conservative — May-4 FSM did command non-trivial lateral, so true delta is smaller than computed)
- Use as a **wrong-move filter**, NOT a success predictor