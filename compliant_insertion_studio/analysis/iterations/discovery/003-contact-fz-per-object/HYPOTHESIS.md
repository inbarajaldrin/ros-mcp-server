# Discovery 003: Contact-Fz distribution per object (GOLD vs FAIL)

## Question (from STATE.json open_questions[2])

> What's the right approach-velocity for first contact? Operators land at ~6.0N median; algorithm at ~7.2N — 20% harder.

## Sub-questions

1. Is the GOLD contact-Fz target object-dependent, or universal at ~6.0 N?
2. Does I009 (FAIL Fz_at_contact > GOLD Fz_at_contact) generalize beyond u_orange? (We only have u_orange FAIL data, so the cross-object check is GOLD-only — but we can ask: do all 4 objects' GOLDs cluster similarly tight?)
3. Is `cmd_fz` (the commanded value) related to the actual sensed Fz at contact in a clean way? If `commanded_fz` is constant across attempts but sensed Fz_at_contact varies by object, the variation is geometry/approach-velocity-coupled, not config-coupled.

## Data

- 60 GOLD demos × 4 objects (May-3)
- 132 May-4 u_orange autonomous attempts (47 success / 77 abort / 7 timeout)

## Method

For each per_sample.json:
- read `fz_t` and `t_s` and `commanded_fz` arrays
- contact_idx = `contact_idx_active` from summaries.json (index into per_sample arrays)
- Fz_at_contact = `fz_t[contact_idx]` (sensed, tool frame, sign convention used elsewhere in pipeline)
- Fz_at_contact_smoothed = mean of `fz_t[contact_idx-5 : contact_idx+5]` (10-sample / 100ms window)
- cmd_fz_at_contact = `commanded_fz[contact_idx]` if available

Aggregate per (object, outcome_class) where outcome_class ∈ {GOLD, AUTO_success, AUTO_abort, AUTO_timeout}.

GOLD includes any May-3 demo regardless of `outcome` field (those were operator-assisted demonstrations; they all "succeeded" under operator hand-guidance, which is the gold standard we want the algorithm to imitate).

AUTO is May-4 u_orange; split by outcome flag, but per trust_hierarchy item 3, the FSM outcome flag is NOT trusted as a label — re-derive label by `final_z_drop_mm ≥ 20mm` (success) vs `< 20mm` (fail).

## What "answers" the question

A clean per-object statistic (median, IQR, n) for Fz_at_contact in GOLD across 4 objects. If the distribution is tight and object-similar (median 5.5–6.5 N for all 4 objects, IQR < 2 N), then **6.0 N is a universal target** and the right approach-velocity is whatever produces 6.0 N at first contact. If it varies by object (e.g. line_green is 4 N, u_brown is 8 N), then `cmd_fz` should be per-object.

For I009 portability: we only have u_orange FAIL data. So the BEST we can do is verify GOLD u_orange (~6 N expected) sits below the GOLD bands of the other 3 objects (or with them). If u_orange GOLD is uniquely soft, the I009 finding (FAIL u_orange lands at 7.2 N) gains a "soft contact is easier here than other parts" interpretation.

## Pre-registered predictions

- P1: GOLD median Fz_at_contact is 5–7 N for all 4 objects.
- P2: GOLD IQR < 2 N within each object.
- P3: AUTO_fail u_orange median Fz_at_contact > 7 N (matches I009).
- P4: AUTO_success u_orange median Fz_at_contact ≈ GOLD u_orange median (the successes that DID happen autonomously had soft contacts).

## Predicted invariant if all 4 confirm

> "GOLD-class first-contact Fz is 5–7 N, object-independent. AUTO failures land at >7 N. Suggests reducing approach velocity / softening cmd_fz ramp until first-contact Fz lands in [5, 7] N is portable across objects."

## Predicted refutation

If P1 is wrong (large object-spread in GOLD), then approach-velocity is intrinsically coupled to object geometry; need per-object cmd_fz.
