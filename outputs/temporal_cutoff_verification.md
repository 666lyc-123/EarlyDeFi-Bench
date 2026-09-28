# Temporal Cutoff Verification

Audits the released feature table and case-study trace against the
contract `max_included_time <= prediction_time` (zero tolerance).

Boundary note: historical RPC and participant-graph windows were indexed
by estimated prediction blocks. `feature_observed_at` is the declared
cutoff recorded at collection time; this audit verifies that declared
cutoff against `prediction_time`, matching the paper's stated boundary.
It does not claim timestamp-level visibility of individual logs.

## Feature cutoff (main table)

- Samples: 1056
- Feature rows: 71808
- Feature names: 68
- Row violations (observed_at > prediction_time): 0
- Sample violations (max included time > prediction time): 0
- Min margin (seconds): 0.0
- Samples with zero margin (cutoff exactly at prediction time): 1056

## Pre-event validity

- Positive samples warned not strictly before event: 0
- Samples warned before pool creation: 0

## Case-study trace

- Skipped: True

Overall pass: **True**
