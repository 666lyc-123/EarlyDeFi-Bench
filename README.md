# EarlyDeFi-Bench

Minimal reproducibility package for the benchmark code requested by the
reviewers. It contains the benchmark construction logic, entity-isolated
temporal splitting, feature construction, supervised baselines, evaluation
metrics, tests, processed benchmark tables, and the paper-facing result files.

The package does not redistribute raw Ethereum RPC downloads. It provides the
processed tables and collection caches needed for the main benchmark rebuild.
Upstream data remain subject to their original terms.

## Contents

```text
src/             benchmark, split, feature, model, and metric code
scripts/         construction, feature rebuild, training, evaluation, and checks
data/processed/  processed benchmark and feature tables
data/caches/     per-family feature caches used to rebuild the main table
outputs/         core main, random-split, opcode, calibration, and scale results
tests/           unit tests for splits, leakage, features, metrics, and calibration
```

## Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Rebuild the main feature table

```powershell
$env:PYTHONPATH = "src"
.\\.venv\\Scripts\\python.exe scripts\\rebuild_main_feature_table.py
```

The released table is rebuilt from the released candidate table and the
metadata, RPC, and participant-graph caches. The script checks row counts,
feature values, observed-time fields, and the declared prediction-time cutoff.

## Run the baseline evaluation

```powershell
$env:PYTHONPATH = "src"
.\\.venv\\Scripts\\python.exe scripts\\run_baselines_for_dataset.py `
  --samples data\\processed\\pilot_samples_200_200_full_matched_stratified_temporal.parquet `
  --features data\\processed\\pilot_features_200_200_full_matched_rpc_events_participant_graph.parquet `
  --drop-features pool_creation_block,token_creation_block,rpc_pool_events_cache_complete,rpc_pool_events_rpc_error,graph_pool_participant_cache_complete,graph_pool_participant_rpc_error,graph_pool_participant_event_count_last_1h,graph_pool_participant_event_count_last_6h,graph_pool_participant_event_count_last_24h,graph_pool_participant_unique_actor_count_last_1h,graph_pool_participant_unique_actor_count_last_6h,graph_pool_participant_unique_actor_count_last_24h,graph_pool_participant_swap_sender_count_last_1h,graph_pool_participant_swap_sender_count_last_6h,graph_pool_participant_swap_sender_count_last_24h,graph_pool_participant_swap_receiver_count_last_1h,graph_pool_participant_swap_receiver_count_last_6h,graph_pool_participant_swap_receiver_count_last_24h,graph_pool_participant_burn_sender_count_last_1h,graph_pool_participant_burn_sender_count_last_6h,graph_pool_participant_burn_sender_count_last_24h,graph_pool_participant_burn_receiver_count_last_1h,graph_pool_participant_burn_receiver_count_last_6h,graph_pool_participant_burn_receiver_count_last_24h,graph_pool_participant_mint_sender_count_last_1h,graph_pool_participant_mint_sender_count_last_6h,graph_pool_participant_mint_sender_count_last_24h,graph_pool_participant_mint_event_count_last_1h,graph_pool_participant_mint_event_count_last_6h,graph_pool_participant_mint_event_count_last_24h,graph_pool_participant_burn_event_count_last_1h,graph_pool_participant_burn_event_count_last_6h,graph_pool_participant_burn_event_count_last_24h,graph_pool_participant_swap_event_count_last_1h,graph_pool_participant_swap_event_count_last_6h,graph_pool_participant_swap_event_count_last_24h `
  --output-dir reproduced\\defiguard_nc
```

## Checks

```powershell
$env:PYTHONPATH = "src"
.\\.venv\\Scripts\\python.exe scripts\\verify_temporal_cutoff.py
.\\.venv\\Scripts\\python.exe -m pytest -q
```

The external RugPull1K scope diagnostic mentioned in the paper is not part of
this minimal package because its source dataset and collection code are not
included here. The package covers the benchmark construction and evaluation
components requested by the reviewers.

## License and data rights

Project code is released under the MIT License. See `DATA_AND_RIGHTS.md` for
the treatment of upstream blockchain data, derived tables, and source terms.
