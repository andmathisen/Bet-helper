# RL Betting Agent - Implementation Status

## ✅ Completed

### Phase 1: Foundation
- [x] Created `bet_helper/rl/` directory structure
- [x] Created `PLAN.md` with comprehensive architecture
- [x] Implemented `data.py` - Gameweek loading and data utilities
  - `load_gameweeks()` - Groups matches by gameweek
  - `load_historical_data()` - Loads historical outcomes for simulation
  - `MatchWithOdds` and `GameweekData` dataclasses
  - Outcome mapping from historical matches

- [x] Implemented `env.py` - BettingEnv Gymnasium environment
  - State space: ML features + tokens + gameweek context
  - Action space: Dict space for bet type, matches, outcomes, bet amount
  - Reward function: Normalized profit/loss
  - Single bet and accumulator bet processing
  - Feature extraction integration with existing ML pipeline
  - Outcome simulation using historical data or odds-based sampling

- [x] Updated `pyproject.toml` with dependencies:
  - `stable-baselines3>=2.0.0`
  - `gymnasium>=0.28.0`

## ✅ Completed (All Phases)

### Phase 2: Training Setup
- [x] Create `train.py` - Training script with PPO agent
- [x] Add hyperparameter configuration (learning_rate, n_steps, batch_size, etc.)
- [x] Implement checkpointing and logging
- [x] Add tensorboard integration for monitoring
- [x] Train/eval split with validation environment
- [x] Early stopping and best model saving

### Phase 3: Evaluation
- [x] Create `eval.py` - Evaluation metrics and visualization
  - [x] Profit/loss tracking
  - [x] Win rate and ROI calculations
  - [x] Per-episode statistics
  - [x] Aggregate performance metrics
  - [x] JSON export of results

### Phase 4: CLI Integration
- [x] Add `bet_helper.cli` commands:
  - [x] `bet-helper rl train --leagues PL --total-timesteps 100000`
  - [x] `bet-helper rl eval --model-path path/to/model --leagues PL`
  - [x] `bet-helper rl play --model-path path/to/model --leagues PL` (live simulation)

### Phase 5: Testing & Refinement
- [ ] Test environment with real data
- [ ] Tune reward function
- [ ] Optimize action space (may need simplification)
- [ ] Add unit tests

## 📦 Installation Required

Before running, install new dependencies:

```bash
# Use pip directly to avoid Poetry build issues with llvmlite/numba
source venv/bin/activate
pip install stable-baselines3 gymnasium --no-build-isolation
```

See `INSTALL.md` for details.

## 🚀 Quick Start (After Installation)

1. **Train agent:**
```bash
# Train on specific leagues
python3 -m bet_helper.cli rl train --leagues PL SerieA --total-timesteps 100000

# Train on all leagues with custom parameters
python3 -m bet_helper.cli rl train --all --total-timesteps 500000 --initial-tokens 2000 --learning-rate 1e-4
```

2. **Evaluate trained agent:**
```bash
# Evaluate on test data
python3 -m bet_helper.cli rl eval --model-path data/rl_models/betting_agent_20260120_120000 --leagues PL --n-episodes 50

# Auto-saves results to data/rl_eval_results_TIMESTAMP.json
```

3. **Run live simulation:**
```bash
python3 -m bet_helper.cli rl play --model-path data/rl_models/betting_agent_20260120_120000 --leagues PL
```

## 📝 Notes

- **Action Space Complexity**: The current Dict action space may be challenging for PPO. Consider:
  - Simplifying to discrete actions (pre-defined bet types)
  - Using a hierarchical action space
  - Training separate models for singles vs accumulators

- **Feature Extraction**: Currently extracts features from first match in gameweek. Future enhancement:
  - Aggregate features across all matches
  - Include gameweek-level statistics

- **Outcome Simulation**: Currently uses:
  1. Historical outcomes if available
  2. Odds-based probability sampling otherwise
  - Consider using ML model predictions for more realistic simulation

## 🔧 Known Issues / Future Improvements

1. **Action space may be too complex** - Monitor training convergence
2. **Feature dimension** - Needs to be calculated at runtime (currently using sample)
3. **Gameweek grouping** - Currently groups by ISO week; may need league-specific logic
4. **Memory efficiency** - Loading all gameweeks upfront; consider lazy loading for large datasets
