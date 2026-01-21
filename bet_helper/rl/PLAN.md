# Reinforcement Learning Betting Agent - Implementation Plan

## Overview
Train an RL agent to maximize token balance by placing bets on simulated football gameweeks. The agent can place:
- Single match bets (1 match, 1 outcome)
- Accumulator bets (2+ matches, all must win)

## Architecture

### Components

1. **Data Layer** (`data.py`)
   - Load and group matches into gameweeks by date
   - Extract odds from upcoming/predictions files
   - Map matches to historical outcomes for simulation
   - Provide feature vectors using existing ML pipeline

2. **Environment** (`env.py`)
   - Gymnasium-compatible `BettingEnv`
   - State: ML features + current tokens + gameweek context
   - Actions: Bet type (none/single/accumulator) + match selections + outcomes + bet amount
   - Rewards: Token profit/loss normalized by initial balance
   - Simulation: Use historical match outcomes when available

3. **Agent** (`agent.py`)
   - Stable-Baselines3 PPO agent wrapper
   - Custom policy if needed for complex action space
   - Training callbacks for monitoring

4. **Training** (`train.py`)
   - Training loop with hyperparameter tuning
   - Evaluation on held-out gameweeks
   - Model checkpointing and logging

5. **Evaluation** (`eval.py`)
   - Profit/loss tracking
   - Win rate and ROI metrics
   - Betting strategy analysis
   - Visualization of performance

## Implementation Steps

### Phase 1: Foundation (Tasks RL-1 to RL-3)
- [x] Create directory structure
- [ ] Implement data loading and gameweek grouping
- [ ] Implement basic BettingEnv with simple action space

### Phase 2: Integration (Tasks RL-4 to RL-5)
- [ ] Connect feature extraction to ML pipeline
- [ ] Implement outcome simulation from historical data
- [ ] Test environment with real data

### Phase 3: Training (Task RL-6)
- [ ] Set up PPO agent with proper action/observation spaces
- [ ] Create training script with hyperparameters
- [ ] Add checkpointing and logging

### Phase 4: Evaluation (Task RL-7)
- [ ] Implement evaluation metrics
- [ ] Create visualization tools
- [ ] Compare RL agent vs baseline strategies

### Phase 5: CLI Integration (Task RL-8)
- [ ] Add `bet_helper.cli rl train` command
- [ ] Add `bet_helper.cli rl eval` command
- [ ] Add `bet_helper.cli rl play` for live simulation

## Action Space Design

### Simplified Version (v1)
- **Action Type**: Discrete(3) - No bet, Single, Accumulator
- **Match Selection**: MultiDiscrete([max_matches]) - Select up to 5 matches
- **Outcomes**: MultiDiscrete([3] * 5) - H/D/A for each match
- **Bet Amount**: Box([0, 1]) - Fraction of current tokens

### Future Enhancement (v2)
- Hierarchical action space
- Separate models for singles vs accumulators
- Dynamic accumulator size

## Reward Design

- **Win**: `(bet_amount * odds - bet_amount) / initial_tokens` (normalized profit)
- **Loss**: `-bet_amount / initial_tokens` (normalized loss)
- **No bet**: `-0.01` (small penalty to encourage action)
- **Invalid action**: `-0.1` (penalty for invalid bets)
- **Bankruptcy**: Episode terminates with `truncated=True`

## Data Requirements

1. **Training Data**: Historical gameweeks with:
   - Match odds (from predictions/upcoming files)
   - Actual outcomes (from historical matches)
   - Match features (from ML model)

2. **Evaluation Data**: Held-out recent gameweeks

3. **Simulation Data**: Full historical match database for outcome lookup

## Dependencies

```python
# Add to pyproject.toml
stable-baselines3>=2.0.0
gymnasium>=0.28.0
```

## File Structure

```
bet_helper/rl/
├── __init__.py
├── PLAN.md (this file)
├── data.py          # Data loading and gameweek grouping
├── env.py           # BettingEnv implementation
├── agent.py         # Agent wrapper and utilities
├── train.py         # Training script
├── eval.py          # Evaluation and metrics
└── utils.py         # Helper functions
```

## Success Metrics

- **ROI**: Positive return on investment over test set
- **Win Rate**: >50% on single bets
- **Risk Management**: Avoids bankruptcy, manages bet sizing
- **Strategy Diversity**: Uses both singles and accumulators appropriately
