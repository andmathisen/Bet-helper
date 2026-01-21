# RL Dependencies Installation

Due to build issues with Poetry and `llvmlite`/`numba`, install the RL dependencies directly with pip:

```bash
# Activate your virtual environment
source venv/bin/activate

# Install RL dependencies
pip install stable-baselines3 gymnasium --no-build-isolation
```

This uses pre-built wheels and avoids building from source, which requires LLVM to be installed on your system.

## Verification

After installation, verify it works:

```python
from bet_helper.rl.data import load_gameweeks
from bet_helper.rl.env import BettingEnv
import gymnasium
import stable_baselines3

print("✓ All imports successful!")
```
