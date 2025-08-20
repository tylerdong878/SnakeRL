## SnakeRL

### A reinforcement learning Snake agent with live visualization, fast training, and multi-screen viewer

SnakeRL lets you: play Snake, train a PPO agent at max speed on CPU/GPU, watch training live at normal game speed, render many concurrent games in a single window, and evaluate saved models — all with simple commands.

## Tech stack
- Core
  - **Python 3.11**
  - **Poetry** (dependency management & virtual env)
- RL & ML
  - **PyTorch** (deep learning backend)
  - **Stable-Baselines3** (PPO, VecEnv, VecNormalize)
  - **Gymnasium** (environment API)
  - **NumPy** (array operations)
- Rendering & UI
  - **Pygame** (game rendering and input)
- Logging & Visualization
  - **TensorBoard** (training metrics)
  - **Matplotlib** (optional plotting)
- Dev Tooling & Testing
  - **PyTest** (unit tests)
  - **Black** (code formatting)
  - **Flake8** (linting)

## Features
- Fast headless training with parallel environments and GPU support
- Single-screen live training at normal FPS
- Multi-screen viewer: render N games in a grid while training a single shared policy
- Clean Gymnasium env with normalized observations
- Optional VecNormalize for observation/reward normalization (saved and auto-loaded at eval)
- Easy evaluation and model versioning

## Install
```powershell
pip install poetry
poetry install
```

## Quickstart
- Play the raw game (manual):
```powershell
poetry run python -m src.snake_game
```

- Fast training (terminal only, max speed):
```powershell
# Finite run, saves when done (use your own name)
poetry run python src/train.py --n-envs 8 --device auto --total-timesteps 200000 --model-path models/model_name

# Indefinite run, saves on CTRL+C
poetry run python src/train.py --n-envs 8 --device auto --chunk-steps 200000 --model-path models/model_name
```

- Live training (single screen, normal speed):
```powershell
poetry run python src/train_watch.py --device auto --max-steps 1000 --chunk-steps 10000 --model-path models/model_name
```

- Live training (multi-screen grid, auto-scaled tiles):
```powershell
# 36 games in a window (auto grid/tiles)
poetry run python src/train_watch_multi.py --n-envs 36 --device auto --model-path models/model_name

# Larger window or fixed columns
poetry run python src/train_watch_multi.py --n-envs 36 --window-width 1200 --window-height 800 --cols 10
```

- Evaluate a saved model (watch it play):
```powershell
poetry run python src/evaluate.py --model-path models/model_name.zip --episodes 5 --render

# Watch with custom board size
poetry run python src/evaluate.py --model-path models/model_name.zip --episodes 10 --render --width 500 --height 500 --grid-size 50
```

- Watch a trained model play (with custom board size):
```powershell
# Small 20x20 grid for quick games
poetry run python src/watch.py --width 400 --height 400 --grid-size 20

# Large 50x50 grid for longer games
poetry run python src/watch.py --width 1000 --height 1000 --grid-size 20
```

- TensorBoard (metrics):
```powershell
poetry run tensorboard --logdir logs --port 6006
```

### Reading TensorBoard (what to look at)
- Scalars tab → set X-Axis to Step, Smoothing ~0.6
- Select the latest run (e.g., PPO_*) in the left sidebar
- Key charts:
  - rollout/ep_rew_mean: average episodic reward. Should trend upward.
  - rollout/ep_len_mean: average episode length. Often rises as survival improves.
  - train/value_loss: value function loss. Should stabilize/downtrend; spikes can happen.
  - train/entropy_loss: policy entropy. Typically decreases as the policy becomes more certain.
  - train/approx_kl: PPO KL divergence (~0.01–0.02 is healthy). Large values can mean too big updates.
  - train/clip_fraction: fraction of clipped policy updates. Persistent high values may indicate overly large steps.
  - time/fps: training throughput (higher is faster).
- Comparing runs: tick multiple PPO_* runs in the sidebar to overlay curves.
- Troubleshooting empty graphs:
  - Ensure events exist: `dir logs -Recurse -Filter *.tfevents*`
  - Keep TensorBoard running while training for live updates
  - Point to the parent `logs/` dir, not a single run subfolder

## Normalized observations
The environment returns float32 features scaled to [0,1]:
- **head_x_norm, head_y_norm**: head position / (grid dims - 1)
- **food_x_norm, food_y_norm**: food position / (grid dims - 1)
- **direction_norm**: direction(0..3)/3
- **length_norm**: length / (grid_width*grid_height)

This improves stability and speed of PPO.

## Optional: VecNormalize
Enable runtime observation/reward normalization and save stats:
```powershell
poetry run python src/train.py \
  --n-envs 8 --device auto \
  --vecnorm --vecnorm-path models/vecnormalize.pkl \
  --total-timesteps 200000 \
  --model-path models/model_name

# evaluate auto-loads vecnormalize stats when present
poetry run python src/evaluate.py --model-path models/model_name.zip --episodes 5 --render
```

## Reward shaping (configurable)
Defaults in the env:
- Food eaten: +10.0
- Per-step: -0.1
- Death: -10.0

Optional (off by default; enable via flags in training):
- Distance shaping (`--distance-shaping X`): +X when moving closer to food, -X when moving away.
- Dynamic food scaling (`--dynamic-food-length-scale K`): scales food reward by (1 + length/K).
- Efficiency bonus (`--efficiency-bonus-coeff C`): adds C * (1/steps_since_last_food) on eating.

Example training with shaping:
```powershell
poetry run python src/train.py \
  --n-envs 8 --device auto --chunk-steps 200000 \
  --reward-food 10 --reward-death -100 --reward-step -0.1 \
  --distance-shaping 0.1 \
  --dynamic-food-length-scale 10 \
  --efficiency-bonus-coeff 5
```

## Key CLI flags
- **--n-envs**: number of parallel envs (CPU). More = faster (up to your cores)
- **--device**: cpu | cuda | auto
- **--total-timesteps**: finite run (saves model when complete)
- **--chunk-steps**: chunk size for indefinite training loops
- **--max-steps**: per-episode step cap in the env (truncates episode). Set 0 to disable (unlimited; ends only on death).
- **--model-path**: base path for the saved model (.zip appended)
- **--width/--height/--grid-size**: board dimensions in pixels (default: 1000x1000 with 100px grid)
- **--window-width/--window-height** (multi-screen): size of grid window; tiles auto-scale
- **--cols** (multi-screen): force grid columns (otherwise auto)
- **--vecnorm/--vecnorm-path**: enable VecNormalize and set stats file path
- **Reward shaping** (training):
  - `--reward-food`: base reward for eating food (default 10.0)
  - `--reward-step`: per-step penalty (default -0.1)
  - `--reward-death`: penalty on death (default -10.0)
  - `--distance-shaping`: +/- when moving closer/farther from food (default 0.0 = off)
  - `--dynamic-food-length-scale`: scale food reward by (1 + length/K) (default 0.0 = off)
  - `--efficiency-bonus-coeff`: add C*(1/steps_since_last_food) on eating (default 0.0 = off)

## Controls (manual game)
- Arrow keys: move
- R: restart
- ESC: quit

## Stopping runs
- Terminal training: CTRL+C (saves model).
- Live training (single/multi): click the window X (saves model).

## Windows: fix editor import warnings
If your editor shows “import could not be resolved”, point it to Poetry’s Python:
```powershell
poetry env info --path
```
Then in Cursor/VS Code: Ctrl+Shift+P → “Python: Select Interpreter” → “Enter interpreter path…” → paste <that-path>\Scripts\python.exe.

## Project layout
```
SnakeRL/
  src/
    snake_game.py         # Pygame Snake
    snake_env.py          # Gymnasium env (normalized obs)
    train.py              # Fast headless training (parallel, GPU)
    train_watch.py        # Single-screen live training
    train_watch_multi.py  # Multi-screen live training (grid)
    evaluate.py           # Evaluate saved model
  tests/
    test_snake_game.py
    test_snake_env.py
  models/                 # Saved models & vecnormalize stats
  logs/                   # TensorBoard & monitor CSVs
```