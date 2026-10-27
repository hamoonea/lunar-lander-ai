# Lunar Lander AI

A reinforcement-learning project that trains a neural-network pilot to land a rocket in **Gymnasium's `LunarLander-v3`** environment using **Proximal Policy Optimization (PPO)** from Stable Baselines3.

Run the included trained pilot, compare it with random actions, train your own agent, and export flight recordings. Everything runs locally on CPU—no API keys or cloud account required.

![Trained pilot on the landing pad](demo/preview.png)

**[Watch the three-flight demo](demo/landing.mp4)** · **[Trained model](models/lander-ppo.zip)** · **[Evaluation results](results/trained.json)**

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Results and evaluation](#results-and-evaluation)
- [How the agent learns](#how-the-agent-learns)
- [Training your own agent](#training-your-own-agent)
- [Watching and recording flights](#watching-and-recording-flights)
- [Command reference](#command-reference)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations and experiments](#limitations-and-experiments)
- [References](#references)

## Features

- A trained PPO checkpoint ready to run without retraining.
- Training with four simulation environments, periodic validation, and saved checkpoints.
- Separate evaluation commands for learned and random policies using repeatable episode seeds.
- Live playback in a Pygame window or MP4 recording without a live window.
- CSV training metrics, per-episode monitor logs, and JSON evaluation reports.
- Resume support and protection against accidentally overwriting a training directory.

## Quick start

### 1. Clone the repository

```sh
git clone https://github.com/hamoonea/lunar-lander-ai.git
cd lunar-lander-ai
```

### 2. Create an environment and install dependencies

Python **3.11–3.13** is recommended. The project was tested with Python **3.13.2 on an Apple Silicon Mac**. Other operating systems have not been verified in this project.

**macOS / Linux:**

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use the Python version you have installed if it differs from the Windows example. If PowerShell prevents activation, invoke `.\.venv\Scripts\python.exe` directly in place of `python`.

`requirements.txt` specifies supported dependency ranges. For the exact package versions used in the recorded run, install `requirements-lock.txt` instead. That file records the tested Mac environment; it does not guarantee identical behavior across platforms.

### 3. Watch the included pilot

```sh
python lander.py watch --model models/lander-ppo.zip
```

This opens a window and runs three flights. You do **not** need to train first. Keep `--model models/lander-ppo.zip` when using the bundled checkpoint: the CLI's default model path points to the best checkpoint from a new local training run.

For a quick check without a display window:

```sh
python lander.py evaluate --model models/lander-ppo.zip --episodes 5
```

## Results and evaluation

The included model was evaluated on **100 episodes**, using seeds **20000–20099**. The random baseline uses the same environment seeds.

| Policy | Mean episode reward | Standard deviation | Episodes |
| --- | ---: | ---: | ---: |
| Random actions | −185.45 | 111.99 | 100 |
| Included PPO pilot | **222.46** | **30.93** | 100 |

A mean reward of **200 over 100 episodes** is the conventional benchmark target used here. Reward is a cumulative score, **not a landing success percentage**. The standard deviation describes variation across episodes, not uncertainty across independent training runs.

The bundled policy is the **400,000-transition checkpoint** from training seed **42**. The run requested one million transitions but was stopped at **524,288** after the selected checkpoint passed the evaluation target. The original requested settings are in [results/training-config.json](results/training-config.json).

This is one training run. It demonstrates the included policy's performance on the reported seeds; it does not establish reliability across training seeds or different physics settings. The test result informed when training was stopped, so it should not be treated as an untouched final benchmark for future tuning.

### Reproduce the comparison

```sh
python lander.py evaluate --random --episodes 100 --seed 20000 --output runs/random.json
python lander.py evaluate --model models/lander-ppo.zip --episodes 100 --seed 20000 --output runs/trained.json
```

The command prints each episode's reward and length, followed by a summary. `--output` also saves the policy path, starting seed, episode count, mean reward, population standard deviation, and individual rewards as JSON. Episode `i` uses the starting seed plus `i`, counting from zero.

Published reports are preserved in [results/random.json](results/random.json) and [results/trained.json](results/trained.json). New results go into `runs/` in the examples. For an additional check on new starting conditions, choose another seed range, such as `--seed 30000`.

## How the agent learns

### Environment and observations

Gymnasium supplies the simulated physics and rewards. At each step, the agent receives eight numbers:

| Observation | Meaning |
| --- | --- |
| x and y | Lander position |
| x and y velocity | Horizontal and vertical movement |
| Angle | Lander orientation |
| Angular velocity | How quickly the lander rotates |
| Two leg-contact indicators | Whether each leg touches the ground |

The policy uses these numerical observations rather than camera images. It chooses one of four discrete actions:

| Action | Control |
| ---: | --- |
| 0 | Do nothing |
| 1 | Fire the left orientation engine |
| 2 | Fire the main engine |
| 3 | Fire the right orientation engine |

### Rewards

The project uses Gymnasium's standard Lunar Lander reward without a custom reward wrapper. It encourages approaching the landing pad, reducing speed, staying upright, and making leg contact. Engine use has a cost, and crashes and safe landings affect the score.

Keeping the original reward makes this project's scores easier to compare with the standard environment. Changing rewards would change the meaning of the reported score.

### PPO training loop

1. The policy observes the lander and chooses an action.
2. Gymnasium advances the simulation and returns a reward and new observation.
3. PPO collects experience from four environments.
4. PPO updates the policy to favor actions associated with better returns, while limiting the size of policy changes.
5. Periodic validation evaluates the current policy and saves it if its mean reward improves.

The policy is a multilayer perceptron (`MlpPolicy`). Training and inference use CPU, with PyTorch configured to use one compute thread. The four environments use Stable Baselines3's default vector-environment implementation; this project does not launch four separate training worker processes.

## Training your own agent

```sh
python lander.py train --steps 1000000 --run runs/my-lander --seed 42
```

Use a **new directory** for every run. The command intentionally fails if `--run` already exists. Training time depends on hardware and episode length; the requested budget is not a guarantee that the agent will reach the target score.

PPO collects complete rollouts, so the actual transition count can exceed `--steps`. With the configuration below, a rollout contains 4,096 transitions.

### Training configuration

These settings are defined in [lander.py](lander.py). Only step budget, run directory, seed, and resume checkpoint are exposed as training CLI options.

| Setting | Value |
| --- | --- |
| Environment | `LunarLander-v3` |
| Algorithm / policy | PPO / `MlpPolicy` |
| Simulation environments | 4 |
| Steps per environment per rollout | 1,024 |
| Total transitions per rollout | 4,096 |
| Minibatch size | 64 |
| Optimization epochs per rollout | 4 |
| Discount factor (`gamma`) | 0.999 |
| GAE parameter (`gae_lambda`) | 0.98 |
| Entropy coefficient | 0.01 |
| Device / PyTorch threads | CPU / 1 |
| Validation interval | 20,000 transitions |
| Episodes per validation | 10 |
| Periodic checkpoint interval | 100,000 transitions |

Other PPO options retain the installed Stable Baselines3 version's defaults. Validation uses deterministic predictions and a separate environment initialized with `training seed + 10000`; its random sequence then advances between evaluations. The separate evaluation CLI resets each episode to an explicit seed.

### Saved outputs

For `--run runs/my-lander`:

| Path | Contents |
| --- | --- |
| `config.json` | Requested CLI settings, not a full dependency or hyperparameter snapshot |
| `progress.csv` | Training metrics written by Stable Baselines3 |
| `episodes/` | Per-environment episode monitor logs |
| `eval/evaluations.npz` | Validation timesteps, rewards, and episode lengths |
| `best/best_model.zip` | Policy with the highest validation mean so far |
| `checkpoints/` | Periodic model snapshots |
| `final_model.zip` | Most recent policy saved when training ends |

The best policy and final policy may differ. A short run below the first validation interval will not produce `best/best_model.zip`; use `final_model.zip` instead.

To inspect validation history:

```python
import numpy as np

data = np.load("runs/my-lander/eval/evaluations.npz")
for steps, rewards in zip(data["timesteps"], data["results"]):
    print(f"{steps:>8} transitions: mean reward {rewards.mean():.1f}")
```

### Stop and resume

Press **Ctrl-C** during learning to save the current policy as `final_model.zip`. A forced process termination or system shutdown cannot guarantee that save.

Resume into a new directory:

```sh
python lander.py train --resume runs/my-lander/final_model.zip --run runs/my-lander-v2 --steps 500000
```

Here, `--steps` requests **additional** training transitions. The model and optimizer are loaded, while new environments, callbacks, and logs are created. This is a continuation of training, not a bit-for-bit restoration of the interrupted simulation. The new run selects its own best model from its own validation evaluations.

### Quick pipeline check

```sh
python lander.py train --steps 4096 --run runs/smoke-test
python lander.py evaluate --model runs/smoke-test/final_model.zip --episodes 2
```

This checks training, saving, and loading. A 4,096-step model is not expected to land well. Choose a fresh directory if repeating the check.

## Watching and recording flights

Watch five flights from the bundled policy:

```sh
python lander.py watch --model models/lander-ppo.zip --episodes 5
```

Watch random controls for comparison:

```sh
python lander.py watch --random --episodes 3
```

Export three flights to a single MP4:

```sh
python lander.py watch --model models/lander-ppo.zip --episodes 3 --video runs/my-demo.mp4
```

`--video` records frames instead of opening a live window, and works with both `watch` and `evaluate`. Episodes are concatenated in order. Add `--output runs/demo-results.json` to also save their scores. Recording uses ImageIO and its FFmpeg dependency.

## Command reference

```sh
python lander.py --help
python lander.py train --help
python lander.py evaluate --help
python lander.py watch --help
```

### `train`

| Option | Default | Purpose |
| --- | --- | --- |
| `--steps` | `1000000` | Positive training-transition budget |
| `--run` | `runs/lander` | New output directory; must not already exist |
| `--seed` | `42` | Training seed |
| `--resume` | None | Load a checkpoint and continue training |

### `evaluate` and `watch`

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `runs/lander/best/best_model.zip` | Policy checkpoint; specify `models/lander-ppo.zip` for the included pilot |
| `--random` | Off | Use random actions instead of loading a model |
| `--episodes` | 10 for evaluate; 3 for watch | Positive number of flights |
| `--seed` | `20000` | First episode's seed |
| `--video` | None | Output MP4 path; replaces live rendering |
| `--output` | None | Output JSON report path |

Evaluation and playback use deterministic policy predictions. Output parent directories are created automatically. Unlike training directories, existing JSON and video output files can be overwritten; use distinct filenames to keep earlier reports.

## Project structure

```text
lunar-lander-ai/
├── lander.py                   # Training, evaluation, playback, and recording CLI
├── requirements.txt            # Dependency version ranges
├── requirements-lock.txt       # Exact versions from the tested environment
├── models/
│   └── lander-ppo.zip          # Included 400,000-transition checkpoint
├── demo/
│   ├── landing.mp4             # Three recorded flights
│   └── preview.png             # README preview
├── results/
│   ├── random.json             # 100-episode random baseline
│   ├── trained.json            # 100-episode PPO evaluation
│   └── training-config.json    # Original requested training settings
└── runs/                       # Generated local outputs; ignored by Git
```

The virtual environment and Python cache files are also ignored by Git. The `runs/` directory is created when needed and is not included in a fresh clone.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv` and run `python -m pip install -r requirements.txt` using the same interpreter that runs `lander.py`. |
| Box2D installation fails | If pip is building Box2D from source, install SWIG with `python -m pip install swig` and retry. Native builds can also require platform compiler tools. |
| `FileExistsError` during training | Select a new `--run` directory. To continue training, also pass `--resume` with the existing checkpoint path. |
| Model file is missing | For the included policy use `--model models/lander-ppo.zip`. For a short training run use its `final_model.zip`. |
| No graphical display is available | Use `evaluate` for scores or add `--video runs/demo.mp4` for recording instead of live playback. |
| Video export fails | Check that the `imageio[ffmpeg]` dependency installed successfully and that the output directory is writable. |
| Reward drops during training | PPO performance can fluctuate. Evaluate the best checkpoint, inspect validation history, and try additional training or a separate seed. |
| Rerunning commands gives different results | Compare dependency versions, hardware, seeds, and checkpoint paths. Fixed seeds alone do not guarantee identical training across platforms. |

## Limitations and experiments

This is a simulated control project, not a real spacecraft flight controller. The included result covers one training seed and the standard environment settings. The CLI records rewards and episode lengths; it does not classify every episode as a successful landing or crash.

Possible extensions, not currently implemented:

- Train several independent seeds and compare their evaluation distributions.
- Plot reward curves from `progress.csv` and `evaluations.npz`.
- Test robustness to wind or altered gravity using Gymnasium environment options.
- Compare PPO with another algorithm that supports discrete actions.
- Experiment with custom rewards while reporting standard-environment evaluation separately.

## References

- [Gymnasium repository](https://github.com/Farama-Foundation/Gymnasium)
- [Lunar Lander environment documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

Gymnasium provides the environment; Stable Baselines3 provides PPO; PyTorch performs neural-network training; ImageIO and FFmpeg handle video export.
