# World Model Project

This project is a minimal research scaffold for the workflow:

```text
Isaac Lab -> data collection -> latent dynamics -> MPC -> policy evaluation -> sim2real interface
```

The current version already runs end to end, but it uses a lightweight placeholder environment in
`isaac_tasks/wrappers.py` instead of a real Isaac Lab task. The purpose of this first version is to
let us validate the whole training and evaluation loop before replacing the environment wrapper with
an actual Isaac Lab Franka task.

## What Is Included

This scaffold covers five things:

1. Collect transition data from an environment.
2. Train a world model that predicts `next_obs`, `reward`, `done`, and `success`.
3. Evaluate one-step prediction quality.
4. Run CEM-based MPC with the learned world model.
5. Score policy action chunks with the world model.

## Project Layout

```text
world_model_project/
  configs/
    franka_reach_state.yaml
    franka_lift_state.yaml
  data/
    raw/
    processed/
  scripts/
    collect_isaac_data.py
    train_world_model.py
    eval_rollout.py
    run_mpc.py
    eval_policy_with_world_model.py
  wm/
    datasets.py
    encoders.py
    dynamics.py
    losses.py
    mpc.py
    metrics.py
  policies/
    bc_policy.py
    act_adapter.py
    diffusion_policy_adapter.py
  isaac_tasks/
    wrappers.py
  outputs/
    checkpoints/
    videos/
    logs/
```

## Current Assumptions

- The project currently uses a dummy task wrapper rather than a real Isaac Lab environment.
- Observations are state-based only.
- The world model is an MLP latent dynamics model.
- MPC uses CEM.
- Policy evaluation currently uses a simple BC policy placeholder plus adapters.

That means this repo is ideal for learning the pipeline and for quickly testing design ideas.

## Environment Setup

Use Python 3.10 or newer. Install the following packages:

```bash
pip install torch numpy h5py pyyaml
```

If you want to switch to a GPU device later, update the config file:

```yaml
train:
  device: cuda
```

## Config Files

Two example configs are included:

- `configs/franka_reach_state.yaml`
- `configs/franka_lift_state.yaml`

They define:

- task dimensions
- number of episodes
- batch size
- model size
- optimizer settings
- CEM planner settings

## Step-By-Step Usage

Run all commands from the project root:

```bash
cd work/world_model_project
```

### 1. Collect Data

Collect random exploration data:

```bash
python scripts/collect_isaac_data.py --config configs/franka_reach_state.yaml --policy_mode random
```

Collect scripted success-oriented data:

```bash
python scripts/collect_isaac_data.py --config configs/franka_reach_state.yaml --policy_mode scripted
```

Collect noisy scripted data:

```bash
python scripts/collect_isaac_data.py --config configs/franka_reach_state.yaml --policy_mode noisy_scripted
```

Generated files will appear under:

```text
data/raw/franka_reach_state/<policy_mode>/
```

Each episode is stored as one HDF5 file with:

```text
obs
action
next_obs
reward
done
success
```

### 2. Train The World Model

Train on one dataset split, for example scripted data:

```bash
python scripts/train_world_model.py --config configs/franka_reach_state.yaml --dataset_dir data/raw/franka_reach_state/scripted
```

This writes:

```text
outputs/checkpoints/franka_reach_state_world_model.pt
outputs/checkpoints/franka_reach_state_train_history.json
```

### 3. Evaluate Prediction Quality

Evaluate one-step prediction error:

```bash
python scripts/eval_rollout.py --config configs/franka_reach_state.yaml --dataset_dir data/raw/franka_reach_state/scripted --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt
```

This prints summary metrics such as:

- observation MSE
- reward MSE
- done prediction accuracy

### 4. Run MPC

Run the CEM planner inside the environment:

```bash
python scripts/run_mpc.py --config configs/franka_reach_state.yaml --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt
```

This reports:

- number of episodes
- number of successful episodes
- success rate

### 5. Evaluate A Policy With The World Model

Score an action chunk produced by a policy adapter:

```bash
python scripts/eval_policy_with_world_model.py --config configs/franka_reach_state.yaml --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt
```

This prints:

- predicted return
- predicted success score

In the current scaffold, the policy is a simple placeholder BC model. Later, the same interface can
be replaced by ACT, Diffusion Policy, or SmolVLA-style action chunk generators.

## How The Main Components Work

### `isaac_tasks/wrappers.py`

This file hides the environment implementation behind a small interface:

```python
obs = env.reset()
next_obs, reward, done, info = env.step(action)
action = env.scripted_action(obs)
```

When you replace the dummy task with a real Isaac Lab task, the rest of the project can stay mostly
unchanged.

### `wm/dynamics.py`

Defines the world model:

- `encoder(obs) -> latent`
- `transition(latent, action) -> next_latent`
- `decoder(next_latent) -> next_obs`
- reward head
- done head
- success head

### `wm/mpc.py`

Implements a simple CEM planner:

- sample candidate action sequences
- rollout them through the learned dynamics
- rank by predicted return and success
- update the action distribution
- execute the first action

### `policies/`

Contains policy interfaces:

- `bc_policy.py`: a minimal behavior cloning policy
- `act_adapter.py`: adapter for ACT-style chunked actions
- `diffusion_policy_adapter.py`: adapter for Diffusion Policy-style action chunks

## Recommended Learning Order

If you are studying this project, use it in this order:

1. Run data collection.
2. Train a world model on scripted data.
3. Evaluate one-step prediction quality.
4. Run MPC and compare it with scripted behavior.
5. Use the world model to score action chunks.
6. Replace the dummy task with a real Isaac Lab environment.
7. Add image observations and a stronger encoder.

## How To Replace The Dummy Task With Isaac Lab

The main migration path is:

1. Keep all training and evaluation scripts unchanged.
2. Replace `DummyIsaacTask` in `isaac_tasks/wrappers.py`.
3. Make sure the new wrapper returns:

```python
obs: np.ndarray
reward: float
done: bool
info["success"]: bool
```

4. Match the config dimensions:

```yaml
task:
  obs_dim: ...
  action_dim: ...
```

Good first replacements:

- Franka Reach
- Franka Lift
- state-only Isaac Lab task before adding RGB

## Suggested Next Improvements

After this first scaffold, the most valuable upgrades are:

1. Add multi-step rollout loss to training.
2. Add a real BC trainer for the placeholder policy.
3. Replace the dummy task with an Isaac Lab Franka task.
4. Add image + proprioception latent dynamics.
5. Add uncertainty estimation for safer MPC.
6. Add a unified `ObsPacket` and `ActionPacket` for sim2real.

## Notes

- This version is intentionally simple and research-friendly.
- It is designed to teach the structure of a world-model project before the full Isaac Lab
  integration work.
- The `processed/`, `videos/`, and `logs/` directories are placeholders for the next iteration.
