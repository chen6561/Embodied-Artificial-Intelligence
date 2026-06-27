# World Model Project

Minimal scaffold for an Isaac Lab to latent dynamics to MPC and policy evaluation workflow.

## Quickstart

Create random data:

```bash
python scripts/collect_isaac_data.py --config configs/franka_reach_state.yaml --policy_mode random
```

Create scripted data:

```bash
python scripts/collect_isaac_data.py --config configs/franka_reach_state.yaml --policy_mode scripted
```

Train the world model:

```bash
python scripts/train_world_model.py --config configs/franka_reach_state.yaml --dataset_dir data/raw/franka_reach_state/scripted
```

Evaluate one-step predictions:

```bash
python scripts/eval_rollout.py --config configs/franka_reach_state.yaml --dataset_dir data/raw/franka_reach_state/scripted --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt
```

Run MPC:

```bash
python scripts/run_mpc.py --config configs/franka_reach_state.yaml --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt
```

Score an action chunk:

```bash
python scripts/eval_policy_with_world_model.py --config configs/franka_reach_state.yaml --checkpoint outputs/checkpoints/franka_reach_state_world_model.pt
```
