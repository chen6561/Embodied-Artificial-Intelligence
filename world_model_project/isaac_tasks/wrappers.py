from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskSpec:
    name: str
    obs_dim: int
    action_dim: int
    goal_dim: int
    max_steps: int
    success_threshold: float


class DummyIsaacTask:
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.step_count = 0
        self.state = np.zeros(self.spec.obs_dim, dtype=np.float32)
        self.goal = np.zeros(self.spec.goal_dim, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.state = np.random.uniform(-0.5, 0.5, size=self.spec.obs_dim).astype(np.float32)
        self.goal = np.random.uniform(-0.3, 0.3, size=self.spec.goal_dim).astype(np.float32)
        self.state[: self.spec.goal_dim] = self.goal + np.random.normal(0.0, 0.1, size=self.spec.goal_dim)
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        clipped = np.clip(action, -1.0, 1.0)
        self.state[: self.spec.action_dim] = self.state[: self.spec.action_dim] + 0.05 * clipped
        distance = np.linalg.norm(self.state[: self.spec.goal_dim] - self.goal)
        reward = -float(distance)
        success = distance < self.spec.success_threshold
        done = success or self.step_count >= self.spec.max_steps
        info = {"success": success, "goal": self.goal.copy(), "distance": distance}
        return self.state.copy(), reward, done, info

    def scripted_action(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.spec.action_dim, dtype=np.float32)
        delta = self.goal - obs[: self.spec.goal_dim]
        action[: self.spec.goal_dim] = np.clip(delta * 4.0, -1.0, 1.0)
        return action


def make_task(config: dict) -> DummyIsaacTask:
    spec = TaskSpec(
        name=config["task"]["name"],
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        goal_dim=config["task"]["goal_dim"],
        max_steps=config["task"]["max_steps"],
        success_threshold=config["task"]["success_threshold"],
    )
    return DummyIsaacTask(spec)
