# Enable forward references for type hints (supports using class name as type inside the class itself)
from __future__ import annotations

# Import dataclass decorator to automatically generate init, repr, eq, etc. for data storage classes
from dataclasses import dataclass

# Import NumPy for numerical array operations, random sampling and math calculations
import numpy as np


@dataclass
class TaskSpec:
    """
    Data class to store all static hyperparameters and dimensional specifications of a single robot task environment.
    All fixed task configuration parameters are encapsulated here for easy passing and access.
    """
    # Unique identifier string for the task type (e.g., "pick_place", "push_box")
    name: str
    # Dimension of the full observation vector returned by env.reset() and env.step()
    obs_dim: int
    # Dimension of the action vector that the policy sends to the environment step function
    action_dim: int
    # Dimension of the target goal vector used for sparse/dense reward calculation
    goal_dim: int
    # Maximum allowed timesteps per single episode; episode terminates automatically when reaching this step count
    max_steps: int
    # Euclidean distance threshold to judge task success: if distance between current state and goal < this value, success flag = True
    success_threshold: float


class DummyIsaacTask:
    """
    Simplified dummy simulation environment mimicking Isaac Sim robot task environments for offline RL algorithm testing.
    Implements standard Gym-style env API: reset(), step(action), plus a scripted expert action function for demonstration.
    Simulates a continuous control task where agent adjusts state variables to approach a randomly sampled target goal.
    """
    def __init__(self, spec: TaskSpec):
        """
        Initialize dummy simulation task with pre-defined task specification parameters.
        Args:
            spec: TaskSpec dataclass instance containing all dimensional and limit hyperparameters of the task
        """
        # Bind static task specification config to environment instance
        self.spec = spec
        # Counter tracking timesteps elapsed within current episode, resets on each env.reset() call
        self.step_count = 0
        # Full observation state vector of the agent, initialized to all zeros with float32 precision
        self.state = np.zeros(self.spec.obs_dim, dtype=np.float32)
        # Target goal vector the agent needs to reach, fixed per episode after reset
        self.goal = np.zeros(self.spec.goal_dim, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset environment to start a new independent episode: reinitialize step counter, randomize agent state and target goal.
        Goal-relevant state dimensions are offset with small Gaussian noise to simulate initial position offset from target.
        Returns:
            np.ndarray: Copy of the newly initialized full observation state vector (shape: [obs_dim], dtype: float32)
        """
        # Reset episode timestep counter back to zero for new trajectory
        self.step_count = 0
        # Randomize full observation state within uniform range [-0.5, 0.5]
        self.state = np.random.uniform(-0.5, 0.5, size=self.spec.obs_dim).astype(np.float32)
        # Randomize target goal vector within tighter uniform range [-0.3, 0.3]
        self.goal = np.random.uniform(-0.3, 0.3, size=self.spec.goal_dim).astype(np.float32)
        # Add small Gaussian noise to goal-aligned state dimensions to create initial offset between agent and target
        self.state[: self.spec.goal_dim] = self.goal + np.random.normal(0.0, 0.1, size=self.spec.goal_dim)
        # Return copied state array to prevent external in-place modification of internal env state
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one single timestep of environment dynamics given agent action, compute reward, termination flags and auxiliary info.
        Dynamics rule: action modifies first N(action_dim) dimensions of agent state with fixed step scaling factor.
        Reward is dense negative Euclidean distance to target goal (closer to goal = higher reward value).
        Episode terminates if agent reaches goal OR hits maximum allowed episode steps.
        Args:
            action: np.ndarray, agent policy output action vector (shape: [action_dim], float32)
        Returns:
            tuple:
                1. np.ndarray: New full observation state after applying action dynamics
                2. float: Dense reward value for current transition step
                3. bool: Done flag indicating whether current episode is fully terminated
                4. dict: Auxiliary information dictionary containing success flag, goal vector and state-goal distance
        """
        # Increment episode timestep counter after executing action step
        self.step_count += 1
        # Clip raw action values to [-1.0, 1.0] to bound control signal range
        clipped = np.clip(action, -1.0, 1.0)
        # Update state dimensions controlled by action: scaled action delta added to current state position
        self.state[: self.spec.action_dim] = self.state[: self.spec.action_dim] + 0.05 * clipped
        # Calculate L2 Euclidean distance between goal-matching state dimensions and target goal vector
        distance = np.linalg.norm(self.state[: self.spec.goal_dim] - self.goal)
        # Dense reward function: negative distance to goal (minimize distance = maximize cumulative reward)
        reward = -float(distance)
        # Judge task success: distance between agent state and target below predefined threshold
        success = distance < self.spec.success_threshold
        # Termination condition: episode ends on task success OR maximum timestep limit reached
        done = success or self.step_count >= self.spec.max_steps
        # Pack auxiliary diagnostic information for logging, evaluation and replay buffer storage
        info = {"success": success, "goal": self.goal.copy(), "distance": distance}
        # Return copied state array to isolate internal environment state from external code modifications
        return self.state.copy(), reward, done, info

    def scripted_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Hard-coded expert policy function to generate near-optimal heuristic actions for the task.
        Computes direct delta between observed goal-aligned state and target goal, amplifies and clips as valid control signal.
        Can be used to collect expert demonstration trajectories for imitation learning / offline RL.
        Args:
            obs: np.ndarray, full observation vector retrieved from environment reset/step
        Returns:
            np.ndarray: Heuristic expert action vector (shape: [action_dim], dtype: float32) bounded within [-1, 1]
        """
        # Initialize zero vector as base expert action with matching action dimension and float32 precision
        action = np.zeros(self.spec.action_dim, dtype=np.float32)
        # Compute position delta: target goal minus current observed goal-relevant state dimensions
        delta = self.goal - obs[: self.spec.goal_dim]
        # Scale delta by gain factor 4.0 and clip to valid action range [-1, 1], assign to controllable state dimensions
        action[: self.spec.goal_dim] = np.clip(delta * 4.0, -1.0, 1.0)
        return action


def make_task(config: dict) -> DummyIsaacTask:
    """
    Environment factory construction function: parse hierarchical config dictionary, build TaskSpec then instantiate DummyIsaacTask.
    Standard entry point for creating task environments from yaml/json loaded configuration dictionaries.
    Args:
        config: dict, nested configuration dict with top-level "task" key storing all task hyperparameters
    Returns:
        DummyIsaacTask: Fully initialized dummy simulation environment instance ready for reset/step calls
    """
    # Unpack nested task configuration dict fields into TaskSpec dataclass instance
    spec = TaskSpec(
        name=config["task"]["name"],
        obs_dim=config["task"]["obs_dim"],
        action_dim=config["task"]["action_dim"],
        goal_dim=config["task"]["goal_dim"],
        max_steps=config["task"]["max_steps"],
        success_threshold=config["task"]["success_threshold"],
    )
    # Construct and return dummy Isaac simulation task bound to the generated task specification
    return DummyIsaacTask(spec)