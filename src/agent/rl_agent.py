"""
RL Agent for DEMS
Uses stable-baselines3 for intelligent energy management
Implements PPO and SAC algorithms for DER control
"""

from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import os


class BaseRLAgent(ABC):
    """Base class for RL agents"""

    @abstractmethod
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Select an action for the given environment state.
        
        Parameters:
            state (np.ndarray): Current observation from the environment.
        
        Returns:
            tuple: `(action, info)` where `action` is an action compatible with the environment's action_space, and `info` is a dictionary with metadata about the decision (e.g., whether a trained model was used and which algorithm is configured).
        """
        pass

    @abstractmethod
    def train(self, total_timesteps: int, callback: Optional[Any] = None) -> Dict:
        """
        Train the underlying RL model for a specified number of timesteps.
        
        Parameters:
            total_timesteps (int): Number of environment timesteps to train the model.
            callback (Optional[Any]): Optional callback passed through to the learner during training.
        
        Returns:
            Dict: If the model is not initialized, a dictionary with status `"error"` and an explanatory `message`. If training runs, a dictionary with status `"success"`, the `algorithm` name, `total_timesteps` trained, and `training_episodes` (the number of recorded training runs).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent model"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent model"""
        pass


class RLAgent(BaseRLAgent):
    """
    Reinforcement Learning Agent for Dynamic Energy Management
    Supports PPO (default) and SAC algorithms
    Trained to optimize energy distribution, DER control, and grid stability
    """

    def __init__(
        self,
        env: Any,
        algorithm: str = "PPO",
        learning_rate: float = 0.0003,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Create an RL agent bound to an environment and configured for a chosen algorithm.
        
        Parameters:
            env (Any): Gymnasium-compatible environment providing `observation_space` and `action_space`.
            algorithm (str): Algorithm identifier, either "PPO" or "SAC". Determines the underlying model type.
            learning_rate (float): Optimizer learning rate for the chosen algorithm.
            verbose (int): Verbosity level (0 = silent, 1 = info, 2 = debug).
            tensorboard_log (Optional[str]): Path to write TensorBoard logs, or `None` to disable.
        """
        self.env = env
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.model = None
        self.training_history = []
        
        # Get observation and action space from environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """
        Create and assign a stable-baselines3 model for the agent based on the configured algorithm.
        
        Initializes self.model to a PPO or SAC instance configured with the agent's environment and hyperparameters. If self.algorithm is not 'PPO' or 'SAC', raises a ValueError. If stable_baselines3 cannot be imported, prints a warning, sets self.model to None, and leaves the agent in random-action mode until a model is loaded.
        """
        try:
            if self.algorithm.upper() == "PPO":
                from stable_baselines3 import PPO
                self.model = PPO(
                    policy="MlpPolicy",
                    env=self.env,
                    learning_rate=self.learning_rate,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    normalize_advantage=True,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[128, 128], vf=[128, 128]),
                    ),
                    verbose=self.verbose,
                    tensorboard_log=self.tensorboard_log,
                )
            elif self.algorithm.upper() == "SAC":
                from stable_baselines3 import SAC
                self.model = SAC(
                    policy="MlpPolicy",
                    env=self.env,
                    learning_rate=self.learning_rate,
                    buffer_size=1_000_000,
                    learning_starts=100,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    verbose=self.verbose,
                    tensorboard_log=self.tensorboard_log,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}. Use 'PPO' or 'SAC'")
        except ImportError as e:
            print(f"Warning: Could not import stable_baselines3: {e}")
            print("Agent will operate in random action mode until trained model is loaded.")
            self.model = None

    def predict(self, state: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Return an action for the given environment state using the trained model or a random action if no model is available.
        
        Parameters:
        	state (np.ndarray): Observation from the environment.
        	deterministic (bool): If True, use the policy deterministically; if False, allow stochastic action sampling.
        
        Returns:
        	action (np.ndarray): Action to apply in the environment.
        	info (dict): Metadata about the prediction with keys:
        		- "info": "trained" if a model produced the action, "untrained" if the action was sampled randomly.
        		- "algorithm": Name of the configured algorithm (e.g., "PPO" or "SAC").
        """
        if self.model is None:
            # Random action if model not initialized
            action = self.action_space.sample()
            return action, {"info": "untrained", "algorithm": self.algorithm}

        # Use trained model for prediction
        action, _states = self.model.predict(state, deterministic=deterministic)
        return action, {"info": "trained", "algorithm": self.algorithm}

    def train(self, total_timesteps: int, callback: Optional[Any] = None) -> Dict:
        """
        Train the agent on the environment
        
        Args:
            total_timesteps: Number of timesteps to train
            callback: Optional callback for monitoring training
            
        Returns:
            Dictionary with training statistics
        """
        if self.model is None:
            return {
                "status": "error",
                "message": "Model not initialized. Check stable_baselines3 installation."
            }
        
        print(f"Training {self.algorithm} agent for {total_timesteps} timesteps...")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )
        
        # Update training history
        self.training_history.append({
            "timesteps": total_timesteps,
            "algorithm": self.algorithm,
        })
        
        return {
            "status": "success",
            "algorithm": self.algorithm,
            "total_timesteps": total_timesteps,
            "training_episodes": len(self.training_history),
        }

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Compute the mean and standard deviation of episode rewards for the current model over a number of evaluation episodes.
        
        Parameters:
            n_episodes (int): Number of episodes to run for evaluation.
        
        Returns:
            dict: If the model is initialized, returns {
                "mean_reward": float(mean reward across evaluated episodes),
                "std_reward": float(standard deviation of rewards),
                "n_episodes": n_episodes
            }. If the model is not initialized, returns {"error": "Model not initialized"}.
        """
        if self.model is None:
            return {"error": "Model not initialized"}
        
        from stable_baselines3.common.evaluation import evaluate_policy
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )
        
        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_episodes": n_episodes,
        }

    def save(self, path: str) -> None:
        """
        Persist the current trained model to the given filesystem path.
        
        Creates parent directories if they do not exist and saves the model file at the provided path. If no model is initialized, no file is written and a warning is emitted.
        
        Parameters:
            path (str): Destination filesystem path (including filename) where the model will be saved.
        """
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("Warning: No model to save")

    def load(self, path: str) -> None:
        """
        Load trained model from disk
        
        Args:
            path: File path to load model from
        """
        try:
            if self.algorithm.upper() == "PPO":
                from stable_baselines3 import PPO
                self.model = PPO.load(path, env=self.env)
            elif self.algorithm.upper() == "SAC":
                from stable_baselines3 import SAC
                self.model = SAC.load(path, env=self.env)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_training_stats(self) -> Dict:
        """
        Retrieve aggregated training statistics for the agent.
        
        Returns:
            stats (dict): A dictionary with the following keys:
                - algorithm (str): The algorithm name configured for the agent.
                - training_runs (int): Number of recorded training runs.
                - total_timesteps (int): Sum of timesteps across all recorded training runs.
                - model_initialized (bool): `True` if a model instance is initialized, `False` otherwise.
        """
        return {
            "algorithm": self.algorithm,
            "training_runs": len(self.training_history),
            "total_timesteps": sum(h.get("timesteps", 0) for h in self.training_history),
            "model_initialized": self.model is not None,
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the agent.
        
        Returns:
        	str: A string showing the algorithm, observation space, and action space.
        """
        return (
            f"RLAgent(algorithm={self.algorithm}, "
            f"obs_space={self.observation_space}, "
            f"action_space={self.action_space})"
        )