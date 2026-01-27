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
        """Predict action given state"""
        pass

    @abstractmethod
    def train(self, total_timesteps: int, callback: Optional[Any] = None) -> Dict:
        """Train the agent"""
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
        Initialize RL Agent
        
        Args:
            env: Gymnasium environment (DEMSEnvironment)
            algorithm: 'PPO' or 'SAC'
            learning_rate: Learning rate for optimizer
            verbose: Verbosity level (0=none, 1=info, 2=debug)
            tensorboard_log: Path for tensorboard logs
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
        """Initialize the RL model based on algorithm choice"""
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
        Predict optimal action given current state
        
        Args:
            state: Current observation from environment
            deterministic: If True, use deterministic policy; else sample
            
        Returns:
            action: Predicted action
            info: Additional information dictionary
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
        Evaluate the agent's performance
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
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
        Save trained model to disk
        
        Args:
            path: File path to save model
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
        """Get training statistics"""
        return {
            "algorithm": self.algorithm,
            "training_runs": len(self.training_history),
            "total_timesteps": sum(h.get("timesteps", 0) for h in self.training_history),
            "model_initialized": self.model is not None,
        }

    def __repr__(self) -> str:
        return (
            f"RLAgent(algorithm={self.algorithm}, "
            f"obs_space={self.observation_space}, "
            f"action_space={self.action_space})"
        )
