from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import GraphLabelingEnv
import numpy as np

class RLAgent:
    def __init__(self, graph):
        self.graph = graph
        self.env = DummyVecEnv([lambda: GraphLabelingEnv(graph)])
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self):
        reset_output = self.env.reset()
        
        # **Handle different reset output formats**
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, _ = reset_output
        else:
            obs = reset_output  # Handle cases where only obs is returned

        for _ in range(self.graph.number_of_nodes()):
            action, _ = self.model.predict(obs)
            
            # **Ensure action is always a NumPy array**
            action = np.array([action]) if isinstance(action, int) else action  

            step_result = self.env.step(action)

            # **Ensure correct unpacking**
            if isinstance(step_result, tuple) and len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                raise ValueError(f"Unexpected step() output format: {step_result}")

        return self.env.envs[0].get_labeled_graph()




    
