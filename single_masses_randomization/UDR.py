from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import copy

class UDRCallback(BaseCallback):
    """
    A custom callback that applies Uniform Domain Randomization (UDR) on the masses of the environment.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param mass: Index or indices of the masses to be randomized.
    """

    def __init__(self, env, range_value, verbose: int = 0, mass=None):
        super().__init__(verbose)
        self.training_env = env
        self.range_value = range_value
        self.original_masses = copy.deepcopy(self.training_env.sim.model.body_mass)
        
        # Set the indices of the masses to randomize
        if mass in [1, 2, 3]:
            self.mass_indices = [mass+1]
        elif mass == 12:
            self.mass_indices = [2, 3]
        elif mass == 13:
            self.mass_indices = [2, 4]
        elif mass == 23:
            self.mass_indices = [3, 4]
        else:
            self.mass_indices = list(range(1, 4))  # Default to indices 1, 2, and 3 if mass is None or not specified

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        np.random.seed(42)

    def _on_rollout_start(self) -> None:
        """
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: If the callback returns False, training is aborted early.
        """
        if self.locals.get("done"):
            for i in self.mass_indices:
                min_value = self.original_masses[i] - self.range_value if self.original_masses[i] - self.range_value > 0 else 0.5
                max_value = self.original_masses[i] + self.range_value
                self.training_env.envs[0].sim.model.body_mass[i] = np.random.uniform(min_value, max_value)
            if self.verbose > 0:
                print(self.training_env.envs[0].sim.model.body_mass)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

