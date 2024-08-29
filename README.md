
## Files Structure
The majority of the folders, except for Actor-Critic and Reinforce contain 3 files that generally have the same code, with only small variations tailored to a specific task.
- **`train_sb3.py`**: The main script for training and testing RL agents. It includes various configurable options via command-line arguments.
- **`env/custom_hopper.py`**: Contains the definition of the custom environment "Custom Hopper".
- **`UDR.py`**: Defines the callback for domain randomization.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- Gym
- Stable Baselines3 (`PPO`, `SAC`)
- PyTorch
- TensorBoard

## train_sb3.py arguments 
1.**`--train-timesteps`**
   - Description: Specifies the number of timesteps for which the agent should be trained.
   - Type: int
   - Default value: None (if not specified)

2.**`--outName`**
   - Description: The output filename for saving the trained model. If not specified, a name will be generated automatically based on other parameters.
   - Type: str
   - Default value: None

3.**`--model`**
   - Description: Specifies the file from which to load a pre-trained model. It can be a specific file or values like best (for the best model) or final (for the final model).
   - Type: str
   - Default value: None

4.**`--test-episodes`**
   - Description: Number of episodes to use for testing the model. This parameter is useful for evaluating the performance of the trained model.
   - Type: int
   - Default value: 2000

5.**`--algorithm`**
   - Description: Specifies the reinforcement learning algorithm to use for training. Available options are PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic).
   - Type: str
   - Default value: PPO

6.**`--render`**
   - Description: If specified, enables rendering of the simulation during the testing phase, allowing you to visualize the agent's execution.
   - Type: bool
   - Default value: False

7.**`--device`**
   - Description: Specifies the device on which to run the training. It can be cpu for running on the processor or cuda for running on a GPU.
   - Type: str
   - Default value: cpu

8.**`--DR`**
   - Description: Type of Domain Randomization to apply during training. Currently supports UDR (Uniform Domain Randomization) or None if no randomization is desired.
   - Type: str
   - Default value: None

9.**`--range`**
   - Description: Specifies the range for Uniform Domain Randomization (UDR). This parameter is used to define the variation range applied to the environment parameters during training.
   - Type: float
   - Default value: 5

10.**`--mass`**
    - Description: Specifies the mass to use for domain randomization. It determines the mass value that varies the environment's dynamics during training.
    - Type: int
    - Default value: None

