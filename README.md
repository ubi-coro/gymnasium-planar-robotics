# Gymnasium-Planar-Robotics (GymPR)
This library contains reinforcement learning environments for motion planning and object manipulation in the field of planar robotics. The environments follow either the [Gymnasium](https://gymnasium.farama.org/) API for single-agent RL or the [PettingZoo parallel API](https://pettingzoo.farama.org/api/parallel/) for multi-agent RL. All environments are based on the [MuJoCo](https://mujoco.org/) physics engine. Note that this library depends on the latest MuJoCo Python bindings. 
[mujoco-py](https://github.com/openai/mujoco-py) is not supported.

<img src="https://github.com/ubi-coro/gymnasium-planar-robotics/raw/main/docs/images/visual_abstract.png" />

## Installation
The Gymnasium-Planar-Robotics package can be installed via PIP:
```
pip install gymnasium-planar-robotics
```
To install optional dependencies, to build the documentation, or to run the tests, use:
```
pip install gymnasium-planar-robotics[docs, tests]
```
**Note:** Depending on your shell (e.g. when using Zsh), you may need to use additional quotation marks: 
```
pip install "gymnasium-planar-robotics[docs, tests]"
```

## Documentation
The documentation is available at: [https://ubi-coro.github.io/gymnasium-planar-robotics/](https://ubi-coro.github.io/gymnasium-planar-robotics/)

## License
GymPR is published under the GNU General Public License v3.0.

## Example
The following example shows how to use a trained policy with an example environment that follows the Gymnasium API:

```python
import gymnasium as gym

env = gym.make("BenchmarkPushingEnv-v0", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(0,100):
    while not terminated and not truncated:
        action = policy(observation)  # custom policy
        observation, reward, terminated, truncated, info = env.step(action)

    observation, info = env.reset()
env.close()
```

## Maintainer
Gymnasium-Planar-Robotics is currently maintained by Lara Bergmann (@lbergmann1).
