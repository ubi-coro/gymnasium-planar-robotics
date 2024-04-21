.. Gymnasium-Planar-Robotics documentation master file, created by
   sphinx-quickstart on Tue Apr 16 22:11:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gymnasium-Planar-Robotics
=========================
`Gymnasium-Planar-Robotics (GPR) <https://github.com/ubi-coro/gymnasium-planar-robotics/>`_ is a collection of reinforcement learning environments for planar robotics based on the `MuJoCo <https://mujoco.org/>`_ physics engine.
This library contains environments for two main kinds of problems: object manipulation and path planning.

Paper: 

All environments follow the `Gymnasium <https://gymnasium.farama.org/>`_ API:

.. code-block:: python

   import gymnasium as gym

   env = gym.make("BenchmarkPushingEnv", render_mode="human")
   observation, info = env.reset(seed=42)

   for _ in range(0,100):
      while not terminated and not truncated:
         action = policy(observation)  # custom policy
         observation, reward, terminated, truncated, info = env.step(action)

      observation, info = env.reset()
   env.close()

Citing Gymnasium-Planar-Robotics
--------------------------------

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   overview
   planning_envs
   manipulation_envs
   make_own_env
   changelog