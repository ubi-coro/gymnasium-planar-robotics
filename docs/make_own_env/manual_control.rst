Manual Controller
=================

The `ManualControl` utility in Gymnasium-Planar-Robotics allows controlling movers manually using the keyboard.
This feature is useful for debugging and testing mover behavior and for creating custom trajectories.

**Note:** This utility is currently only supported for the 2D viewer.

**Prerequisites:**
  - The 2D viewer must be enabled in the environment
  - Manual control is inactive by default and can be toggled with a key press

Usage
-----
After creating an instance of an environment with the 2D viewer enabled, the handle ``manual_controller``
of :class:`Matplotlib2DViewer` should be used.

When passing an action to the environment's :func:`step` function, the manually induced kinematics can be
utilized using :func:`get_action_manual` or :func:`overwrite_action`.

Key Bindings
------------
+-----------+-----------------------------------------------+
| Key       | Action                                        |
+===========+===============================================+
| `C`       | Toggle manual control mode                    |
+-----------+-----------------------------------------------+
| `M`       | Switch to the next mover                      |
+-----------+-----------------------------------------------+
| `Arrow ↑` | Move the controlled mover upward              |
+-----------+-----------------------------------------------+
| `Arrow ↓` | Move the controlled mover downward            |
+-----------+-----------------------------------------------+
| `Arrow ←` | Move the controlled mover leftward            |
+-----------+-----------------------------------------------+
| `Arrow →` | Move the controlled mover rightward           |
+-----------+-----------------------------------------------+

A legend for the available keystrokes is displayed next to the simulation view. The currently controlled mover
is highlighted with a steering wheel icon.

Code Example
------------
The code below shows a minimal example for a manually controlled mover in a planning environment with random
mover kinematics.

.. code-block:: python

  # run.py
  
  import numpy as np
  from gymnasium_planar_robotics.envs.planning.benchmark_planning_env import BenchmarkPlanningEnv

  # create the environment with the 2D viewer enabled
  env = BenchmarkPlanningEnv(
    layout_tiles=np.ones((4, 5)),
    num_movers=3,
    show_2D_plot=True,
    mover_colors_2D_plot=["red", "blue", "green"],
    render_mode="human",
    render_every_cycle=False,
    num_cycles=4,
  )
  manual_controller = env.matplotlib_2D_viewer.manual_controller  # get manual controller handle

  env.reset(seed=42)  # initially reset environment

  terminated = False
  while not terminated:
    # get random action for all movers
    action = env.action_space.sample()

    # overwrite action for first mover (first two elements)
    action = manual_controller.overwrite_action(action)
    
    # pass action to environment
    observation, reward, terminated, truncated, info = env.step(action)

  env.close()   # close environment

.. autoclass:: gymnasium_planar_robotics.utils.rendering.ManualControl
  :members:
  :inherited-members:
  :noindex:
