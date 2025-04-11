Manual Controller
=================

Gymnasium-Planar-Robotics allows controlling movers manually using the keyboard. A legend for the available
keystrokes is displayed next to the simulation view.

This function is currently only supported for the 2D viewer.

When passing an action to :func:`step` function of the environment, the manually induced kinematics can be
utilized using :func:`get_action_manual` or :func:`overwrite_action`.

.. autoclass:: gymnasium_planar_robotics.utils.rendering.ManualControl
  :members:
  :inherited-members:
  :noindex:
