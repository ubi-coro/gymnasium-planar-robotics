Changelog
=========

Release v1.0.4 (2025-04-21)
---------------------------

General
^^^^^^^
1. Add the ``ManualControl`` utility that enables users to manually control movers in the ``Matplotlib2DViewer`` using keyboard inputs.

Release v1.0.3 (2025-01-14)
---------------------------

General
^^^^^^^
1. Add grid lines to visualize tile boundaries :spelling:ignore:`(@cedricgrothues)`

Release v1.0.2 (2024-07-30)
---------------------------

Bug Fixes
^^^^^^^^^
1. Fixed a bug in ``BenchmarkPlanningEnv`` (mover actuator XML string) where different mover masses were not taken into account


Release v1.0.1 (2024-05-28)
---------------------------

General
^^^^^^^
1.  Added the basic environments: ``BasicPlanarRoboticsEnv``, ``BasicPlanarRoboticsSingleAgentEnv``, ``BasicPlanarRoboticsMultiAgentEnv``
2.  Added the ``BenchmarkPushingEnv`` and ``BenchmarkPlanningEnv`` environments: two simple example environments 
    for motion planning and object manipulation.
3.  Added a ``MoverImpedanceController`` that solves a position and orientation task.
4.  Added the ``MujocoOffScreenViewer``: an extension of the Gymnasium OffScreenViewer that allows to also specify the groups 
    of geoms to be rendered by the off-screen renderer.
5.  Added the ``MujocoViewerCollection``: a manager for all renderers in a MuJoCo environment.
6.  Added the ``Matplotlib2DViewer``: a simple viewer that displays the tile and mover configuration together with the mover 
    collision offsets for debugging and analyzing planning tasks.
7.  Added auxiliary functions for MuJoCo, collision checking and rotations