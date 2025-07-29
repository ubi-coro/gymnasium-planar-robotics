Changelog
=========

Release v1.1.1a2 (2025-07-29)
-----------------------------

General
^^^^^^^^^
1. ``BasicPlanarRoboticsEnv``: Add a parameter to specify the tile sliding friction :spelling:ignore:`(@cedricgrothues)`


Release v1.1.1a1 (2025-06-27)
-----------------------------

General
^^^^^^^^^
1. ``BasicPlanarRoboticsEnv``: Add an parameter to specify the scale for a mover mesh separately from the size option :spelling:ignore:`(@cedricgrothues)`


Release v1.1.1a0 (2025-06-27)
-----------------------------

General
^^^^^^^
1. ``BenchmarkPushingEnv``: add the option to set a specific start position of the mover and a specific goal position of the object when resetting the environment

Release v1.1.0 (2025-06-17)
---------------------------

General
^^^^^^^
1. Add different mover types (cylinder, mesh) :spelling:ignore:`(@cedricgrothues)`

Release v1.0.4 (2025-06-12)
---------------------------

General
^^^^^^^
1. Update dependencies (now supports ``gymnasium>=0.29.1``)
2. Update supported Python versions (3.11, 3.12, 3.13)

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
