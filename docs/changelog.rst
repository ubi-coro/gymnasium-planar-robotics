Changelog
=========

Pre-Release v0.0.1a9
--------------------

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