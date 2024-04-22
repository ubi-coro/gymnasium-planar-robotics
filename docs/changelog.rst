Changelog
=========

Pre-Release v0.0.1a3
--------------------

General
^^^^^^^
1.  Added the ``BenchmarkPushingEnv`` and ``BenchmarkPlanningEnv`` environments: two simple example environments 
    for motion planning and object manipulation.
2.  Added a ``MoverImpedanceController`` that solves a position and orientation task.
3.  Added the ``MujocoOffScreenViewer``: an extension of the Gymnasium OffScreenViewer that allows to also specify the groups 
    of geoms to be rendered by the off-screen renderer.
4.  Added the ``MujocoViewerCollection``: a manager for all renderers in a MuJoCo environment.
5.  Added the ``Matplotlib2DViewer``: a simple viewer that displays the tile and mover configuration together with the mover 
    collision offsets for debugging and analyzing planning tasks.
6.  Added auxiliary functions for MuJoCo, collision checking and rotations
