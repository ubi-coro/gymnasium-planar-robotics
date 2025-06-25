__version__ = '1.1.0'

from gymnasium.envs.registration import register

from gymnasium_planar_robotics.envs.basic_envs import (
    BasicPlanarRoboticsSingleAgentEnv,
    BasicPlanarRoboticsMultiAgentEnv,
)
from gymnasium_planar_robotics.utils.rendering import Matplotlib2DViewer, MujocoViewerCollection
from gymnasium_planar_robotics.utils.impedance_control import MoverImpedanceController

__all__ = [
    'BasicPlanarRoboticsMultiAgentEnv',
    'BasicPlanarRoboticsSingleAgentEnv',
    'Matplotlib2DViewer',
    'MoverImpedanceController',
    'MujocoViewerCollection',
]


def register_gymnasium_envs():
    ################
    # Planning     #
    ################
    register(
        id='BenchmarkPlanningEnv-v0',
        entry_point='gymnasium_planar_robotics.envs.planning.benchmark_planning_env:BenchmarkPlanningEnv',
        max_episode_steps=50,
    )

    ################
    # Manipulation #
    ################
    register(
        id='BenchmarkPushingEnv-v0',
        entry_point='gymnasium_planar_robotics.envs.manipulation.benchmark_pushing_env:BenchmarkPushingEnv',
        max_episode_steps=50,
    )


register_gymnasium_envs()
