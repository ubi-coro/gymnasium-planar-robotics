__version__ = '0.0.1a6'

from gymnasium.envs.registration import register


def register_gymnasium_envs():
    ################
    # Planning     #
    ################
    register(
        id='BenchmarkPlanningEnv',
        entry_point='gymnasium_planar_robotics.envs.planning.benchmark_planning_env:BenchmarkPlanningEnv',
        max_episode_steps=50,
    )

    ################
    # Manipulation #
    ################
    register(
        id='BenchmarkPushingEnv',
        entry_point='gymnasium_planar_robotics.envs.manipulation.benchmark_pushing_env:BenchmarkPushingEnv',
        max_episode_steps=50,
    )
