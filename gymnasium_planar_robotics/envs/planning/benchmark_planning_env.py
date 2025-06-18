##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

"""The ``BenchmarkPlanningEnv`` is a simple motion planning environment that should be understood as an example of how motion planning
with planar motor systems can look like. This environment is therefore intended for parameter or algorithm tests.

The aim is to learn to move all movers from random (x,y) start positions to variable (x,y) goal positions without collisions
by specifying either the jerk or the acceleration. In this environment, positions, velocities, accelerations, and jerks have
the units m, m/s, m/s² and m/s³, respectively.

Observation Space
-----------------

The observation space of this environment is a dictionary containing the following keys and values:

+---------------+------------------------------------------------------------------------------------------------------------+
|     key       |                                          value                                                             |
+===============+============================================================================================================+
| observation   | - if ``learn_jerk=True``:                                                                                  |
|               |   a numpy array of shape (num_movers*2*2,) containing the (x,y)-velocities and (x,y)-accelerations of each |
|               |   mover ((x,y)-velo mover 1, (x,y)-velo mover 2, ..., (x,y)-acc mover 1, (x,y)-acc mover 2, ...)           |
|               | - if ``learn_jerk=False``:                                                                                 |
|               |   a numpy array of shape (num_movers*2,) containing the (x,y)-velocities and of each mover                 |
|               |   ((x,y)-velo mover 1, (x,y)-velo mover 2, ...)                                                            |
+---------------+------------------------------------------------------------------------------------------------------------+
| achieved_goal | a numpy array of shape (num_movers*2,) containing the current (x,y)-positions of all movers w.r.t. the     |
|               | frame ((x,y)-pos mover 1, (x,y)-pos mover 2, ...)                                                          |
+---------------+------------------------------------------------------------------------------------------------------------+
| desired_goal  | a numpy array of shape (num_movers*2,) containing the desired (x,y)-positions of all movers w.r.t the      |
|               | base frame ((x,y) goal pos mover 1, (x,y) goal pos mover 2, ...)                                           |
+---------------+------------------------------------------------------------------------------------------------------------+

Action Space
------------

The action space is continuous. If ``learn_jerk=True``, an action

.. math::
    a_j := [j_{1x}, j_{1y}, ..., j_{nx}, j_{ny}]

represents the desired jerks for each mover in x and y direction of the base frame (unit: m/s³), where

.. math::
    j_{1x}, j_{1y}, ..., j_{nx}, j_{ny} \in [-j_{max},j_{max}]

``j_max`` is the maximum possible jerk (see environment parameters) and n denotes the number of movers.

Accordingly, if ``learn_jerk=False``, an action

.. math::
    a_a := [a_{1x}, a_{1y}, ..., a_{nx}, a_{ny}]

represents the accelerations for each mover in x and y direction of the base frame (unit: m/s²), where

.. math::
    a_{1x}, a_{1y}, ..., a_{nx}, a_{ny} \in [-a_{max},a_{max}]

``a_max`` is the maximum possible acceleration (see environment parameters) and n denotes the number of movers.

Immediate Rewards
-----------------

The agent receives a reward of 50 if all movers reach their goals without collisions. In case of a collision either with another mover
or with a wall, the agent receives a reward of -50. For each timestep in which at least one mover has not reached its goal and in which
there is no collision, the environment emits the following immediate reward:
number of movers that have not reached their goals * (-1)

Episode Termination
-------------------

Each episode has a time limit of 50 environment steps. If the time limit is reached, the episode is truncated. Thus, each episode has
50 environment steps, except that all movers have reached their goals or there has been a collision. In these cases, the episode
terminates immediately, regardless of the time limit.

Version History
---------------

- v0: initial version of the environment

Parameters
----------

"""

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import logger

from gymnasium_planar_robotics import BasicPlanarRoboticsSingleAgentEnv, Matplotlib2DViewer
from gymnasium_planar_robotics.utils import mujoco_utils


class BenchmarkPlanningEnv(BasicPlanarRoboticsSingleAgentEnv):
    """A simple planning environment.

    :param layout_tiles: a numpy array of shape (num_tiles_x, num_tiles_y) indicating where to add a tile (use 1 to add a tile
        and 0 to leave cell empty). The x-axis and y-axis correspond to the axes of the numpy array, so the origin of the base
        frame is in the upper left corner.
    :param num_movers: the number of movers to add
    :param show_2D_plot: whether to show a 2D matplotlib plot (useful for debugging)
    :param mover_colors_2D_plot: a list of matplotlib colors, one for each mover (only used if ``show_2D_plot=True``), defaults to
        None. None is only accepted if ``show_2D_plot = False``.
    :param tile_params: a dictionary that can be used to specify the mass and size of a tile using the keys 'mass' or 'size',
        defaults to None. Since one planar motor system usually only contains tiles of one type, i.e. with the same mass and size,
        the mass is a single float value and the size must be specified as a numpy array of shape (3,). If set to None or only one
        key is specified, both mass and size or the missing value are set to the following default values:

        - mass: 5.6 [kg]
        - size: [0.24/2, 0.24/2, 0.0352/2] (x,y,z) [m] (note: half-size)
    :param mover_params: a dictionary that can be used to specify the mass and size of each mover using the keys 'mass' or 'size',
        defaults to None. To use the same mass and size for each mover, the mass can be specified as a single float value and the
        size as a numpy array of shape (3,). However, the movers can also be of different types, i.e. different masses and sizes.
        In this case, the mass and size should be specified as numpy arrays of shapes (num_movers,) and (num_movers,3),
        respectively. If set to None or only one key is specified, both mass and size or the missing value are set to the following
        default values:

        - mass: 1.24 [kg]
        - size: [0.155/2, 0.155/2, 0.012/2] (x,y,z) [m] (note: half-size)
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.003 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle: whether to call 'render' after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after 'num_cycles' have been executed if 'render_mode != None'.
    :param num_cycles: the number of control cycles for which to apply the same action, defaults to 40
    :param collision_params: a dictionary that can be used to specify the following collision parameters, defaults to None:

        - collision shape (key: 'shape'): can be 'box' or 'circle', defaults to 'circle'
        - size of the collision shape (key: 'size'), defaults to 0.11 [m]:

            - collision shape 'circle':
                a single float value which corresponds to the radius of the circle, or a numpy array of shape (num_movers,) to specify
                individual values for each mover
            - collision shape 'box':
                a numpy array of shape (2,) to specify x and y half-size of the box, or a numpy array of shape (num_movers, 2) to
                specify individual sizes for each mover

        - additional size offset (key: 'offset'), defaults to 0.0 [m]: an additional safety offset that is added to the size of the
            collision shape. Think of this offset as increasing the size of a mover by a safety margin.
        - additional wall offset (key: 'offset_wall'), defaults to 0.0 [m]: an additional safety offset that is added to the size
            of the collision shape to detect wall collisions. Think of this offset as moving the wall, i.e. the edge of a tile
            without an adjacent tile, closer to the center of the tile.

    :param v_max: the maximum velocity, defaults to 2.0 [m/s]
    :param a_max: the maximum acceleration, defaults to 10.0 [m/s²]
    :param j_max: the maximum jerk (only used if ``learn_jerk=True``), defaults to 100.0 [m/s³]
    :param learn_jerk: whether to learn the jerk, defaults to False. If set to False, the acceleration is learned, i.e. the policy
        output.
    :param threshold_pos: the position threshold used to determine whether a mover has reached its goal position, defaults
        to 0.1 [m]
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2.0,
        a_max: float = 10.0,
        j_max: float = 100.0,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        self.learn_jerk = learn_jerk

        # cam config
        default_cam_config = {
            'distance': 1.1,
            'azimuth': 90.0,
            'elevation': -65.0,
            'lookat': np.array([0.44, 0.18, 0.067]),
        }

        super().__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
            tile_params=tile_params,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=default_cam_config,
            render_every_cycle=render_every_cycle,
            num_cycles=num_cycles,
            collision_params=collision_params,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        # maximum velocity, acceleration and jerk
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max

        # position threshold in m
        self.threshold_pos = threshold_pos
        # reward in case of success
        self.reward_success = 50
        # whether to show a 2D matplotlib plot
        self.show_2D_plot = show_2D_plot
        if self.show_2D_plot and mover_colors_2D_plot is None:
            raise ValueError('Please specify the colors of the movers for the 2D plot.')

        # remember actuator names
        self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(
            self.model, obj_type='actuator', name_pattern='mover_actuator_x'
        )
        self.mover_actuator_y_names = mujoco_utils.get_mujoco_type_names(
            self.model, obj_type='actuator', name_pattern='mover_actuator_y'
        )

        # observation space
        # observation:
        #   - velocities in x and y direction of each mover
        #   - accelerations in x and y direction of each mover if learn_jerk = True
        # achieved_goal:
        #   the current (x,y)-position of each mover
        # desired_goal:
        #   the (x,y) goal position of each mover
        low_goals = np.zeros((self.num_movers * 2,))
        high_goals = np.array(
            [np.max(self.x_pos_tiles) + (self.tile_size[0] / 2), np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)] * self.num_movers
        )
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_movers * (1 + int(self.learn_jerk)) * 2,), dtype=np.float64
                ),
                'achieved_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
                'desired_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
            }
        )

        # action space
        as_low = -self.j_max if self.learn_jerk else -self.a_max
        as_high = self.j_max if self.learn_jerk else self.a_max
        self.action_space = gym.spaces.Box(low=as_low, high=as_high, shape=(self.num_movers * 2,), dtype='float64')

        # minimum and maximum possible mover (x,y)-positions
        safety_margin = self.c_size + self.c_size_offset_wall + self.c_size_offset
        self.min_xy_pos = np.zeros(2) + safety_margin
        self.max_xy_pos = (
            np.array([np.max(self.x_pos_tiles) + (self.tile_size[0] / 2), np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)])
            - safety_margin
        )

        # minimum distance between any two goals
        if self.c_shape == 'circle':
            self.min_goal_dist = 2 * (self.c_size + self.c_size_offset)
        else:
            # self.c_shape == 'box'
            self.min_goal_dist = 2 * np.linalg.norm(self.c_size + self.c_size_offset, ord=2)

        # 2D plot
        if self.show_2D_plot:
            self.matplotlib_2D_viewer = Matplotlib2DViewer(
                layout_tiles=layout_tiles,
                num_movers=self.num_movers,
                mover_sizes=self.mover_size,
                mover_colors=mover_colors_2D_plot,
                tile_size=self.tile_size,
                x_pos_tiles=self.x_pos_tiles,
                y_pos_tiles=self.y_pos_tiles,
                c_shape=self.c_shape,
                c_size=self.c_size,
                c_size_offset=self.c_size_offset,
                arrow_scale=0.2,
                figure_size=(9, 9),
            )

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict | None) -> dict[str, str]:
        """For each mover, this callback adds actuators to the ``custom_model_xml_strings``-dict, depending on whether the jerk or
        acceleration is the output of the policy.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback
        :return: the modified the current ``custom_model_xml_strings``-dict
        """
        mover_actuator_xml_str = '\n\n\t<actuator>' + '\n\t\t<!-- mover actuators -->'
        for idx_mover in range(0, self.num_movers):
            joint_name = f'mover_joint_{idx_mover}'
            mover_mass = self.mover_mass if isinstance(self.mover_mass, float) else self.mover_mass[idx_mover]

            if self.learn_jerk:
                mover_actuator_xml_str += (
                    f'\n\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="integrator" '
                    + f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none" actearly="true"/>'
                    + f'\n\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                    + f'dyntype="integrator" gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none" actearly="true"/>'
                    + '\n'
                )
            else:
                # learn acceleration
                mover_actuator_xml_str += (
                    f'\n\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="none" '
                    + f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>'
                    + f'\n\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" dyntype="none" '
                    + f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>'
                    + '\n'
                )

        mover_actuator_xml_str += '\t</actuator>'

        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
        custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
        if custom_outworldbody_xml_str is not None:
            custom_outworldbody_xml_str += mover_actuator_xml_str
        else:
            custom_outworldbody_xml_str = mover_actuator_xml_str
        custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray, mover_goal_xy_pos: np.ndarray) -> None:
        """Generate a new model xml string with new start and goal positions and reload the model. In this environment, it is necessary
        to reload the model to ensure that the actuators work as expected.

        :param mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover.
        :param mover_goal_xy_pos: a numpy array of shape (num_movers_with_goals,2) containing the (x,y) goal positions of the
            movers (num_movers_with_goals <= num_movers)
        """
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=mover_goal_xy_pos, custom_xml_strings=custom_model_xml_strings
        )
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the start and goal positions of all movers and reload the model. It is also checked whether the start positions are
        collision-free (mover and wall collisions) and whether the new goals can be reached without mover or wall collisions.

        :param options: not used in this environment
        """
        # sample new mover start positions
        start_qpos = np.zeros((self.num_movers, 7))
        start_qpos[:, 2] = self.initial_mover_zpos
        start_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        # ensure that the start positions are chosen such that there no wall or mover collisions
        counter = 0
        all_start_pos_valid = False
        while not all_start_pos_valid:
            counter += 1
            if counter > 0 and counter % 100 == 0:
                logger.warn(
                    'Trying to find a collision-free configuration of start positions for all movers. '
                    + f'No valid configuration found within {counter} trails. Consider choosing fewer movers or more tiles.'
                )

            start_qpos[:, :2] = self.np_random.uniform(low=self.min_xy_pos, high=self.max_xy_pos, size=(self.num_movers, 2))
            # check wall collision
            pos_is_valid = self.qpos_is_valid(qpos=start_qpos, c_size=self.c_size, add_safety_offset=True)
            # check mover collision
            mover_collision = self.check_mover_collision(
                mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=True, mover_qpos=start_qpos
            )
            if not mover_collision and pos_is_valid.all():
                all_start_pos_valid = True

        # sample new goal positions
        goal_qpos = np.zeros((self.num_movers, 7))
        goal_qpos[:, 2] = self.initial_mover_zpos
        goal_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        # ensure that all goal positions can be reached without wall and mover collisions
        counter = 0
        all_goal_pos_reachable, dist_goals_valid = False, False
        while not all_goal_pos_reachable or not dist_goals_valid:
            counter += 1
            if counter > 0 and counter % 100 == 0:
                logger.warn(
                    'Trying to find valid goal positions for all movers. '
                    + f'No valid configuration found within {counter} trails. Consider choosing fewer movers or more tiles.'
                )
            goal_qpos[:, :2] = self.np_random.uniform(low=self.min_xy_pos, high=self.max_xy_pos, size=(self.num_movers, 2))
            goal_pos_reachable = self.qpos_is_valid(qpos=goal_qpos, c_size=self.c_size, add_safety_offset=True)
            all_goal_pos_reachable = np.sum(goal_pos_reachable) == self.num_movers
            if all_goal_pos_reachable:
                dist_goals_valid = True
                for i in range(0, self.num_movers):
                    for j in range(i + 1, self.num_movers):
                        if np.linalg.norm(goal_qpos[i, :2] - goal_qpos[j, :2], ord=2) < self.min_goal_dist:
                            dist_goals_valid = False
                            break
                    if not dist_goals_valid:
                        break

        self.goals = goal_qpos[:, :2].copy()

        # reload model with new start pos and goal pos
        self.reload_model(mover_start_xy_pos=start_qpos[:, :2], mover_goal_xy_pos=self.goals)

    def _mujoco_step_callback(self, action: np.ndarray) -> None:
        """Apply the next action, i.e. it sets the jerk or acceleration, ensuring the minimum and maximum velocity and acceleration
        (for one cycle).

        :param action: a numpy array of shape (num_movers * 2,), which specifies the next action (jerk or acceleration)
        """
        action = action.reshape((self.num_movers, 2))

        for idx_mover in range(0, self.num_movers):
            mover_name = self.mover_names[idx_mover]
            vel = self.get_mover_qvel(mover_name=mover_name, add_noise=True)[:2]

            if self.learn_jerk:
                acc = self.get_mover_qacc(mover_name=mover_name, add_noise=False)[:2]
                next_acc_tmp, next_jerk = self.ensure_max_dyn_val(
                    current_values=acc, max_value=self.a_max, next_derivs=action[idx_mover, :]
                )
                _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=next_acc_tmp)
                if (next_acc_tmp != next_acc).any():
                    next_jerk = (next_acc - acc) / self.cycle_time
                ctrl = next_jerk.copy()
            else:
                _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=action[idx_mover, :])
                ctrl = next_acc.copy()
            mujoco_utils.set_actuator_ctrl(
                model=self.model, data=self.data, actuator_name=self.mover_actuator_x_names[idx_mover], value=ctrl[0, 0]
            )

            mujoco_utils.set_actuator_ctrl(
                model=self.model, data=self.data, actuator_name=self.mover_actuator_y_names[idx_mover], value=ctrl[0, 1]
            )

    def _render_callback(self) -> None:
        """Update the Matplotlib2DViewer if ``show_2D_plot=True``."""
        if self.show_2D_plot:
            mover_qpos = self.get_mover_qpos_arr(mover_names=self.mover_names, add_noise=False)
            mover_qvel = self.get_mover_qvel_arr(mover_names=self.mover_names, add_noise=False)
            self.matplotlib_2D_viewer.render(mover_qpos=mover_qpos, mover_qvel=mover_qvel, mover_goals=self.goals)

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. A state is terminal when there is a collision between two movers or between a
        mover and a wall or when all movers have reached their goals without collisions.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,). An entry is True if the state is terminal, False otherwise
            - if batch_size = 1:
                True if the state is terminal, False otherwise
        """
        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        terminated = np.bitwise_or(reward == self.reward_success, reward == -self.reward_success)
        return terminated

    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. The truncation condition (a time limit in this environment) is
        automatically checked by the Gymnasium TimeLimit Wrapper, which is why this method always returns False.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,) in which all entries are False
            - if batch_size = 1:
                False
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        return np.array([False] * batch_size) if batch_size > 1 else False

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | float:
        """Compute the immediate reward.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        batch_size, mover_collisions, wall_collisions = self._preprocess_info_dict(info=info)
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        mask_collision = np.bitwise_or(mover_collisions, wall_collisions)
        mask_no_collision = np.bitwise_not(mask_collision)

        dist_goal = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal)
        assert dist_goal.shape == (batch_size, self.num_movers)
        goal_reached = dist_goal <= self.threshold_pos
        num_goals_reached = np.sum(goal_reached, axis=1)
        assert num_goals_reached.shape == (batch_size,)
        mask_all_goals_reached = num_goals_reached == self.num_movers

        reward = -self.reward_success * mask_collision.astype(np.float64)
        reward += -1.0 * (self.num_movers - num_goals_reached) * mask_no_collision.astype(np.float64)
        reward[np.bitwise_and(mask_all_goals_reached, mask_no_collision)] = self.reward_success

        assert reward.shape == (batch_size,)
        return reward if batch_size > 1 else reward[0]

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a dictionary containing the following keys and values:
            - 'observation':

                - if ``learn_jerk=True``: a numpy array of shape (num_movers*2*2,) containing the (x,y)-velocities and
                                          (x,y)-accelerations of each mover
                                          ((x,y)-velo mover 1, (x,y)-velo mover 2, ..., (x,y)-acc mover 1, (x,y)-acc mover 2, ...)
                - if ``learn_jerk=False``: a numpy array of shape (num_movers*2,) containing the (x,y)-velocities and of each mover
                                           ((x,y)-velo mover 1, (x,y)-velo mover 2, ...)
            - 'achieved_goal':
                a numpy array of shape (num_movers*2,) containing the current (x,y)-positions of all movers
                ((x,y)-pos mover 1, (x,y)-pos mover 2, ...)
            - 'desired_goal':
                a numpy array of shape (num_movers*2,) containing the desired (x,y)-positions of all movers
                ((x,y) goal pos mover 1, (x,y) goal pos mover 2, ...)
        """
        mover_xy_pos = self.get_mover_qpos_arr(mover_names=self.mover_names, add_noise=True)[:, :2]
        mover_xy_velos = self.get_mover_qvel_arr(mover_names=self.mover_names, add_noise=True)[:, :2]
        if self.learn_jerk:
            # no noise, because only SetAcc is available in a real system
            mover_xy_accs = self.get_mover_qacc_arr(mover_names=self.mover_names, add_noise=False)[:, :2]

        observation = np.concatenate((mover_xy_velos, mover_xy_accs), axis=0) if self.learn_jerk else mover_xy_velos.copy()
        achieved_goal = mover_xy_pos.copy()
        desired_goal = self.goals.copy()

        return OrderedDict(
            [
                # (x,y)-velo mover 1, (x,y)-velo mover 2, ..., (x,y)-acc mover 1, (x,y)-acc mover 2, ...
                ('observation', observation.flatten()),
                # (x,y)-pos mover 1, (x,y)-pos mover 2, ...
                ('achieved_goal', achieved_goal.flatten()),
                # (x,y) goal pos mover 1, (x,y) goal pos mover 2, ...
                ('desired_goal', desired_goal.flatten()),
            ]
        )

    def _get_info(
        self, mover_collision: bool, wall_collision: bool, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> dict[str, Any]:
        """Return a dictionary that contains auxiliary information.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param achieved_goal: a numpy array of shape (length achieved_goal,) containing the (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (length achieved_goal,) containing the desired (x,y)-positions
        :return: the info dictionary with keys 'is_success', 'mover_collision' and 'wall_collision'
        """
        dist = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal).flatten()
        is_success = np.sum(dist <= self.threshold_pos) == self.num_movers and not mover_collision and not wall_collision
        assert not isinstance(is_success, np.ndarray)
        assert not isinstance(mover_collision, np.ndarray)
        assert not isinstance(wall_collision, np.ndarray)
        info = {'is_success': is_success, 'mover_collision': mover_collision, 'wall_collision': wall_collision}
        return info

    def close(self) -> None:
        """Close the environment."""
        super().close()
        if self.show_2D_plot:
            self.matplotlib_2D_viewer.close()

    def ensure_max_dyn_val(
        self, current_values: np.ndarray, max_value: float, next_derivs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure the minimum and maximum dynamic values.

        :param current_values: the current velocity or acceleration specified as a numpy array of shape (2,) or
            (num_checks,2)
        :param max_value: the maximum velocity or acceleration (float)
        :param next_derivs: the next derivative (acceleration or jerk) used for one integrator step specified as a numpy array of
            shape (2,) or (num_checks,2)
        :return: the next velocity or acceleration and the next derivative (acceleration or jerk) corresponding to the next action
            that must be applied to ensure the minimum and maximum dynamics (each of shape (num_checks,2))
        """
        if len(current_values.shape) == 1:
            current_values = current_values.reshape((1, -1))
        if len(next_derivs.shape) == 1:
            next_derivs = next_derivs.reshape((1, -1))

        next_values = np.zeros((current_values.shape[0], 2))
        next_derivs_new = np.zeros((current_values.shape[0], 2))

        next_values_tmp = self.cycle_time * next_derivs + current_values

        norm_next_values_tmp = np.linalg.norm(next_values_tmp, ord=2, axis=1)
        mask_norm = norm_next_values_tmp >= max_value

        next_values[np.bitwise_not(mask_norm), :] = next_values_tmp[np.bitwise_not(mask_norm), :]
        next_derivs_new[np.bitwise_not(mask_norm), :] = next_derivs[np.bitwise_not(mask_norm), :]

        if mask_norm.any():
            next_values[mask_norm] = max_value * np.divide(
                next_values_tmp[mask_norm], np.tile(norm_next_values_tmp[mask_norm], reps=(1, 2))
            )
            next_derivs_new[mask_norm] = (next_values[mask_norm] - current_values[mask_norm]) / self.cycle_time

        return next_values, next_derivs_new

    def _calc_eucl_dist_xy(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        """Calculate the Euclidean distance.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :return: a numpy array of shape (batch_size,), which contains the distances between the achieved and the desired goals
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        achieved_goal_tmp = achieved_goal.reshape((batch_size, self.num_movers, 2))
        desired_goal_tmp = desired_goal.reshape((batch_size, self.num_movers, 2))

        return np.linalg.norm(achieved_goal_tmp - desired_goal_tmp, ord=2, axis=2)

    def _preprocess_info_dict(self, info: np.ndarray | dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray]:
        """Extract information about mover collisions, wall collisions and the batch size from the info dictionary.

        :param info: the info dictionary or an array of info dictionary to be preprocessed. All dictionaries must contain the keys
            'mover_collision' and 'wall_collision'.
        :return: the batch_size (int), a numpy array of shape (batch_size,) containing the mover collision values (bool),
            a numpy array of shape (batch_size,) containing the wall collision values (bool)
        """
        if isinstance(info, np.ndarray):
            batch_size = info.shape[0]
            mover_collisions = np.zeros(batch_size).astype(bool)
            wall_collisions = np.zeros(batch_size).astype(bool)

            for i in range(0, batch_size):
                mover_collisions[i] = info[i]['mover_collision']
                wall_collisions[i] = info[i]['wall_collision']
        else:
            assert isinstance(info, dict)
            batch_size = 1
            mover_collisions = np.array([info['mover_collision']])
            wall_collisions = np.array([info['wall_collision']])

        return batch_size, mover_collisions, wall_collisions
