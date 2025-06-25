##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

"""The ``BenchmarkPushingEnv`` is a simple pushing environment that should be understood as an example of how object manipulation
with planar motor systems can look like. This environment is therefore intended for parameter or algorithm tests.

The aim is to push an object with the mover to the desired goal position without collisions between a mover and a wall by
specifying either the jerk or the acceleration of the mover. The starting positions of the mover and object as well as the
object goal position are chosen randomly at the start of a new episode. This environment contains only one object and one
mover. In addition, the tile layout is set to a 3x3 layout. In this environment, positions, velocities, accelerations, and jerks have
the units m, m/s, m/s² and m/s³, respectively.

Observation Space
-----------------

The observation space of this environment is a dictionary containing the following keys and values:

+---------------+------------------------------------------------------------------------------------------------------------+
|     key       |                                          value                                                             |
+===============+============================================================================================================+
| observation   | - if ``learn_jerk=True``:                                                                                  |
|               |   a numpy array of shape (2*2,) containing the (x,y)-position, (x,y)-velocities and (x,y)-accelerations    |
|               |   of the mover                                                                                             |
|               | - if ``learn_jerk=False``:                                                                                 |
|               |   a numpy array of shape (2,) containing the (x,y)-position and (x,y)-velocities of the mover              |
+---------------+------------------------------------------------------------------------------------------------------------+
| achieved_goal | a numpy array of shape (2,) containing the current (x,y)-position of the object w.r.t. the base frame      |
+---------------+------------------------------------------------------------------------------------------------------------+
| desired_goal  | a numpy array of shape (2,) containing the desired (x,y)-position of the object w.r.t. the base frame      |
+---------------+------------------------------------------------------------------------------------------------------------+

Action Space
------------

The action space is continuous and 2-dimensional. If ``learn_jerk=True``, an action

.. math::
    a_j = [a_{jx}, a_{jy}] \in [-j_{max},j_{max}]²

represents the desired jerks in x and y direction of the base frame (unit: m/s³), where ``j_max`` is the maximum possible jerk (see
environment parameters).

Accordingly, if ``learn_jerk=False``, an action

.. math::
    a_a = [a_{ax}, a_{ay}] \in [-a_{max},a_{max}]²

represents the accelerations in x and y direction of the base frame (unit: m/s²), where ``a_max`` is the maximum possible acceleration
(see environment parameters).

Immediate Rewards
-----------------

The agent receives a reward of 0 if the object has reached its goal without a collision between the mover and a wall.
For each timestep in which the object has not reached its goal and in which there is no collision between the mover and a wall, the
environment emits a small negative reward of -1.
In the case of a collision between the mover and a wall, the agent receives a large negative reward of -50.

Episode Termination and Truncation
----------------------------------

Each episode has a time limit of 50 environment steps. If the time limit is reached, the episode is truncated. Thus, each episode
has 50 environment steps, except that the mover collides with a wall. In this case, the episode terminates immediately
regardless of the time limit. An episode is not terminated when the object reaches its goal position, as the object has to remain close
to the desired goal position until the end of an episode.

.. note::
    The term 'episode steps' refers to the number of calls of env.steps(). The number of MuJoCo simulation steps, i.e. control
    cycles, is a multiple of the number of episode steps.

Environment Reset
-----------------
When the environment is reset, new (x,y) starting positions for the mover and the object as well as a new goal position are chosen at
random. It is ensured that the new start position of the mover is collision-free, i.e. no wall collision and no collision with the
object. In addition, the object's start position is chosen such that the mover fits between the wall and the object. This is important
to ensure that the object can be pushed in all directions.

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

from gymnasium_planar_robotics import BasicPlanarRoboticsSingleAgentEnv, MoverImpedanceController
from gymnasium_planar_robotics.utils import mujoco_utils


class BenchmarkPushingEnv(BasicPlanarRoboticsSingleAgentEnv):
    """A simple object pushing environment.

    :param mover_params: a dictionary that can be used to specify the mass and size of each mover using the keys 'mass' or 'size',
        defaults to None. To use the same mass and size for each mover, the mass can be specified as a single float value and the
        size as a numpy array of shape (3,). However, the movers can also be of different types, i.e. different masses and sizes.
        In this case, the mass and size should be specified as numpy arrays of shapes (num_movers,) and (num_movers,3),
        respectively. If set to None or only one key is specified, both mass and size or the missing value are set to the following
        default values:

        - mass: 1.24 [kg]
        - size: [0.155/2, 0.155/2, 0.012/2] (x,y,z) [m] (note: half-size)
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.003
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle: whether to call 'render' after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after 'num_cycles' have been executed if ``render_mode != None``.
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
    :param j_max: the maximum jerk (only used if 'learn_jerk=True'), defaults to 100.0 [m/s³]
    :param learn_jerk: whether to learn the jerk, defaults to False. If set to False, the acceleration is learned, i.e. the policy
        output.
    :param threshold_pos: the position threshold used to determine whether a mover has reached its goal position, defaults
        to 0.05 [m]
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
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
        threshold_pos: float = 0.05,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        self.learn_jerk = learn_jerk

        # object parameters, object type: box
        self.object_length_xy = 0.07 / 2  # [m] (half-size)
        self.object_height = 0.04 / 2  # [m] (half-size)
        self.object_mass = 0.01  # [kg]
        self.object_xy_start_pos = np.array([0.12, 0.36])
        self.object_xy_goal_pos = np.array([0.36, 0.36])
        self.object_noise_xy_pos = 1e-5

        # there is only one mover in this environment -> remember index
        self.idx_mover = 0

        # impedance controller
        self.impedance_controller = None

        # cam config
        default_cam_config = {
            'distance': 0.8,
            'azimuth': 160.0,
            'elevation': -55.0,
            'lookat': np.array([0.8, 0.2, 0.4]),
        }

        super().__init__(
            layout_tiles=np.ones((3, 3)),
            num_movers=1,
            tile_params=None,
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
        # reward for a collision between the mover and a wall
        self.reward_wall_collision = -50

        # remember object joint name
        self.object_joint_name = mujoco_utils.get_mujoco_type_names(self.model, obj_type='joint', name_pattern='object')[0]

        # observation space
        # observation:
        #   - x and y position of the mover
        #   - velocities in x and y direction of the mover
        #   - accelerations in x and y direction of the mover if learn_jerk = True
        # achieved_goal:
        #   the current (x,y)-position of the object
        # desired_goal:
        #   the (x,y) goal position of the object
        low_goals = np.zeros(2)
        high_goals = np.array([np.max(self.x_pos_tiles) + (self.tile_size[0] / 2), np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)])
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_movers * (2 + int(self.learn_jerk)) * 2,), dtype=np.float64
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
        # minimum and maximum possible object (x,y)-positions
        self.object_min_xy_pos = self.min_xy_pos + safety_margin
        self.object_max_xy_pos = self.max_xy_pos - safety_margin

        # impedance contoller
        self.impedance_controller = MoverImpedanceController(
            model=self.model,
            mover_joint_name=self.mover_joint_names[0],
            joint_mask=np.array([0, 0, 1, 1, 1, 1]),
            translational_stiffness=1.0,
            rotational_stiffness=0.1,
        )
        self.reload_model()

        # remember actuator names
        self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(
            self.model, obj_type='actuator', name_pattern='mover_actuator_x'
        )
        self.mover_actuator_y_names = mujoco_utils.get_mujoco_type_names(
            self.model, obj_type='actuator', name_pattern='mover_actuator_y'
        )

        # minimum distance between object and mover after env reset
        if self.c_shape == 'circle':
            self.min_mo_dist = max(
                np.linalg.norm(self.object_length_xy + self.mover_size.flatten()[:2], ord=2), self.c_size + self.c_size_offset
            )
        else:
            # self.c_shape == 'box'
            self.min_mo_dist = max(
                np.linalg.norm(self.object_length_xy + self.mover_size.flatten()[:2], ord=2),
                np.linalg.norm(self.c_size + self.c_size_offset, ord=2),
            )

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict | None) -> dict[str, str]:
        """For each mover, this callback adds actuators to the ``custom_model_xml_strings``-dict, depending on whether the jerk or
        acceleration is the output of the policy.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback
        :return: the modified the current ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
        # actuators
        if self.impedance_controller is not None:
            mover_actuator_xml_str = '\n\n\t<actuator>' + '\n\t\t<!-- mover actuators -->'
            joint_name = self.mover_joint_names[self.idx_mover]
            if self.learn_jerk:
                mover_actuator_xml_str += (
                    f'\n\t\t<general name="mover_actuator_x_{self.idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" '
                    + f'dyntype="integrator" gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none" actearly="true"/>'
                    + f'\n\t\t<general name="mover_actuator_y_{self.idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                    + f'dyntype="integrator" gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none" actearly="true"/>'
                )
            else:
                # learn acceleration
                mover_actuator_xml_str += (
                    f'\n\t\t<general name="mover_actuator_x_{self.idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="none" '
                    + f'gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none"/>'
                    + f'\n\t\t<general name="mover_actuator_y_{self.idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                    + f'dyntype="none" gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none"/>'
                )

            mover_actuator_xml_str += self.impedance_controller.generate_actuator_xml_string(idx_mover=self.idx_mover)
            mover_actuator_xml_str += '\n'

            mover_actuator_xml_str += '\t</actuator>'

            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            if custom_outworldbody_xml_str is not None:
                custom_outworldbody_xml_str += mover_actuator_xml_str
            else:
                custom_outworldbody_xml_str = mover_actuator_xml_str
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        # object
        custom_object_xml_str = (
            '\n\t\t<!-- object -->'
            + f'\n\t\t<body name="object" pos="{self.object_xy_start_pos[0]} {self.object_xy_start_pos[1]} {self.object_height}">'
            + '\n\t\t\t<joint name="object_joint" type="free" damping="0.01"/>'
            + '\n\t\t\t<geom name="object_geom" '
            + f'type="box" size="{self.object_length_xy} {self.object_length_xy} {self.object_height}" '
            + f'mass="{self.object_mass}" material="red"/>'
            + '\n\t\t</body>'
            + '\n\t\t<site name="object_goal_site" type="sphere" material="red" size="0.02" '
            + f'pos="{self.object_xy_goal_pos[0]} {self.object_xy_goal_pos[1]} {self.object_height}"/>'
        )

        custom_worldbody_xml_str = custom_model_xml_strings.get('custom_worldbody_xml_str', None)
        if custom_worldbody_xml_str is not None:
            custom_worldbody_xml_str += custom_object_xml_str
        else:
            custom_worldbody_xml_str = custom_object_xml_str
        custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray | None = None) -> None:
        """Generate a new model xml string with new start positions for mover and object and a new object goal position and reload the
        model. In this environment, it is necessary to reload the model to ensure that the actuators work as expected.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover,
            defaults to None. If set to None, the movers will be placed in the center of the tiles that are added to the xml string
            first.
        """
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=None, custom_xml_strings=custom_model_xml_strings
        )
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the start position of mover and object and the object goal position and reload the model. It is ensured that the
        new start position of the mover is collision-free, i.e. no wall collision and no collision with the object.
        In addition, the object's start position is chosen such that the mover fits between the wall and the object. This is important
        to ensure that the object can be pushed in all directions.

        :param options: not used in this environment
        """
        # sample new mover start positions
        start_qpos = np.zeros((self.num_movers, 7))
        start_qpos[:, 2] = self.initial_mover_zpos
        start_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        # choose a new start position for the mover
        start_qpos[:, :2] = self.np_random.uniform(low=self.min_xy_pos, high=self.max_xy_pos, size=(self.num_movers, 2))

        # sample a new start position for the object and ensure that it does not collide with the mover
        counter = 0
        dist_start_valid = False
        while not dist_start_valid:
            counter += 1
            if counter > 0 and counter % 100 == 0:
                logger.warn(
                    'Trying to find a start position for the object.'
                    + f'No valid configuration found within {counter} trails. Consider choosing more tiles.'
                )
            self.object_xy_start_pos = self.np_random.uniform(
                low=self.object_min_xy_pos, high=self.object_max_xy_pos, size=(self.num_movers, 2)
            )
            # check distance between object and mover
            dist_start_valid = (np.linalg.norm(self.object_xy_start_pos - start_qpos[:, :2], ord=2, axis=1) > self.min_mo_dist).all()

        self.object_xy_start_pos = self.object_xy_start_pos.flatten()

        # sample a new goal position for the object
        self.object_xy_goal_pos = self.np_random.uniform(low=self.object_min_xy_pos, high=self.object_max_xy_pos, size=(2,))

        # reload model with new start pos and goal pos
        self.reload_model(mover_start_xy_pos=start_qpos[:, :2])

    def _mujoco_step_callback(self, action: np.ndarray) -> None:
        """Apply the next action, i.e. it sets the jerk or acceleration, ensuring the minimum and maximum velocity and acceleration
        (for one cycle).

        :param action: a numpy array of shape (num_movers * 2,), which specifies the next action (jerk or acceleration)
        """
        action = action.reshape((self.num_movers, 2))

        mover_name = self.mover_names[self.idx_mover]
        vel = self.get_mover_qvel(mover_name=mover_name, add_noise=True)[:2]

        if self.learn_jerk:
            acc = self.get_mover_qacc(mover_name=mover_name, add_noise=False)[:2]
            next_acc_tmp, next_jerk = self.ensure_max_dyn_val(
                current_values=acc, max_value=self.a_max, next_derivs=action[self.idx_mover, :]
            )
            _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=next_acc_tmp)
            if (next_acc_tmp != next_acc).any():
                next_jerk = (next_acc - acc) / self.cycle_time
            ctrl = next_jerk.copy()
        else:
            _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=action[self.idx_mover, :])
            ctrl = next_acc.copy()
        mujoco_utils.set_actuator_ctrl(
            model=self.model, data=self.data, actuator_name=self.mover_actuator_x_names[self.idx_mover], value=ctrl[0, 0]
        )

        mujoco_utils.set_actuator_ctrl(
            model=self.model, data=self.data, actuator_name=self.mover_actuator_y_names[self.idx_mover], value=ctrl[0, 1]
        )
        # update impedance controller
        self.impedance_controller.update(
            model=self.model,
            data=self.data,
            pos_d=np.array([0, 0, self.initial_mover_zpos + self.mover_size[2]]),
            quat_d=np.array([1, 0, 0, 0]),
        )

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. A state is terminal when the mover collides with a wall.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,). An entry is True if the state is terminal, False otherwise
            - if batch_size = 1:
                True if the state is terminal, False otherwise
        """
        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        terminated = reward == self.reward_wall_collision
        return terminated

    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. The truncation condition (a time limit in this environment) is
        automatically checked by the Gymnasium TimeLimit Wrapper, which is why this method always returns False.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
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
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        batch_size, _, wall_collisions = self._preprocess_info_dict(info=info)
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        mask_no_collision = np.bitwise_not(wall_collisions)

        dist_goal = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal)
        assert dist_goal.shape == (batch_size,)
        mask_goal_reached = dist_goal <= self.threshold_pos

        reward = self.reward_wall_collision * wall_collisions.astype(np.float64)
        reward += -1.0 * mask_no_collision.astype(np.float64)
        reward[np.bitwise_and(mask_goal_reached, mask_no_collision)] = 0

        assert reward.shape == (batch_size,)
        return reward if batch_size > 1 else reward[0]

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a dictionary containing the following keys and values:

            - 'observation':

                - if ``learn_jerk=True``:
                    a numpy array of shape (2*2,) containing the (x,y)-position, (x,y)-velocities and (x,y)-accelerations of the mover
                - if ``learn_jerk=False``:
                    a numpy array of shape (2,) containing the (x,y)-position and (x,y)-velocities and of the mover
            - 'achieved_goal':
                a numpy array of shape (2,) containing the current (x,y)-position of the object
            - 'desired_goal':
                a numpy array of shape (2,) containing the desired (x,y)-position of the object
        """
        # observation
        mover_xy_pos = np.zeros((self.num_movers, 2))
        mover_xy_velos = np.zeros((self.num_movers, 2))
        if self.learn_jerk:
            mover_xy_accs = np.zeros((self.num_movers, 2))

        mover_name = self.mover_names[self.idx_mover]
        mover_xy_pos[self.idx_mover, :] = self.get_mover_qpos(mover_name=mover_name, add_noise=True)[:2]
        mover_xy_velos[self.idx_mover, :] = self.get_mover_qvel(mover_name=mover_name, add_noise=True)[:2]
        if self.learn_jerk:
            # no noise, because only SetAcc is available in a real system
            mover_xy_accs[self.idx_mover, :] = self.get_mover_qacc(mover_name=mover_name, add_noise=False)[:2]

        if self.learn_jerk:
            observation = np.concatenate((mover_xy_pos, mover_xy_velos, mover_xy_accs), axis=0)
        else:
            observation = np.concatenate((mover_xy_pos, mover_xy_velos), axis=0)

        # achieved goal
        object_xy_pos = mujoco_utils.get_joint_qpos(self.model, self.data, self.object_joint_name)[:2]
        achieved_goal = object_xy_pos + self.rng_noise.normal(loc=0.0, scale=self.object_noise_xy_pos, size=2)

        # desired goal
        desired_goal = self.object_xy_goal_pos.copy()

        return OrderedDict(
            [
                ('observation', observation.flatten()),
                ('achieved_goal', achieved_goal),
                ('desired_goal', desired_goal),
            ]
        )

    def _get_info(
        self, mover_collision: bool, wall_collision: bool, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> dict[str, Any]:
        """Return a dictionary that contains auxiliary information.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param achieved_goal: a numpy array of shape (length achieved_goal,) containing the already achieved (x,y)-position of the
            object
        :param desired_goal: a numpy array of shape (length achieved_goal,) containing the desired (x,y)-position of the object
        :return: the info dictionary with keys 'is_success', 'mover_collision' and 'wall_collision'
        """
        assert not mover_collision
        dist = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal).flatten()
        assert dist.shape == (1,)
        is_success = (dist <= self.threshold_pos)[0] and not wall_collision
        assert not isinstance(is_success, np.ndarray)
        assert not isinstance(mover_collision, np.ndarray)
        assert not isinstance(wall_collision, np.ndarray)
        info = {'is_success': is_success, 'mover_collision': mover_collision, 'wall_collision': wall_collision}
        return info

    def close(self) -> None:
        """Close the environment."""
        super().close()

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
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :return: a numpy array of shape (batch_size,), which contains the distances between the achieved and the desired goals
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        return np.linalg.norm(achieved_goal - desired_goal, ord=2, axis=1)

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
