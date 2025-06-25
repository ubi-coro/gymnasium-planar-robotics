##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import logger
from pettingzoo import ParallelEnv

from gymnasium_planar_robotics.utils import geometry_2D_utils, mujoco_utils, rendering

INVALID_MOVER_SHAPE_ERROR = "Invalid mover shape. Supported shapes are: 'box', 'cylinder', 'mesh'"


class BasicPlanarRoboticsEnv:
    """A base class for reinforcement learning environments in the field of planar robotics that is based on MuJoCo.
    Note that MuJoCo does not specify basic physical units (for a more detailed explanation, see
    https://mujoco.readthedocs.io/en/stable/overview.html#units-are-unspecified). Thus, this environment can be used with user-specific
    units. However, note that the units m and kg are used for the default parameters.

    :param layout_tiles: a numpy array of shape (num_tiles_x, num_tiles_y) indicating where to add a tile (use 1 to add a tile
        and 0 to leave cell empty). The x-axis and y-axis correspond to the axes of the numpy array, so the origin of the base
        frame is in the upper left corner.
    :param num_movers: the number of movers to add
    :param tile_params: a dictionary that can be used to specify the mass and size of a tile using the keys 'mass' or 'size',
        defaults to None. Since one planar motor system usually only contains tiles of one type, i.e. with the same mass and size,
        the mass is a single float value and the size must be specified as a numpy array of shape (3,). If set to None or only one
        key is specified, both mass and size or the missing value are set to the following default values:

        - mass: 5.6 [kg]
        - size: [0.24/2, 0.24/2, 0.0352/2] (x,y,z) [m] (note: half-size)
    :param mover_params: Dictionary specifying mover properties. If None, default values are used. Supported keys:

        - mass (float | numpy.ndarray): Mass in kilograms. Options:
            - Single float: Same mass for all movers
            - 1D array (num_movers,): Individual masses per mover

        Default: 1.24 [kg]

        - shape (str | list[str]): Mover shape type. Must be one of:
            - 'box': Rectangular cuboid
            - 'cylinder': Cylindrical shape
            - 'mesh': Custom 3D mesh

            Default: 'box'

        - size (numpy.ndarray): Shape dimensions in meters. Format depends on shape:
            - For 'box': Half-sizes (x, y, z)
            - For 'cylinder': (radius, height, _)
            - For 'mesh': Scale factors (x, y, z)

            Specification options:
            - 1D array (3,): Same size for all movers
            - 2D array (num_movers, 3): Individual sizes per mover

            Default: [0.155/2, 0.155/2, 0.012/2] [m]

        - mesh (dict): Configuration for mesh-based shapes. Required when shape='mesh'. Contains:
            - mover_stl_path (str): Path to mover mesh STL file or one of the predefined meshes:
                - 'beckhoff_apm4330_mover': Beckhoff APM4220 mover mesh (default)
                - 'beckhoff_apm4220_mover': Beckhoff APM4220 mover mesh
                - 'beckhoff_apm4550_mover': Beckhoff APM4550 mover mesh
                - 'planar_motor_M3-06': Planar Motor M3-06 mover mesh
                - 'planar_motor_M3-15': Planar Motor M3-15 mover mesh
                - 'planar_motor_M3-25': Planar Motor M3-25 mover mesh
                - 'planar_motor_M4-11': Planar Motor M4-11 mover mesh
                - 'planar_motor_M4-18': Planar Motor M4-18 mover mesh
            - bumper_stl_path (str | None): Path to bumper mesh STL file or one of the predefined meshes:
                - 'beckhoff_apm4330_bumper': Beckhoff APM4330 bumper mesh (default)
                - 'beckhoff_apm4220_bumper': Beckhoff APM4220 bumper mesh
                - 'beckhoff_apm4550_bumper': Beckhoff APM4550 bumper mesh
            - bumper_mass (float | numpy.ndarray): Bumper mass in kilograms. Can be specified as:
                - Single float: Same mass applied to all bumpers
                - 1D array (num_movers,): Individual masses for each bumper

                Default: 0.1 [kg]

        - material (str | list[str]): Material name to apply to the mover. Can be specified as:
            - Single string: Same material for all movers
            - List of strings: Individual materials for each mover

            Default: "gray" for movers without goals, color-coded materials for movers with goals

        Note: Custom mesh STL files must have their origin at the mover's center.
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.005 [m]
    :param table_height: the height of a table on which the tiles are placed, defaults to 0.4 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param default_cam_config: dictionary with attribute values of the viewer's default camera,
        https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global, defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
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

    :param mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the initial (x,y) starting positions of each mover.
        If set to None, the movers will be placed in the center of a tile, i.e. the number of tiles must be >= the number of
        movers; defaults to None.
    :param mover_goal_xy_pos: a numpy array of shape (num_movers_with_goals,2) containing the initial (x,y) goal positions of the
        movers (num_movers_with_goals <= num_movers). Note that only the first 6 movers have different colors to make the
        movers clearly distinguishable. Movers without goals are shown in gray. If set to None, no goals will be displayed and
        all movers are colored in gray; defaults to None
    :param custom_xml_strings: a dictionary containing additional xml strings to provide the ability to add actuators, sensors,
        objects, robots, etc. to the model. The keys determine where to add a string in the xml structure and the values contain
        the xml string to add. The following keys are accepted:

        - 'custom_compiler_xml_str':
            A custom 'compiler' xml element. Note that the entire default 'compiler' element is replaced.
        - 'custom_visual_xml_str':
            A custom 'visual' xml element. Note that the entire default 'visual' element is replaced.
        - 'custom_option_xml_str':
            A custom 'option' xml element. Note that the entire default 'option' element is replaced.
        - 'custom_assets_xml_str':
            This xml string adds elements to the 'asset' grouping element.
        - 'custom_default_xml_str':
            This xml string adds elements to the 'default' grouping element.
        - 'custom_worldbody_xml_str':
            This xml string adds elements to the 'worldbody' grouping element.
        - 'custom_outworldbody_xml_str':
            This xml string should be used to include files or add elements other than 'compiler', 'visual', 'option', 'asset',
            'default' or 'worldbody'.

        If set to None, only the basic xml string is generated, containing tiles, movers (excluding actuators),
        and possibly goals; defaults to None. This dictionary can be further modified using the ``_custom_xml_string_callback()``.
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.005,
        table_height: float = 0.4,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        default_cam_config: dict[str, Any] | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        collision_params: dict[str, Any] | None = None,
        initial_mover_start_xy_pos: np.ndarray | None = None,
        initial_mover_goal_xy_pos: np.ndarray | None = None,
        custom_model_xml_strings: dict[str, str] | None = None,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        # rng
        self.rng_noise = np.random.default_rng()
        # standard deviation noise
        if isinstance(std_noise, float):
            # use the same standard deviation for position, velocity and acceleration
            self.std_noise = np.array([std_noise, std_noise, std_noise])
        else:
            # use possibly different standard deviations for position, velocity and acceleration
            assert isinstance(std_noise, np.ndarray) and std_noise.shape == (3,), (
                'noise standard deviation has to be a float or a numpy array of shape (3,)'
            )
            self.std_noise = std_noise

        # tile configuration
        self.layout_tiles = layout_tiles.astype(np.int8)
        self.num_tiles = np.sum(self.layout_tiles)
        self.num_tiles_x = self.layout_tiles.shape[0]
        self.num_tiles_y = self.layout_tiles.shape[1]
        if tile_params is None:
            tile_params = {}
        self.tile_size = tile_params.get('size', np.array([0.24 / 2, 0.24 / 2, 0.0352 / 2]))
        self.tile_mass = tile_params.get('mass', 5.6)
        self.x_pos_tiles, self.y_pos_tiles = self.get_tile_xy_pos()
        self._check_tile_config()
        # remember certain indices that belong to specific structures in the tile layout and are important for collision checking
        mask_3x3 = np.ones((3, 3), dtype=np.int8)
        self.idx_x_tiles_3x3, self.idx_y_tiles_3x3 = self.get_tile_indices_mask(mask=mask_3x3)

        mask_2x2_bl = np.array([[1, 1], [0, 1]])
        self.idx_x_tiles_2x2_bl, self.idx_y_tiles_2x2_bl = self.get_tile_indices_mask(mask=mask_2x2_bl)

        mask_2x2_br = np.array([[1, 1], [1, 0]])
        self.idx_x_tiles_2x2_br, self.idx_y_tiles_2x2_br = self.get_tile_indices_mask(mask=mask_2x2_br)

        mask_2x2_tl = np.array([[0, 1], [1, 1]])
        self.idx_x_tiles_2x2_tl, self.idx_y_tiles_2x2_tl = self.get_tile_indices_mask(mask=mask_2x2_tl)

        mask_2x2_tr = np.array([[1, 0], [1, 1]])
        self.idx_x_tiles_2x2_tr, self.idx_y_tiles_2x2_tr = self.get_tile_indices_mask(mask=mask_2x2_tr)
        # padded layout used for wall collision check
        self.layout_tiles_wc = np.pad(layout_tiles, ((0, 1), (0, 1)), mode='constant', constant_values=0)

        # mover configuration
        self.num_movers = num_movers
        self.num_movers_wo_goal = (
            self.num_movers - initial_mover_goal_xy_pos.shape[1] if initial_mover_goal_xy_pos is not None else self.num_movers
        )
        if mover_params is None:
            mover_params = {}
        self.mover_size = mover_params.get('size', np.array([0.155 / 2, 0.155 / 2, 0.012 / 2]))
        self.mover_mass = mover_params.get('mass', 1.24)
        self.mover_shape = mover_params.get('shape', 'box')
        self.mover_material = mover_params.get('material')

        mover_mesh = mover_params.get('mesh', {})
        self.mover_mesh_mover_stl_path = self._resolve_mesh_path(mover_mesh.get('mover_stl_path', 'beckhoff_apm4330_mover'))
        self.mover_mesh_bumper_stl_path = self._resolve_mesh_path(mover_mesh.get('bumper_stl_path', 'beckhoff_apm4330_bumper'))
        self.mover_mesh_bumper_mass = mover_mesh.get('bumper_mass', 0.1)

        self.initial_mover_zpos = initial_mover_zpos
        self._check_mover_config(initial_mover_start_xy_pos, initial_mover_goal_xy_pos)

        self.resolved_mover_size = self._resolve_mover_size(self.mover_size, self.mover_shape)

        # collision detection
        if collision_params is None:
            collision_params = {}
        self.c_shape = collision_params.get('shape', 'circle')
        self.c_size = collision_params.get('size', 0.11)
        self.c_size_offset = collision_params.get('offset', 0.0)
        self.c_size_offset_wall = collision_params.get('offset_wall', 0.0)
        self._check_collision_params()

        self.custom_model_xml_strings_before_cb = custom_model_xml_strings
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings)

        # MuJoCo
        self.table_height = table_height
        # generate model xml string
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=initial_mover_start_xy_pos,
            mover_goal_xy_pos=initial_mover_goal_xy_pos,
            custom_xml_strings=custom_model_xml_strings,
        )

        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data, nstep=1)

        # cycle time
        self.cycle_time = self.model.opt.timestep

        # remember mover names, mover joint names and goal site names (if goals exist)
        self.mover_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='body', name_pattern='mover')
        self.mover_joint_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='joint', name_pattern='mover')
        self.mover_goal_site_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='site', name_pattern='goal_site_mover')
        self._check_mujoco_name_order()

        # rendering
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        if default_cam_config is None:
            default_cam_config = {
                'distance': 2.0,
                'azimuth': 160.0,
                'elevation': -45.0,
                'lookat': np.array([0.7, -0.3, 0.4]),
            }
        # setup viewer collection
        if render_mode is not None:
            self.viewer_collection = rendering.MujocoViewerCollection(
                model=self.model,
                data=self.data,
                default_cam_config=default_cam_config,
                width_no_camera_specified=width_no_camera_specified,
                height_no_camera_specified=height_no_camera_specified,
                use_mj_passive_viewer=use_mj_passive_viewer,
            )

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str] | None:
        """A callback that should be used to add further functionality to the ``__init__()`` method. This callback should be used to
        modify the custom xml string in the ``custom_model_xml_strings`` dictionary after the tile, mover and collision parameters have
        been preprocessed and checked, but before the MuJoCo model xml string is generated. This allows adding custom xml strings based
        on the tile or mover configuration, e.g. to add actuators for each mover.

        :param custom_model_xml_strings: a dictionary containing additional xml strings to provide the ability to add actuators,
            sensors, objects, robots, etc. to the model., defaults to None (see documentation of the  ``__init__()`` method for more
            detailed information). Note that this dictionary may be modified within this method.
        :return: the possibly modified dictionary with additional xml strings
        """
        return custom_model_xml_strings

    ###################################################
    # RL                                              #
    ###################################################

    def render(self) -> np.ndarray | None:
        """Compute frames depending on the initially specified ``render_mode``. Before the corresponding viewer is updated,
        the ``_render_callback()`` is called to give the opportunity to add more functionality.

        :return: returns a numpy array if render_mode != 'human', otherwise it returns None (render_mode 'human')
        """
        self._render_callback()
        if self.render_mode is not None:
            return self.viewer_collection.render(self.render_mode)
        else:
            return None

    def _render_callback(self) -> None:
        """A callback that should be used to add further functionality to the ``render()`` method (see documentation of the
        ``render()`` method for more information about when the callback is called).
        """
        pass

    def close(self) -> None:
        """Close the environment."""
        if self.render_mode is not None:
            self.viewer_collection.close()

    ###################################################
    # Collision and position validation checks        #
    ###################################################
    def check_mover_collision(
        self,
        mover_names: list[str],
        c_size: float | np.ndarray,
        add_safety_offset: bool = False,
        mover_qpos: np.ndarray | None = None,
        add_qpos_noise: bool = False,
    ) -> bool:
        """Check whether two movers specified in ``mover_names`` collide. In case of collision shape 'box', this method takes the
        orientation of the movers into account.

        :param mover_names: a list of mover names that should be checked (correspond to the body name of the mover in
            the MuJoCo model)
        :param c_size: the size of the collision shape of the movers

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_movers,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_movers,2) to
                specify individual sizes for each mover
        :param add_safety_offset: whether to add the size offset (can be specified using: collision_params["offset"]), defaults to
            False. Note that the same size offset is added for both movers.
        :param mover_qpos: the qpos of the movers specified as a numpy array of shape (num_movers,7) (x_p,y_p,z_p,w_o,x_o,y_o,z_o).
            If set to None, the current qpos of the movers in the MuJoCo model is used; defaults to None
        :param add_qpos_noise: whether to add Gaussian noise to the qpos of the movers, defaults to False. Only used if mover_qpos is
            not None.
        :return: True if the movers collide, False otherwise
        """
        if mover_qpos is None:
            mover_qpos = self.get_mover_qpos_arr(mover_names=mover_names, add_noise=add_qpos_noise)

        num_movers = mover_qpos.shape[0]
        assert mover_qpos.shape == (num_movers, 7)

        c_size_arr = self.get_c_size_arr(c_size=c_size + self.c_size_offset * int(add_safety_offset), num_reps=num_movers)

        num_checks = np.sum(np.arange(start=1, stop=num_movers, step=1))
        mover_i_qpos = np.zeros((num_checks, 7))
        mover_j_qpos = np.zeros((num_checks, 7))
        c_size_arr_i = np.zeros((num_checks, c_size_arr.shape[1]))
        c_size_arr_j = np.zeros((num_checks, c_size_arr.shape[1]))

        start_idx = 0
        for i in range(0, num_movers - 1):
            offset_idx = num_movers - (i + 1)
            stop_idx = start_idx + offset_idx
            mover_i_qpos[start_idx:stop_idx, :] = np.repeat(mover_qpos[i : i + 1, :], offset_idx, axis=0)
            mover_j_qpos[start_idx:stop_idx, :] = mover_qpos[i + 1 :, :]
            c_size_arr_i[start_idx:stop_idx, :] = np.repeat(c_size_arr[i : i + 1, :], offset_idx, axis=0)
            c_size_arr_j[start_idx:stop_idx, :] = c_size_arr[i + 1 :, :]
            start_idx = stop_idx

        if self.c_shape == 'circle':
            mover_collision = np.linalg.norm(mover_i_qpos[:, :2] - mover_j_qpos[:, :2], ord=2, axis=1) <= (c_size_arr_i + c_size_arr_j)
        elif self.c_shape == 'box':
            dist = np.linalg.norm(mover_i_qpos[:, :2] - mover_j_qpos[:, :2], ord=2, axis=1)
            max_size = np.max(np.concatenate((c_size_arr_i, c_size_arr_j), axis=1), axis=1)
            diag_size = np.tile(max_size, reps=(2, 1)).T
            mask_add_check = dist <= 2 * np.linalg.norm(diag_size, ord=1, axis=1)
            mover_collision = np.zeros(num_checks)
            if mask_add_check.any():
                mover_collision[mask_add_check] = geometry_2D_utils.check_rectangles_intersect(
                    qpos_r1=mover_i_qpos[mask_add_check, :],
                    qpos_r2=mover_j_qpos[mask_add_check, :],
                    size_r1=c_size_arr_i[mask_add_check, :],
                    size_r2=c_size_arr_j[mask_add_check, :],
                )

        return mover_collision.any()

    def check_wall_collision(
        self,
        mover_names: list[str],
        c_size: float | np.ndarray,
        add_safety_offset: bool = False,
        mover_qpos: np.ndarray | None = None,
        add_qpos_noise: bool = False,
    ) -> np.ndarray:
        """Check whether the qpos of the movers listed in ``mover_names`` are valid, i.e. no wall collisions.

        :param mover_names: a list of mover names that should be checked (correspond to the body name of the mover in
            the MuJoCo model)
        :param c_size: the size of the collision shape

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_movers,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_movers,2) to
                specify individual sizes for each mover
        :param add_safety_offset: whether to add the size offset (can be specified using: collision_params["offset"]), defaults to
            False. Note that the same size offset is added for all movers.
        :param mover_qpos: a numpy array of shape (num_qpos,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) of each mover or None.
            If set to None, the current qpos of each mover in the MuJoCo model is used; defaults to None
        :param add_qpos_noise: whether to add Gaussian noise to the qpos of the movers, defaults to False. Only used if mover_qpos is
            not None.
        :return: a numpy array of shape (num_movers,), where an element is 1 if the qpos is valid (no wall collision), otherwise 0
        """
        if mover_qpos is None:
            mover_qpos = self.get_mover_qpos_arr(mover_names=mover_names, add_noise=add_qpos_noise)

        return 1 - self.qpos_is_valid(mover_qpos, c_size, add_safety_offset)

    def qpos_is_valid(self, qpos: np.ndarray, c_size: float | np.ndarray, add_safety_offset: bool = False) -> np.ndarray:
        """Check whether qpos is valid. This method considers the edges as imaginary walls if there is no other tile next to that
        edge. A position is valid if it is above a tile and the distance to the walls is greater that the required safety margin,
        i.e. no collision with a wall. This also ensures that the position is reachable in case the specified position is a goal
        position.

        This method allows to check multiple qpos at the same time, where the movers can be of different sizes.
        The orientation of the mover is taken into account if collision_shape = 'box', otherwise (collision_shape = 'circle')
        the orientation of the mover is ignored.

        :param qpos: a numpy array of shape (num_qpos,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) to be checked
        :param c_size: the size of the collision shape

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_qpos,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_qpos,2) to
                specify individual sizes for each mover
        :param add_safety_offset: whether to add the size offset (can be specified using: collision_params["offset"]), defaults to
            False. Note that the same size offset is added for all movers.
        :return: a numpy array of shape (num_qpos,), where an element is 1 if the qpos is valid, otherwise 0
        """
        assert len(qpos.shape) == 2
        assert qpos.shape[1] == 7

        # add safety margins
        num_qpos = qpos.shape[0]
        c_size = c_size + self.c_size_offset_wall + int(add_safety_offset) * self.c_size_offset

        # prepare collision size array
        c_size_arr = self.get_c_size_arr(c_size=c_size, num_reps=num_qpos)
        ignore_orientation = False  # self.c_shape == 'box'
        if self.c_shape == 'circle':
            ignore_orientation = True

        # collision shape == 'box': get mover vertices
        if not ignore_orientation:
            mover_vertices = geometry_2D_utils.get_2D_rect_vertices(qpos=qpos, size=c_size_arr)

        # start test
        pos_is_valid = np.zeros(num_qpos, dtype=np.int8)

        # roughly locate the movers -> find the indices of tiles with a mover above them
        x_pos_tiles = np.tile(self.x_pos_tiles, reps=(num_qpos, 1, 1))
        y_pos_tiles = np.tile(self.y_pos_tiles, reps=(num_qpos, 1, 1))
        qpos_x_all = np.tile(qpos[:, 0].reshape((-1, 1, 1)), reps=(1, self.x_pos_tiles.shape[0], self.x_pos_tiles.shape[1]))
        qpos_y_all = np.tile(qpos[:, 1].reshape((-1, 1, 1)), reps=(1, self.y_pos_tiles.shape[0], self.y_pos_tiles.shape[1]))

        mask_above_tile = (
            (x_pos_tiles - self.tile_size[0] <= qpos_x_all)
            * (qpos_x_all <= x_pos_tiles + self.tile_size[0])
            * (y_pos_tiles - self.tile_size[1] <= qpos_y_all)
            * (qpos_y_all <= y_pos_tiles + self.tile_size[1])
        )
        assert np.sum(mask_above_tile) >= num_qpos, (
            'At least one mover is not above a tile. An episode should be terminated in case of wall collision. '
            + 'This error is probably caused by a missed termination of the episode.'
        )
        idx_qpos, idx_tiles_x, idx_tiles_y = np.where(mask_above_tile)
        if not ignore_orientation:
            mover_vertices = mover_vertices[idx_qpos, :, :]
        # min, max x pos of all relevant tiles
        min_x_tiles = self.x_pos_tiles[idx_tiles_x, idx_tiles_y] - self.tile_size[0]
        max_x_tiles = self.x_pos_tiles[idx_tiles_x, idx_tiles_y] + self.tile_size[0]
        # min, max y pos of all relevant tiles
        min_y_tiles = self.y_pos_tiles[idx_tiles_x, idx_tiles_y] - self.tile_size[1]
        max_y_tiles = self.y_pos_tiles[idx_tiles_x, idx_tiles_y] + self.tile_size[1]

        # check whether the tiles are completely surrounded by other tiles
        # mask_complete.shape == (num_qpos,self.idx_x_tiles_3x3.shape[0])
        mask_complete = (
            np.tile(idx_tiles_x, reps=(self.idx_x_tiles_3x3.shape[0], 1)).T
            == np.tile(self.idx_x_tiles_3x3, reps=(idx_tiles_x.shape[0], 1))
        ) * (
            np.tile(idx_tiles_y, reps=(self.idx_y_tiles_3x3.shape[0], 1)).T
            == np.tile(self.idx_y_tiles_3x3, reps=(idx_tiles_y.shape[0], 1))
        )
        idx_qpos_complete = idx_qpos[np.where(mask_complete)[0]]
        pos_is_valid[idx_qpos_complete] = 1
        if np.sum(pos_is_valid) == num_qpos:
            return pos_is_valid

        # at least one pos is above a tile which is not completely surrounded by other tiles
        # (possibly without required safety margin to the edges of the tile)
        # safe = above_tile and all distances to edges > safety margin
        if ignore_orientation:
            rep = 1
            min_x_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                np.tile(min_x_tiles, reps=(rep, 1)).T < qpos[idx_qpos, 0].reshape((-1, 1)) - c_size_arr[idx_qpos, :]
            ).astype(np.int8)
            max_x_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                qpos[idx_qpos, 0].reshape((-1, 1)) + c_size_arr[idx_qpos, :] < np.tile(max_x_tiles, reps=(rep, 1)).T
            ).astype(np.int8)
            min_y_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                np.tile(min_y_tiles, reps=(rep, 1)).T < qpos[idx_qpos, 1].reshape((-1, 1)) - c_size_arr[idx_qpos, :]
            ).astype(np.int8)
            max_y_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                qpos[idx_qpos, 1].reshape((-1, 1)) + c_size_arr[idx_qpos, :] < np.tile(max_y_tiles, reps=(rep, 1)).T
            ).astype(np.int8)
        else:
            rep = 4
            min_x_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                np.tile(min_x_tiles, reps=(rep, 1)).T < mover_vertices[:, 0, :]
            ).astype(np.int8)
            max_x_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                mover_vertices[:, 0, :] < np.tile(max_x_tiles, reps=(rep, 1)).T
            ).astype(np.int8)
            min_y_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                np.tile(min_y_tiles, reps=(rep, 1)).T < mover_vertices[:, 1, :]
            ).astype(np.int8)
            max_y_safe = np.tile(self.layout_tiles[idx_tiles_x, idx_tiles_y], reps=(rep, 1)).T * (
                mover_vertices[:, 1, :] < np.tile(max_y_tiles, reps=(rep, 1)).T
            ).astype(np.int8)

        # mask minimum and maximum indices
        mask_idx_x_lmin = (idx_tiles_x > 0).astype(np.int8)
        mask_idx_y_lmin = (idx_tiles_y > 0).astype(np.int8)
        mask_idx_x_smax = (idx_tiles_x < self.num_tiles_x - 1).astype(np.int8)
        mask_idx_y_smax = (idx_tiles_y < self.num_tiles_y - 1).astype(np.int8)

        mask_valid = (min_x_safe * max_x_safe * min_y_safe * max_y_safe).astype(np.int8)

        # update min_x_safe
        mask_min_x_update = (1 - min_x_safe) * np.tile(
            mask_idx_x_lmin * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y] * self.layout_tiles_wc[idx_tiles_x - 1, idx_tiles_y],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_min_x_update * min_y_safe * max_y_safe
        # update min_y_safe based on min_x_safe-update
        mask_min_x_min_y_update = (1 - min_y_safe) * np.tile(
            mask_idx_x_lmin
            * mask_idx_y_lmin
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y]
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y - 1]
            * self.layout_tiles_wc[idx_tiles_x - 1, idx_tiles_y - 1],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_min_x_update * mask_min_x_min_y_update
        # update max_y_safe based on min_x_safe-update
        mask_min_x_max_y_update = (1 - max_y_safe) * np.tile(
            mask_idx_x_lmin
            * mask_idx_y_smax
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y]
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y + 1]
            * self.layout_tiles_wc[idx_tiles_x - 1, idx_tiles_y + 1],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_min_x_update * mask_min_x_max_y_update

        # update max_x_safe
        mask_max_x_update = (1 - max_x_safe) * np.tile(
            mask_idx_x_smax * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y] * self.layout_tiles_wc[idx_tiles_x + 1, idx_tiles_y],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_max_x_update * min_y_safe * max_y_safe
        # update min_y_safe based on max_x_safe-update
        mask_max_x_min_y_update = (1 - min_y_safe) * np.tile(
            mask_idx_x_smax
            * mask_idx_y_lmin
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y]
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y - 1]
            * self.layout_tiles_wc[idx_tiles_x + 1, idx_tiles_y - 1],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_max_x_update * mask_max_x_min_y_update
        # update max_y_safe based on max_x_safe-update
        mask_max_x_max_y_update = (1 - max_y_safe) * np.tile(
            mask_idx_x_smax
            * mask_idx_y_smax
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y]
            * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y + 1]
            * self.layout_tiles_wc[idx_tiles_x + 1, idx_tiles_y + 1],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_max_x_update * mask_max_x_max_y_update

        # update min_y_safe
        mask_min_y_update = (1 - min_y_safe) * np.tile(
            mask_idx_y_lmin * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y] * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y - 1],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_min_y_update * min_x_safe * max_x_safe

        # update max_y_safe
        mask_max_y_update = (1 - max_y_safe) * np.tile(
            mask_idx_y_smax * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y] * self.layout_tiles_wc[idx_tiles_x, idx_tiles_y + 1],
            reps=(rep, 1),
        ).T
        mask_valid = mask_valid + mask_max_y_update * min_x_safe * max_x_safe

        assert np.bitwise_or(mask_valid == 0, mask_valid == 1).all()

        if ignore_orientation:
            mask_valid = mask_valid.flatten()
        else:
            mask_valid = np.sum(mask_valid, axis=1) == 4

            # bottom left
            mask_2x2_bl = (
                np.tile(mask_valid * mask_idx_x_smax * mask_idx_y_lmin, reps=(self.idx_x_tiles_2x2_bl.shape[0], 1)).T
                * (
                    np.tile(idx_tiles_x, reps=(self.idx_x_tiles_2x2_bl.shape[0], 1)).T
                    == np.tile(self.idx_x_tiles_2x2_bl, reps=(idx_tiles_x.shape[0], 1))
                )
                * (
                    np.tile(idx_tiles_y, reps=(self.idx_y_tiles_2x2_bl.shape[0], 1)).T
                    == np.tile(self.idx_y_tiles_2x2_bl + 1, reps=(idx_tiles_y.shape[0], 1))
                )
            )
            sum_bl = np.sum(mask_2x2_bl, axis=1)
            assert np.bitwise_or(sum_bl == 0, sum_bl == 1).all()
            idx_qpos_bl = idx_qpos[sum_bl == 1]
            if len(idx_qpos_bl) > 0:
                qpos_missing_tiles = np.array([[0, 0, 0, 1, 0, 0, 0]] * idx_qpos_bl.shape[0], dtype=np.float64)
                idx_mask_bl = np.where(mask_2x2_bl)
                qpos_missing_tiles[:, 0] = np.tile(
                    self.x_pos_tiles[self.idx_x_tiles_2x2_bl + 1, self.idx_y_tiles_2x2_bl], reps=(idx_tiles_x.shape[0], 1)
                )[idx_mask_bl]
                qpos_missing_tiles[:, 1] = np.tile(
                    self.y_pos_tiles[self.idx_x_tiles_2x2_bl + 1, self.idx_y_tiles_2x2_bl], reps=(idx_tiles_y.shape[0], 1)
                )[idx_mask_bl]
                mt_intersect = geometry_2D_utils.check_rectangles_intersect(
                    qpos_r1=qpos[idx_qpos_bl, :],
                    qpos_r2=qpos_missing_tiles,
                    size_r1=c_size_arr[idx_qpos_bl, :],
                    size_r2=np.tile(self.tile_size[:2], reps=(idx_qpos_bl.shape[0], 1)),
                )
                mask_valid[idx_mask_bl[0]] = (1 - mt_intersect) * mask_valid[idx_mask_bl[0]]

            # bottom right
            mask_2x2_br = (
                np.tile(mask_valid * mask_idx_x_smax * mask_idx_y_smax, reps=(self.idx_x_tiles_2x2_br.shape[0], 1)).T
                * (
                    np.tile(idx_tiles_x, reps=(self.idx_x_tiles_2x2_br.shape[0], 1)).T
                    == np.tile(self.idx_x_tiles_2x2_br, reps=(idx_tiles_x.shape[0], 1))
                )
                * (
                    np.tile(idx_tiles_y, reps=(self.idx_y_tiles_2x2_br.shape[0], 1)).T
                    == np.tile(self.idx_y_tiles_2x2_br, reps=(idx_tiles_y.shape[0], 1))
                )
            )
            sum_br = np.sum(mask_2x2_br, axis=1)
            assert np.bitwise_or(sum_br == 0, sum_br == 1).all()
            idx_qpos_br = idx_qpos[sum_br == 1]
            if len(idx_qpos_br) > 0:
                qpos_missing_tiles = np.array([[0, 0, 0, 1, 0, 0, 0]] * idx_qpos_br.shape[0], dtype=np.float64)
                idx_mask_br = np.where(mask_2x2_br)
                qpos_missing_tiles[:, 0] = np.tile(
                    self.x_pos_tiles[self.idx_x_tiles_2x2_br + 1, self.idx_y_tiles_2x2_br + 1], reps=(idx_tiles_x.shape[0], 1)
                )[idx_mask_br]
                qpos_missing_tiles[:, 1] = np.tile(
                    self.y_pos_tiles[self.idx_x_tiles_2x2_br + 1, self.idx_y_tiles_2x2_br + 1], reps=(idx_tiles_y.shape[0], 1)
                )[idx_mask_br]
                mt_intersect = geometry_2D_utils.check_rectangles_intersect(
                    qpos_r1=qpos[idx_qpos_br, :],
                    qpos_r2=qpos_missing_tiles,
                    size_r1=c_size_arr[idx_qpos_br, :],
                    size_r2=np.tile(self.tile_size[:2], reps=(idx_qpos_br.shape[0], 1)),
                )
                mask_valid[idx_mask_br[0]] = (1 - mt_intersect) * mask_valid[idx_mask_br[0]]

            # top left
            mask_2x2_tl = (
                np.tile(mask_valid * mask_idx_x_lmin * mask_idx_y_lmin, reps=(self.idx_x_tiles_2x2_tl.shape[0], 1)).T
                * (
                    np.tile(idx_tiles_x, reps=(self.idx_x_tiles_2x2_tl.shape[0], 1)).T
                    == np.tile(self.idx_x_tiles_2x2_tl + 1, reps=(idx_tiles_x.shape[0], 1))
                )
                * (
                    np.tile(idx_tiles_y, reps=(self.idx_y_tiles_2x2_tl.shape[0], 1)).T
                    == np.tile(self.idx_y_tiles_2x2_tl + 1, reps=(idx_tiles_y.shape[0], 1))
                )
            )
            sum_tl = np.sum(mask_2x2_tl, axis=1)
            assert np.bitwise_or(sum_tl == 0, sum_tl == 1).all()
            idx_qpos_tl = idx_qpos[sum_tl == 1]
            if len(idx_qpos_tl) > 0:
                qpos_missing_tiles = np.array([[0, 0, 0, 1, 0, 0, 0]] * idx_qpos_tl.shape[0], dtype=np.float64)
                idx_mask_tl = np.where(mask_2x2_tl)
                qpos_missing_tiles[:, 0] = np.tile(
                    self.x_pos_tiles[self.idx_x_tiles_2x2_tl, self.idx_y_tiles_2x2_tl], reps=(idx_tiles_x.shape[0], 1)
                )[idx_mask_tl]
                qpos_missing_tiles[:, 1] = np.tile(
                    self.y_pos_tiles[self.idx_x_tiles_2x2_tl, self.idx_y_tiles_2x2_tl], reps=(idx_tiles_y.shape[0], 1)
                )[idx_mask_tl]
                mt_intersect = geometry_2D_utils.check_rectangles_intersect(
                    qpos_r1=qpos[idx_qpos_tl, :],
                    qpos_r2=qpos_missing_tiles,
                    size_r1=c_size_arr[idx_qpos_tl, :],
                    size_r2=np.tile(self.tile_size[:2], reps=(idx_qpos_tl.shape[0], 1)),
                )
                mask_valid[idx_mask_tl[0]] = (1 - mt_intersect) * mask_valid[idx_mask_tl[0]]

            # top right
            mask_2x2_tr = (
                np.tile(mask_valid * mask_idx_x_lmin * mask_idx_y_smax, reps=(self.idx_x_tiles_2x2_tr.shape[0], 1)).T
                * (
                    np.tile(idx_tiles_x, reps=(self.idx_x_tiles_2x2_tr.shape[0], 1)).T
                    == np.tile(self.idx_x_tiles_2x2_tr + 1, reps=(idx_tiles_x.shape[0], 1))
                )
                * (
                    np.tile(idx_tiles_y, reps=(self.idx_y_tiles_2x2_tr.shape[0], 1)).T
                    == np.tile(self.idx_y_tiles_2x2_tr, reps=(idx_tiles_y.shape[0], 1))
                )
            )
            sum_tr = np.sum(mask_2x2_tr, axis=1)
            assert np.bitwise_or(sum_tr == 0, sum_tr == 1).all()
            idx_qpos_tr = idx_qpos[sum_tr == 1]
            if len(idx_qpos_tr) > 0:
                qpos_missing_tiles = np.array([[0, 0, 0, 1, 0, 0, 0]] * idx_qpos_tr.shape[0], dtype=np.float64)
                idx_mask_tr = np.where(mask_2x2_tr)
                qpos_missing_tiles[:, 0] = np.tile(
                    self.x_pos_tiles[self.idx_x_tiles_2x2_tr, self.idx_y_tiles_2x2_tr + 1], reps=(idx_tiles_x.shape[0], 1)
                )[idx_mask_tr]
                qpos_missing_tiles[:, 1] = np.tile(
                    self.y_pos_tiles[self.idx_x_tiles_2x2_tr, self.idx_y_tiles_2x2_tr + 1], reps=(idx_tiles_y.shape[0], 1)
                )[idx_mask_tr]
                mt_intersect = geometry_2D_utils.check_rectangles_intersect(
                    qpos_r1=qpos[idx_qpos_tr, :],
                    qpos_r2=qpos_missing_tiles,
                    size_r1=c_size_arr[idx_qpos_tr, :],
                    size_r2=np.tile(self.tile_size[:2], reps=(idx_qpos_tr.shape[0], 1)),
                )
                mask_valid[idx_mask_tr[0]] = (1 - mt_intersect) * mask_valid[idx_mask_tr[0]]

        idx_valid = [idx for idx in np.unique(idx_qpos) if (mask_valid[idx_qpos == idx] == 1).all()]
        pos_is_valid[idx_valid] = 1

        return pos_is_valid.astype(int)

    ###################################################
    # MuJoCo                                          #
    ###################################################

    def window_viewer_is_running(self) -> bool:
        """Check whether the window viewer (render_mode 'human') is active, i.e. the window is open.

        :return: True if the window is open, False otherwise
        """
        return self.viewer_collection.window_viewer_is_running()

    def get_mover_qpos(self, mover_name: str, add_noise: bool = False) -> np.ndarray:
        """Returns the position and orientation of the desired mover. The orientation is returned as a quaternion (w,x,y,z). Note that
        the z-pos is the distance between the bottom of the mover and the top of a tile. In contrast, the z-pos in the MuJoCo model is
        the previously mentioned distance + half the height of a mover.

        :param mover_name: name of the mover for which the position and orientation should be returned (corresponds to the body name
            of the mover in the MuJoCo model)
        :param add_noise: whether to add Gaussian noise, defaults to False

        :return: position and orientation of the desired mover (x_p,y_p,z_p,w_o,x_o,y_o,z_o)
        """
        mover_idx = self.mover_names.index(mover_name)
        joint_name = self.mover_joint_names[mover_idx]
        qpos = mujoco_utils.get_joint_qpos(self.model, self.data, joint_name)

        if isinstance(self.mover_shape, list):
            shape = self.mover_shape[mover_idx]
        else:  # isinstance(self.mover_shape, str)
            shape = self.mover_shape

        if shape == 'box' or shape == 'mesh':  # height at index 2
            qpos[2] -= self.resolved_mover_size[mover_idx, 2]
        elif shape == 'cylinder':  # height at index 1
            qpos[2] -= self.resolved_mover_size[mover_idx, 1]
        else:
            raise ValueError(INVALID_MOVER_SHAPE_ERROR)

        return qpos + self.rng_noise.normal(loc=0.0, scale=self.std_noise[0] * int(add_noise), size=qpos.shape[0])

    def get_mover_qvel(self, mover_name: str, add_noise: bool = False) -> np.ndarray:
        """Return the linear and angular velocities (qvel) of the desired mover.

        :param mover_name: name of the mover for which the velocity should be returned (corresponds to the body name of the mover
            in the MuJoCo model)
        :param add_noise: whether to add Gaussian noise, defaults to False
        :return: linear and angular velocities of the mover (x,y,z,a,b,c)
        """
        mover_idx = self.mover_names.index(mover_name)
        joint_name = self.mover_joint_names[mover_idx]
        qvel = mujoco_utils.get_joint_qvel(self.model, self.data, joint_name)
        return qvel + self.rng_noise.normal(loc=0.0, scale=self.std_noise[1] * int(add_noise), size=qvel.shape[0])

    def get_mover_qacc(self, mover_name: str, add_noise: bool = False) -> np.ndarray:
        """Returns the linear and angular acceleration (qacc) of the desired mover.

        :param mover_name: name of the mover for which the acceleration should be returned (corresponds to the body name of the mover
            in the MuJoCo model)
        :param add_noise: whether to add Gaussian noise, defaults to False

        :return: linear and angular acceleration of the mover (x,y,z,a,b,c)
        """
        mover_idx = self.mover_names.index(mover_name)
        joint_name = self.mover_joint_names[mover_idx]
        qacc = mujoco_utils.get_joint_qacc(self.model, self.data, joint_name)
        return qacc + self.rng_noise.normal(loc=0.0, scale=self.std_noise[2] * int(add_noise), size=qacc.shape[0])

    def _generate_mover_xml_strings(
        self,
        idx_mover: int,
        x_pos: float,
        y_pos: float,
        z_pos: float,
        material: str,
        mass: float,
        size: np.ndarray,
        shape: str,
    ) -> tuple[str | None, str]:
        """Generate MuJoCo XML asset and body strings for creating mover objects in the simulation.

        :return: A tuple containing an XML string for mesh assets (None for basic shapes) and an XML
        string defining the mover body and its properties.
        """
        assert size.shape == (3,), f'Size must have shape (3,), got shape {size.shape}.'

        if shape == 'box':
            asset_str = None
            body_str = (
                f'\n\t\t<body name="mover_{idx_mover}" pos="{x_pos} {y_pos} {z_pos:.5f}" gravcomp="1">'
                + f'\n\t\t\t<joint name="mover_joint_{idx_mover}" type="free" damping="0" />'
                + f'\n\t\t\t<geom name="mover_geom_{idx_mover}" type="box" '
                + f'size="{size[0]} {size[1]} {size[2]}" mass="{mass}" pos="0 0 0" '
                + f'material="{material}"/>'
                + '\n\t\t</body>'
            )
        elif shape == 'cylinder':
            asset_str = None
            body_str = (
                f'\n\t\t<body name="mover_{idx_mover}" pos="{x_pos} {y_pos} {z_pos:.5f}" gravcomp="1">'
                + f'\n\t\t\t<joint name="mover_joint_{idx_mover}" type="free" damping="0" />'
                + f'\n\t\t\t<geom name="mover_geom_{idx_mover}" type="cylinder" '
                + f'size="{size[0]} {size[1]}" mass="{mass}" pos="0 0 0" '
                + f'material="{material}"/>'
                + '\n\t\t</body>'
            )
        elif shape == 'mesh':
            mover_mesh_name = f'mover_mesh_{idx_mover}'
            bumper_mesh_name = f'bumper_mesh_{idx_mover}'

            asset_str = (
                f'\n\t\t<mesh name="{mover_mesh_name}" file="{self.mover_mesh_mover_stl_path}"'
                f' scale="{size[0]} {size[1]} {size[2]}" />'
            )

            body_str = (
                f'\n\t\t<body name="mover_{idx_mover}" pos="{x_pos} {y_pos} {z_pos:.5f}" gravcomp="1">'
                + f'\n\t\t\t<joint name="mover_joint_{idx_mover}" type="free" damping="0" />'
                + f'\n\t\t\t<geom name="mover_geom_{idx_mover}" type="mesh" mesh="{mover_mesh_name}" '
                + f'mass="{mass}" pos="0 0 0" material="{material}"/>'
            )

            if self.mover_mesh_bumper_stl_path is not None:
                asset_str += (
                    f'\n\t\t<mesh name="{bumper_mesh_name}" file="{self.mover_mesh_bumper_stl_path}"'
                    f' scale="{size[0]} {size[1]} {size[2]}" />'
                )

                if isinstance(self.mover_mesh_bumper_mass, np.ndarray):
                    bumper_mass = self.mover_mesh_bumper_mass[idx_mover]
                else:
                    bumper_mass = self.mover_mesh_bumper_mass

                body_str += (
                    f'\n\t\t\t<geom name="bumper_geom_{idx_mover}" type="mesh" mesh="{bumper_mesh_name}" '
                    f'mass="{bumper_mass}" pos="0 0 0" material="black"/>'
                )

            body_str += '\n\t\t</body>'
        else:
            raise ValueError(INVALID_MOVER_SHAPE_ERROR)

        return (asset_str, body_str)

    def generate_model_xml_string(
        self,
        mover_start_xy_pos: np.ndarray | None = None,
        mover_goal_xy_pos: np.ndarray | None = None,
        custom_xml_strings: dict[str, str] | None = None,
    ) -> str:
        """Generate a MuJoCo model xml string based on the mover-tile configuration of the environment.

        :param mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover.
            If set to None, the movers will be placed in the center of a tile, i.e. the number of tiles must be >= the number of
            movers; defaults to None.
        :param mover_goal_xy_pos: a numpy array of shape (num_movers_with_goals,2) containing the (x,y) goal positions of the
            movers (num_movers_with_goals <= num_movers). Note that only the first 6 movers have different colors to make the
            movers clearly distinguishable. Movers without goals are shown in gray. If set to None, no goals will be displayed and
            all movers are colored in gray; defaults to None
        :param custom_xml_strings: a dictionary containing additional xml strings to provide the ability to add actuators, sensors,
            objects, robots, etc. to the model. The keys determine where to add a string in the xml structure and the values contain
            the xml string to add. The following keys are accepted:

            - 'custom_compiler_xml_str':
                A custom 'compiler' xml element. Note that the entire default 'compiler' element is replaced.
            - 'custom_visual_xml_str':
                A custom 'visual' xml element. Note that the entire default 'visual' element is replaced.
            - 'custom_option_xml_str':
                A custom 'option' xml element. Note that the entire default 'option' element is replaced.
            - 'custom_assets_xml_str':
                This xml string adds elements to the 'asset' grouping element.
            - 'custom_default_xml_str':
                This xml string adds elements to the 'default' grouping element.
            - 'custom_worldbody_xml_str':
                This xml string adds elements to the 'worldbody' grouping element.
            - 'custom_outworldbody_xml_str':
                This xml string should be used to include files or add elements other than 'compiler', 'visual', 'option', 'asset',
                'default' or 'worldbody'.

            If set to None, only the basic xml string is generated, containing tiles, movers (excluding actuators),
            and possibly goals; defaults to None
        :return: MuJoCo model xml string
        """
        # prepare mover and tile strings
        if self.num_movers > self.num_tiles and mover_start_xy_pos is None:
            raise ValueError(
                'Number of movers > number of tiles and no start positions specified. Please use more tiles, fewer '
                + 'movers or specify a start position for each mover'
            )
        # tiles
        if custom_xml_strings is None:
            custom_xml_strings = {}
        valid_xy_pos_mover = []  # remember valid mover positions
        tile_xml_str = ''
        for idx_tile_x in range(0, self.num_tiles_x):
            for idx_tile_y in range(0, self.num_tiles_y):
                if self.layout_tiles[idx_tile_x, idx_tile_y]:
                    if mover_start_xy_pos is None:
                        valid_xy_pos_mover.append(
                            np.array([self.x_pos_tiles[idx_tile_x][idx_tile_y], self.y_pos_tiles[idx_tile_x][idx_tile_y]])
                        )
                    tile_xml_str += (
                        f'\n\t\t\t<geom name="tile_{idx_tile_x}_{idx_tile_y}" class="tile" '
                        + f' pos="{self.x_pos_tiles[idx_tile_x][idx_tile_y]} {self.y_pos_tiles[idx_tile_x][idx_tile_y]} 0"/>'
                    )

        tile_xml_str += '\n\n\t\t\t<!-- Lines -->'
        line_height = 0.001 / 2
        line_z_pos = self.tile_size[2] - line_height + 0.00001

        for tile_row in range(self.num_tiles_x):
            for tile_col in range(self.num_tiles_y):
                if not self.layout_tiles[tile_row, tile_col]:
                    continue

                has_top = (tile_row - 1) >= 0 and self.layout_tiles[tile_row - 1, tile_col] == 1
                has_left = (tile_col - 1) >= 0 and self.layout_tiles[tile_row, tile_col - 1] == 1

                if has_top:
                    x_pos = tile_row * self.tile_size[0] * 2
                    y_start = tile_col * self.tile_size[1] * 2
                    y_end = (tile_col + 1) * self.tile_size[1] * 2

                    tile_xml_str += (
                        f'\n\t\t\t<site type="box" size="{line_height}" material="line_mat"'
                        + f' fromto="{x_pos} {y_start} {line_z_pos} {x_pos} {y_end} {line_z_pos}" />'
                    )

                if has_left:
                    x_start = tile_row * self.tile_size[0] * 2
                    x_end = (tile_row + 1) * self.tile_size[0] * 2
                    y_pos = tile_col * self.tile_size[1] * 2

                    tile_xml_str += (
                        f'\n\t\t\t<site type="box" size="{line_height}" material="line_mat"'
                        + f' fromto="{x_start} {y_pos} {line_z_pos} {x_end} {y_pos} {line_z_pos}" />'
                    )

        # movers and correspondig goals and actuators
        material_str_list = ['green', 'blue', 'orange', 'red', 'yellow', 'light_blue']
        mover_asset_xml_strs = ''
        mover_xml_str = ''
        num_goal_movers = mover_goal_xy_pos.shape[0] if mover_goal_xy_pos is not None else 0

        for idx_mover in range(0, self.num_movers):
            if isinstance(self.mover_shape, list):
                mover_shape = self.mover_shape[idx_mover]
            else:
                mover_shape = self.mover_shape

            if mover_shape == 'box' or mover_shape == 'cylinder':
                mover_size = self.resolved_mover_size[idx_mover, :].copy()
            elif mover_shape == 'mesh':
                # We need the scale, not the size, so we can't use resolved_mover_size here.
                mover_size = self.mover_size[idx_mover, :].copy() if len(self.mover_size.shape) == 2 else self.mover_size.copy()
            else:
                raise ValueError(INVALID_MOVER_SHAPE_ERROR)

            if isinstance(self.mover_mass, np.ndarray):
                mover_mass = self.mover_mass[idx_mover]
            else:
                mover_mass = self.mover_mass

            is_obstacle = idx_mover >= num_goal_movers

            if isinstance(self.mover_material, list):
                material_str = self.mover_material[min(idx_mover, len(self.mover_material) - 1)]
            elif isinstance(self.mover_material, str):
                material_str = self.mover_material
            else:
                # choose color
                if is_obstacle:
                    material_str = 'gray'
                else:
                    material_str = material_str_list[min(idx_mover, len(material_str_list) - 1)]

            if mover_shape == 'box' or mover_shape == 'mesh':
                z_pos = self.initial_mover_zpos + self.resolved_mover_size[idx_mover, 2]
            elif mover_shape == 'cylinder':
                z_pos = self.initial_mover_zpos + mover_size[1]
            else:
                raise ValueError(INVALID_MOVER_SHAPE_ERROR)

            if mover_start_xy_pos is None:
                mover_xpos = valid_xy_pos_mover[idx_mover][0]
                mover_ypos = valid_xy_pos_mover[idx_mover][1]
            else:
                mover_xpos = mover_start_xy_pos[idx_mover, 0]
                mover_ypos = mover_start_xy_pos[idx_mover, 1]

            mover_asset_xml_str, mover_body_xml_str = self._generate_mover_xml_strings(
                idx_mover,
                mover_xpos,
                mover_ypos,
                z_pos,
                material_str,
                mover_mass,
                mover_size,
                mover_shape,
            )

            if mover_asset_xml_str is not None:
                mover_asset_xml_strs += mover_asset_xml_str

            mover_xml_str += mover_body_xml_str

            # visualize goal positions
            if mover_goal_xy_pos is not None and not is_obstacle:
                mover_xml_str += (
                    f'\n\t\t<site name="goal_site_mover_{idx_mover}" type="sphere" material="{material_str}" '
                    + f'size="0.02" pos="{mover_goal_xy_pos[idx_mover, 0]} {mover_goal_xy_pos[idx_mover, 1]} '
                    + f'{self.tile_size[2] + 0.002:.5f}"/>'
                )
            mover_xml_str += '\n'

        # prepare custom xml strings
        custom_compiler_xml_str = custom_xml_strings.get('custom_compiler_xml_str', None)
        custom_visual_xml_str = custom_xml_strings.get('custom_visual_xml_str', None)
        custom_option_xml_str = custom_xml_strings.get('custom_option_xml_str', None)
        custom_assets_xml_str = custom_xml_strings.get('custom_assets_xml_str', '')
        custom_default_xml_str = custom_xml_strings.get('custom_default_xml_str', '')
        custom_worldbody_xml_str = custom_xml_strings.get('custom_worldbody_xml_str', '')
        custom_outworldbody_xml_str = custom_xml_strings.get('custom_outworldbody_xml_str', None)

        # complete xml string
        xml = '<?xml version="1.0" encoding="utf-8"?>'
        xml += '\n<mujoco model="planar_robotics">'
        # compiler
        if custom_compiler_xml_str is None:
            xml += f'\n\t<compiler angle="radian" coordinate="local" meshdir="{Path(__file__).parent.resolve() / "assets"}" />'
        else:
            xml += custom_compiler_xml_str
        # visual
        if custom_visual_xml_str is None:
            xml += '\n\t<visual>' + '\n\t\t<scale framelength="0.7" framewidth="0.05"/>' + '\n\t</visual>'
        else:
            xml += custom_visual_xml_str

        # option
        if custom_option_xml_str is None:
            xml += '\n\t<option timestep="0.001" cone="elliptic" jacobian="auto" gravity="0 0 -9.81"/>'
        else:
            xml += custom_option_xml_str

        # assets
        xml += (
            '\n\n\t<asset>'
            + '\n\t\t<material name="white" reflectance="0.01" shininess="0.01" specular="0.1" rgba="1 1 1 1" />'
            + '\n\t\t<material name="off_white" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.7 0.7 0.7 1" />'
            + '\n\t\t<material name="gray" reflectance="1" shininess="1" specular="1" rgba="0.5 0.5 0.5 1"/>'
            + '\n\t\t<material name="black" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.25 0.25 0.25 1" />'
            + '\n\t\t<material name="green" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.2852 0.5078 0.051 1" />'
            + '\n\t\t<material name="red" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.94 0.191 0.191 1" />'
            + '\n\t\t<material name="red_transparent" reflectance="0.01" shininess="0.01" specular="0.1" rgba="1 0 0 0.15" />'
            + '\n\t\t<material name="yellow" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.98 0.94 0.052 1" />'
            + '\n\t\t<material name="orange" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.98 0.39 0 1" />'
            + '\n\t\t<material name="dark_blue" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0 0 1 1" />'
            + '\n\t\t<material name="light_blue" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.492 0.641 0.98 1" />'
            + '\n\t\t<material name="blue" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0. 0.543 0.649 1" />'
            + '\n\t\t<material name="floor_mat" reflectance="0.05" shininess="0.05" specular="0.1" texture="texplane" '
            + 'texuniform="true" />'
            + '\n\t\t<material name="line_mat" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.5 0.5 0.5 1"/>'
            + '\n\t\t<texture name="texplane" builtin="flat" height="256" width="256" rgb1=".8 .8 .8" rgb2=".8 .8 .8" />'
            + '\n\t\t<texture type="skybox" builtin="gradient" rgb1="0.8 0.898 1" rgb2="0.8 0.898 1" width="32" height="32" />'
            + mover_asset_xml_strs
            + custom_assets_xml_str
            + '\n\t</asset>'
        )

        # default
        xml += (
            '\n\n\t<default>'
            + '\n\t\t<default class="planar_robotics">'
            + '\n\t\t\t<default class="tile">'
            + f'\n\t\t\t\t<geom type="box" size="{self.tile_size[0]} {self.tile_size[1]} {self.tile_size[2]}" '
            + f'mass="{self.tile_mass}" material="off_white" />'
            + '\n\t\t\t</default>'
            + '\n\t\t</default>'
            + custom_default_xml_str
            + '\n\t</default>'
        )

        # worldbody
        x_pos_table = (np.max(self.x_pos_tiles) + self.tile_size[0]) / 2
        y_pos_table = (np.max(self.y_pos_tiles) + self.tile_size[1]) / 2
        xml += (
            '\n\n\t<worldbody>'
            + '\n\t\t<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false"'
            + ' pos="0 0 4" dir="0 0 -1" name="light0"/>'
            + f'\n\t\t<geom name="ground_plane" pos="{x_pos_table} {y_pos_table} {-self.tile_size[2] * 2 - self.table_height}" '
            + 'type="plane" size="10 10 10" material="floor_mat"/>'
            + f'\n\t\t<geom name="table" pos="{x_pos_table} {y_pos_table} {-self.tile_size[2] - self.table_height / 2}" size='
            + f'"{(self.num_tiles_x * (self.tile_size[0] * 2) + 0.1) / 2} {(self.num_tiles_y * (self.tile_size[1] * 2) + 0.1) / 2} '
            + f'{self.table_height / 2}" type="box" material="gray" mass="20"/>'
            + '\n\n\t\t<!-- tiles -->'
            + f'\n\t\t<body name="tile_body" childclass="planar_robotics" pos="0 0 {-self.tile_size[2]}" gravcomp="1">'
            + tile_xml_str
            + '\n\t\t</body>'
            + '\n\n\t\t<!-- movers -->'
            + mover_xml_str
            + custom_worldbody_xml_str
            + '\n\t</worldbody>'
        )

        # custom xml str
        if custom_outworldbody_xml_str is not None:
            xml += custom_outworldbody_xml_str

        # end
        xml += '\n</mujoco>'

        return xml

    ###################################################
    # Utils                                           #
    ###################################################

    def get_c_size_arr(self, c_size: float | np.ndarray, num_reps: int) -> np.ndarray:
        """Return the size of the collision shape as a numpy array of shape (num_reps,1) or (num_reps,2) depending on the collision
        shape. This method should be used to obtain the appropriate c_size_arr if the same size is to be used for all movers.

        :param c_size: the size of the collision shape:

            - collision_shape = 'circle':
                use a single float value to specify the same size for all movers and a numpy array of shape (num_reps,) to specify
                individual sizes for each mover
            - collision_shape = 'box':
                use a numpy array of shape (2,) to specify the same size for all movers and a numpy array of shape (num_reps,2) to
                specify individual sizes for each mover
        :param num_reps: the number of repetitions of c_size if the same size of collision shape is to be used for all movers.
            Otherwise, this value is ignored.
        :return: the collision shape sizes as a numpy array of a suitable shape:

            - collision_shape = 'circle':
                a numpy array of shape (num_reps,1)
            - collision_shape = 'box':
                a numpy array of shape (num_reps,2) if c_size is a numpy array of shape (2,). Otherwise, c_size is not modified.
        """
        # prepare collision size array
        if isinstance(c_size, float):
            assert self.c_shape == 'circle', 'Use a float value or a numpy array of shape (num_reps,) to specify the size parameter.'
            c_size_arr = np.tile(np.array([[c_size]]), reps=(num_reps, 1))
        else:
            if self.c_shape == 'circle':
                c_size_arr = c_size.reshape((num_reps, 1))
            elif self.c_shape == 'box' and c_size.shape == (2,):
                c_size_arr = np.tile(c_size, reps=(num_reps, 1))
            else:
                # collision_shape = 'box'
                c_size_arr = c_size.copy()
        return c_size_arr

    def get_mover_qpos_arr(self, mover_names: list[str], add_noise: bool = False) -> np.ndarray:
        """Return the qpos of several movers as a numpy array of shape (num_movers,7).

        :param mover_names: a list of mover names for which the qpos should be returned (correspond to the body name of the mover in
            the MuJoCo model)
        :param add_noise: whether to add Gaussian noise to the qpos of the movers, defaults to False
        :return: a numpy array of shape (num_movers,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) of each mover. The order of
            the qpos corresponds to the order of the mover names.
        """
        num_movers = len(mover_names)
        mover_qpos = np.zeros((num_movers, 7))

        for i, mover_name in enumerate(mover_names):
            mover_qpos[i, :] = self.get_mover_qpos(mover_name, add_noise=add_noise)
        return mover_qpos

    def get_mover_qvel_arr(self, mover_names: list[str], add_noise: bool = False) -> np.ndarray:
        """Return the qvel of several movers as a numpy array of shape (num_movers,6).

        :param mover_names: a list of mover names for which the qvel should be returned (correspond to the body name of the mover in
            the MuJoCo model)
        :param add_noise: whether to add Gaussian noise to the qvel of the movers, defaults to False
        :return: a numpy array of shape (num_movers,6) containing the qvel (x,y,z,a,b,c) of each mover. The order of
            the qvel corresponds to the order of the mover names.
        """
        num_movers = len(mover_names)
        mover_qvel = np.zeros((num_movers, 6))

        for i, mover_name in enumerate(mover_names):
            mover_qvel[i, :] = self.get_mover_qvel(mover_name, add_noise=add_noise)
        return mover_qvel

    def get_mover_qacc_arr(self, mover_names: list[str], add_noise: bool = False) -> np.ndarray:
        """Return the qacc of several movers as a numpy array of shape (num_movers,6).

        :param mover_names: a list of mover names for which the qacc should be returned (correspond to the body name of the mover in
            the MuJoCo model)
        :param add_noise: whether to add Gaussian noise to the qacc of the movers, defaults to False
        :return: a numpy array of shape (num_movers,6) containing the qacc (x,y,z,a,b,c) of each mover. The order of
            the qacc corresponds to the order of the mover names.
        """
        num_movers = len(mover_names)
        mover_qacc = np.zeros((num_movers, 6))

        for i, mover_name in enumerate(mover_names):
            mover_qacc[i, :] = self.get_mover_qacc(mover_name, add_noise=add_noise)
        return mover_qacc

    def get_tile_xy_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """Find the (x,y)-positions of the tiles. The position of a tile in the tile layout with index (i_x,i_y), can be found using
        ``(x-pos[i_x,i_y], y-pos[i_x,i_y])``, where x-pos and y-pos are returned by this method. Note that the base frame is in the
        upper left corner.

        :return: the x and y positions of the tiles in separate numpy arrays, each of shape (num_tiles_x, num_tiles_y)
        """

        def get_1D_tile_pos(num_tiles: int, tile_wl: int) -> np.ndarray:
            pos = np.linspace(start=tile_wl / 2, stop=(num_tiles - 1) * tile_wl + (tile_wl / 2), num=num_tiles, endpoint=True)
            return pos

        x_pos_tiles, y_pos_tiles = np.meshgrid(
            get_1D_tile_pos(self.num_tiles_x, self.tile_size[0] * 2),
            get_1D_tile_pos(self.num_tiles_y, self.tile_size[1] * 2),
            indexing='ij',
        )

        return x_pos_tiles, y_pos_tiles

    def get_tile_indices_mask(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the x and y indices of tiles that correspond to the specified structure (the mask) in the tile layout.
        Note that the indices of the top left tile in the mask are returned.

        :param mask: a 2D numpy array containing only 0 and 1 which specifies the structure to be found in the tile layout
        :return: the x and y indices of the tiles in separate numpy arrays, each of shape (num_mask_found,)
        """
        assert len(mask.shape) == 2, 'Unexpected shape of the mask array.'
        assert np.bitwise_or(mask == 0, mask == 1).all(), 'Use a numpy array of only 0 and 1 to specify the mask.'

        offsets_x = (int(mask.shape[0] / 2) if mask.shape[0] % 2 == 1 else int(mask.shape[0] / 2) - 1, int(mask.shape[0] / 2) - 1)
        offsets_y = (int(mask.shape[1] / 2) if mask.shape[1] % 2 == 1 else int(mask.shape[1] / 2) - 1, int(mask.shape[1] / 2) - 1)

        tile_indices_x = []
        tile_indices_y = []
        for idx_x in range(offsets_x[0], self.num_tiles_x - offsets_x[1] - 1):
            for idx_y in range(offsets_y[0], self.num_tiles_y - offsets_y[1] - 1):
                if (
                    mask
                    == self.layout_tiles[
                        idx_x - offsets_x[0] : idx_x + offsets_x[1] + 2,
                        idx_y - offsets_y[0] : idx_y + offsets_y[1] + 2,
                    ]
                ).all():
                    tile_indices_x.append(idx_x)
                    tile_indices_y.append(idx_y)

        return np.array(tile_indices_x), np.array(tile_indices_y)

    ###################################################
    # Config Checks                                   #
    ###################################################

    def _check_tile_config(self) -> None:
        """Check that the tile layout, number of tiles, size and mass of a tile are as expected."""
        # check number of tiles and tile layout
        assert len(self.layout_tiles.shape) == 2, 'Unexpected tile layout shape. Expected: (num_tiles_x,num_tiles_y)'
        # fmt: off
        assert np.bitwise_or(self.layout_tiles == 0, self.layout_tiles == 1).all(), (
            'Use a numpy array of only 0 and 1 to specify the tile layout.'
        )
        # fmt: on
        assert self.num_tiles > 0, 'Number of tiles must be >0.'

        # check tile size
        assert self.tile_size.shape == (3,), 'Specify the size of a tile using a numpy array of shape (3,)'
        assert (self.tile_size > 0).all(), 'Tile size must be >0.'

        # check tile mass
        assert self.tile_mass > 0, 'Tile mass must be >0.'

    def _check_mover_config(self, initial_mover_start_xy_pos: np.ndarray | None, initial_mover_goal_xy_pos: np.ndarray | None) -> None:
        """Check that the number of movers, the size and mass of a mover, and the initial (x,y,z) positions are as expected.

        :param initial_mover_start_xy_pos: a numpy array containing individual (x,y) starting positions for each mover; can be None
            if no starting positions are specified
        :param initial_mover_goal_xy_pos: a numpy array containing individual (x,y) goal positions for some movers; can be None
            if no goal positions are specified
        """
        # check number of movers
        assert self.num_movers > 0, 'Number of movers must be >0.'
        assert self.num_movers > (self.num_movers_wo_goal - 1), 'Number of movers without goal >= number of movers'

        # check mover size
        assert (self.mover_size > 0).all(), 'Mover size must be >0.'
        assert self.mover_size.shape == (3,) or self.mover_size.shape == (self.num_movers, 3), (
            'Unexpected mover size. Use a numpy array of shape (3,) for equally sized movers '
            'and a numpy array of shape (num_movers, 3) to specify an individual size for each mover.'
        )

        # check mover mass
        # fmt: off
        assert isinstance(self.mover_mass, float) or isinstance(self.mover_mass, np.ndarray), (
            'Use a single float value or a numpy array of shape (num_movers,) to specify the mass of the movers.'
        )
        # fmt: on
        if isinstance(self.mover_mass, np.ndarray):
            assert self.mover_mass.shape == (self.num_movers,), (
                'Unexpected shape of the mover mass array. Expected: (num_movers,) to specify an individual mass for each mover '
                + 'or a single float value to use the same mass value for all movers'
            )
            assert (self.mover_mass > 0).all(), 'Mover mass must be >0.'
        else:
            assert self.mover_mass > 0, 'Mover mass must be >0.'

        # check intial mover z-pos
        assert self.initial_mover_zpos >= 0, 'Initial mover z position must be >= 0.'

        # check initial start and goal positions
        # fmt: off
        if initial_mover_start_xy_pos is not None:
            assert initial_mover_start_xy_pos.shape == (self.num_movers,2), (
                'Invalid shape of initial mover start positions. Expected: (num_movers,2)'
            )

        if initial_mover_goal_xy_pos is not None:
            assert initial_mover_goal_xy_pos.shape == (self.num_movers,2), (
                'Invalid shape of initial mover goal positions. Expected: (num_movers,2)'
            )
        # fmt: on

        # check that the mover shape is valid
        valid_shapes = ['box', 'cylinder', 'mesh']
        if isinstance(self.mover_shape, list):
            assert all(shape in valid_shapes for shape in self.mover_shape), (
                "Invalid mover shape. Must be one of: 'box', 'cylinder', 'mesh'."
            )
        else:
            assert self.mover_shape in valid_shapes, (
                f"Invalid mover shape '{self.mover_shape}'. Must be one of: 'box', 'cylinder', 'mesh'."
            )

        # check mover mesh params
        assert self.mover_mesh_bumper_mass >= 0, 'Bumper mass must be non-negative.'

    def _check_collision_params(self) -> None:
        """Check that the collision shape and the size of the collision shape are as expected."""
        # check collision shape
        assert self.c_shape == 'circle' or self.c_shape == 'box', 'Unexpected collision shape. You can choose between circle and box.'
        # fmt: off
        if self.c_shape == 'circle' and isinstance(self.c_size, np.ndarray):
            assert self.c_size.shape == (self.num_movers,), (
                'Use a single float value (radius) or a numpy array of shape (num_movers,) to specify the size parameter.'
            )
        elif self.c_shape == 'box':
            assert not isinstance(self.c_size, float), (
                'Use a numpy array of shape (2,) or (num_movers,2) to specify the size parameter.'
            )
            assert self.c_size.shape == (2,) or self.c_size.shape == (self.num_movers,2), (
                'The shape of the size array (collision_params["size"]) has to be (2,) or (num_movers,2).'
            )
        # fmt: on

        # check size of collision shape
        for idx_mover in range(0, self.num_movers):
            if isinstance(self.mover_shape, list):
                mover_shape = self.mover_shape[idx_mover]
            else:  # isinstance(self.mover_shape, str)
                mover_shape = self.mover_shape

            if mover_shape == 'box' or mover_shape == 'mesh':
                mover_size_x = self.resolved_mover_size[idx_mover, 0]
                mover_size_y = self.resolved_mover_size[idx_mover, 1]
            elif mover_shape == 'cylinder':
                mover_size_x = self.resolved_mover_size[idx_mover, 0]
                mover_size_y = self.resolved_mover_size[idx_mover, 0]
            else:
                raise ValueError(INVALID_MOVER_SHAPE_ERROR)

            if self.c_shape == 'circle':
                c_size = self.c_size[idx_mover] if isinstance(self.c_size, np.ndarray) else self.c_size
                if c_size < np.sqrt(mover_size_x**2 + mover_size_y**2):
                    logger.warn(
                        f'Mover {idx_mover} is not completely included in collision shape. You can avoid this warning by choosing '
                        + 'a larger collision_params["size"] value.'
                    )
            elif self.c_shape == 'box':
                c_size = self.c_size[:, idx_mover] if self.c_size.shape == (2, self.num_movers) else self.c_size
                if (c_size < np.array([mover_size_x, mover_size_y])).any():
                    logger.warn(
                        f'Mover {idx_mover} is not completely included in collision shape. You can avoid this warning by choosing '
                        + 'a larger collision_params["size"] value.'
                    )

        # check offsets
        assert isinstance(self.c_size_offset, float), 'Use a single float value to specify the size offset.'
        assert isinstance(self.c_size_offset_wall, float), 'Use a single float value to specify the wall offset.'
        assert self.c_size_offset >= 0, 'collision_params["offset"] must be >= 0.'
        assert self.c_size_offset_wall >= 0, 'collision_params["offset_wall"] must be >= 0.'

    def _check_mujoco_name_order(self) -> None:
        """Ensure that the mover names, joint and site names are ordered correctly. Thus, joint and site names for a
        specific mover can be found using the index of the mover name.
        """
        assert len(self.mover_names) == self.num_movers, 'Number of MuJoCo model mover names != number of movers'
        assert len(self.mover_joint_names) == self.num_movers, 'Number of MuJoCo model mover joint names != number of movers'
        # fmt: off
        assert len(self.mover_goal_site_names) == self.num_movers - self.num_movers_wo_goal, (
            'Number of MuJoCo model mover goal site names != number of movers - number of obstacle movers'
        )

        for idx_mover in range(0, self.num_movers):
            idx_str = self.mover_names[idx_mover].split('_')[-1]
            assert idx_str == self.mover_joint_names[idx_mover].split('_')[-1], (
                'Order of MuJoCo model mover joint names does not match the order of MuJoCo model mover names.'
            )
            if idx_mover < self.num_movers - self.num_movers_wo_goal:
                assert idx_str == self.mover_goal_site_names[idx_mover].split('_')[-1], (
                    'Order of MuJoCo model mover goal site names does not match the order of MuJoCo model mover names.'
                )
        # fmt: on

    def _find_mesh_dimensions(self, asset_xml_str: str | None, body_xml_str: str) -> np.ndarray:
        """Compute the axis-aligned bounding box dimensions of a mesh.

        This function creates a temporary MuJoCo model from the provided XML strings,
        simulates one step, and computes the bounding box dimensions by analyzing vertex
        positions of all mesh geoms attached to the specified body.

        Note: The function assumes all geoms are of type mesh and are attached to a body
        named 'mover_0'.
        """
        model_xml_str = f"""<?xml version="1.0" encoding="utf-8"?>
        <mujoco model="planar_robotics">
            <compiler angle="radian" coordinate="local" meshdir="{Path(__file__).parent.resolve() / 'assets'}" />

            <asset>
                <material name="black" reflectance="0.01" shininess="0.01" specular="0.1" rgba="0.25 0.25 0.25 1" />
                {asset_xml_str}
            </asset>

            <worldbody>{body_xml_str}</worldbody>
        </mujoco>"""

        model = mujoco.MjModel.from_xml_string(model_xml_str)  # type: ignore
        data = mujoco.MjData(model)  # type: ignore
        mujoco.mj_step(model, data, nstep=1)  # type: ignore

        body_id = model.body('mover_0').id
        body_vertices = []

        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] != body_id:
                continue

            assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH  # type: ignore

            mesh_id = model.geom_dataid[geom_id]

            geom_xpos = data.geom_xpos[geom_id]
            geom_xmat = data.geom_xmat[geom_id].reshape(3, 3).T

            vertadr = model.mesh_vertadr[mesh_id]
            vertnum = model.mesh_vertnum[mesh_id]

            vert = model.mesh_vert[vertadr : vertadr + vertnum]
            vert_xpos = geom_xpos + vert @ geom_xmat

            body_vertices.append(vert_xpos)

        # Just to make sure there's at least one vertex.
        assert body_vertices

        body_vertices = np.vstack(body_vertices)

        return np.max(body_vertices, axis=0) - np.min(body_vertices, axis=0)

    def _resolve_mover_size(self, mover_size: np.ndarray, mover_shape: str | list[str]) -> np.ndarray:
        """Resolve input size parameters to physical dimensions.

        This function handles the conversion between specified sizes and actual physical dimensions,
        which is particularly important for mesh geoms where MuJoCo allows scaling rather than direct
        size specification.

        Note: Fox 'box' and 'cylinder' shapes, the input sizes are used directly. For 'mesh' shapes,
        the function simulates the mesh to determine its actual dimensions based on the scaling
        parameters. All dimensions are half-sizes.
        """
        resolved_mover_size = np.zeros((self.num_movers, 3))

        for mover_idx in range(self.num_movers):
            if mover_size.shape == (3,):
                _mover_size = mover_size
            elif mover_size.shape == (self.num_movers, 3):
                _mover_size = mover_size[mover_idx]
            else:
                raise ValueError(f'Size must either be of shape (3,) or (num_movers, 3), but is {mover_size.shape}.')

            if isinstance(mover_shape, str):
                _mover_shape = mover_shape
            elif isinstance(mover_shape, list):
                _mover_shape = mover_shape[mover_idx]
            else:
                raise ValueError(f'Shape must be specified as either a `str` or a `list[str]`, but is {type(mover_shape)}.')

            if _mover_shape == 'box' or _mover_shape == 'cylinder':
                resolved_mover_size[mover_idx] = _mover_size
            elif _mover_shape == 'mesh':
                asset_xml_str, body_xml_str = self._generate_mover_xml_strings(0, 0, 0, 0, '', 1, _mover_size, _mover_shape)
                resolved_mover_size[mover_idx] = self._find_mesh_dimensions(asset_xml_str, body_xml_str) / 2  # half-sized

        return resolved_mover_size

    def _resolve_mesh_path(self, path: str) -> str | None:
        """Resolve a mesh path string to a Path object, either from predefined
        meshes or as a direct path.
        """
        predefined_meshes = {
            'beckhoff_apm4220_mover',
            'beckhoff_apm4220_bumper',
            'beckhoff_apm4330_mover',
            'beckhoff_apm4330_bumper',
            'beckhoff_apm4550_mover',
            'beckhoff_apm4550_bumper',
            'planar_motor_M3-06',
            'planar_motor_M3-15',
            'planar_motor_M3-25',
            'planar_motor_M4-11',
            'planar_motor_M4-18',
        }

        if path is None:
            return None

        if path in predefined_meshes:
            return f'./{path}.stl'

        return path


class BasicPlanarRoboticsMultiAgentEnv(BasicPlanarRoboticsEnv, ParallelEnv):
    """A base class for multi-agent reinforcement learning environments in the field of planar robotics that follow the PettingZoo
    API. A more detailed explanation of all parameters can be found in the documentation of the ``BasicPlanarRoboticsEnv``.

    :param layout_tiles: the tile layout
    :param num_movers: the number of movers
    :param tile_params: tile parameters such as the size and mass, defaults to None
    :param mover_params: mover parameters such as the size and mass, defaults to None
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.005 [m]
    :param table_height: the height of a table on which the tiles are placed, defaults to 0.4 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'
    :param default_cam_config: dictionary with attribute values of the viewer's default camera,
        https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global, defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
    :param collision_params: a dictionary that can be used to specify collision parameters, defaults to None
    :param initial_mover_start_xy_pos: the initial (x,y) starting positions of the movers, defaults to None
    :param initial_mover_goal_xy_pos: the initial (x,y) goal positions of the movers, defaults to None
    :param custom_model_xml_strings: a dictionary containing additional xml strings to provide the ability to add actuators, sensors,
        objects, robots, etc. to the model, defaults to None
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.005,
        table_height: float = 0.4,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        default_cam_config: dict[str, Any] | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        collision_params: dict[str, Any] | None = None,
        initial_mover_start_xy_pos: np.ndarray | None = None,
        initial_mover_goal_xy_pos: np.ndarray | None = None,
        custom_model_xml_strings: dict[str, str] | None = None,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        super(BasicPlanarRoboticsEnv, self).__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
            tile_params=tile_params,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            table_height=table_height,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=default_cam_config,
            width_no_camera_specified=width_no_camera_specified,
            height_no_camera_specified=height_no_camera_specified,
            collision_params=collision_params,
            initial_mover_start_xy_pos=initial_mover_start_xy_pos,
            initial_mover_goal_xy_pos=initial_mover_goal_xy_pos,
            custom_model_xml_strings=custom_model_xml_strings,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        self.agents = self.mover_names
        self.possible_agents = self.mover_names


class BasicPlanarRoboticsSingleAgentEnv(BasicPlanarRoboticsEnv, gym.Env, ABC):
    """A base class for single-agent reinforcement learning environments in the field of planar robotics that follow the Gymnasium
    API. A more detailed explanation of all parameters can be found in the documentation of the ``BasicPlanarRoboticsEnv``.

    :param layout_tiles: the tile layout
    :param num_movers: the number of movers
    :param tile_params: tile parameters such as the size and mass, defaults to None
    :param mover_params: mover parameters such as the size and mass, defaults to None
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.005 [m]
    :param table_height: the height of a table on which the tiles are placed, defaults to 0.4 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'
    :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after 'num_cycles' have been executed if 'render_mode != None'.
    :param default_cam_config: dictionary with attribute values of the viewer's default camera,
        https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global, defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
    :param num_cycles: the number of control cycles for which to apply the same action, defaults to 40
    :param collision_params: a dictionary that can be used to specify collision parameters, defaults to None
    :param initial_mover_start_xy_pos: the initial (x,y) starting positions of the movers, defaults to None
    :param initial_mover_goal_xy_pos: the initial (x,y) goal positions of the movers, defaults to None
    :param custom_model_xml_strings: a dictionary containing additional xml strings to provide the ability to add actuators, sensors,
        objects, robots, etc. to the model, defaults to None
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.005,
        table_height: float = 0.4,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        default_cam_config: dict[str, Any] | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        initial_mover_start_xy_pos: np.ndarray | None = None,
        initial_mover_goal_xy_pos: np.ndarray | None = None,
        custom_model_xml_strings: dict[str, str] | None = None,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        super().__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
            tile_params=tile_params,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            table_height=table_height,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=default_cam_config,
            width_no_camera_specified=width_no_camera_specified,
            height_no_camera_specified=height_no_camera_specified,
            collision_params=collision_params,
            initial_mover_start_xy_pos=initial_mover_start_xy_pos,
            initial_mover_goal_xy_pos=initial_mover_goal_xy_pos,
            custom_model_xml_strings=custom_model_xml_strings,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        self.render_every_cycle = render_every_cycle
        # number of control cycles for which to apply the same action
        self.num_cycles = num_cycles

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment returning an initial observation and auxiliary information. More detailed information about the
        parameters and return values can be found in the Gymnasium documentation:
        https://gymnasium.farama.org/api/env/#gymnasium.Env.reset.

        This method performs the following steps:

        - reset RNG, if desired
        - call ``_reset_callback(option)`` to give the user the opportunity to add more functionality
        - call ``mj_forward()``
        - check whether there are mover or wall collisions
        - call ``render()``
        - get initial observation and info dictionary

        :param seed: if set to None, the RNG is not reset; if int, sets the desired seed; defaults to None
        :param options: a dictionary that can be used to specify additional reset options, e.g. object parameters; defaults to None
        :return: initial observation and auxiliary information contained in the 'info' dictionary
        """
        # reset RNG of the environment if seed is not None
        super().reset(seed=seed)
        if seed is not None:
            self.rng_noise = np.random.default_rng(seed=seed)

        # custom callback to add more functionality
        self._reset_callback(options)

        # update sim
        mujoco.mj_forward(self.model, self.data)
        # check mover and wall collision
        wall_collision = self.check_wall_collision(
            mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=True, mover_qpos=None, add_qpos_noise=True
        ).any()
        # check mover collision
        mover_collision = self.check_mover_collision(
            mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=False, mover_qpos=None, add_qpos_noise=True
        ).any()

        # rendering
        self.render()

        # get new observation and info
        observation = self._get_obs()
        if isinstance(observation, dict) and 'achieved_goal' in observation.keys() and 'desired_goal' in observation.keys():
            info = self._get_info(
                mover_collision=mover_collision,
                wall_collision=wall_collision,
                achieved_goal=observation['achieved_goal'],
                desired_goal=observation['desired_goal'],
            )
        else:
            info = self._get_info(mover_collision=mover_collision, wall_collision=wall_collision)

        return observation, info

    def step(self, action: int | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one step of the environment's dynamics applying the given action.
        Note that the environment executes as many MuJoCo simulation steps as the number of cycles specified for this environment
        (``num_cycles``). The duration of one cycle is determined by the cycle time, which must be specified in the MuJoCo xml
        string using the ``option/timestep`` parameter. The same action is applied for all cycles.

        This method performs the following steps:

        - check whether the dimension of the action matches the dimension of the action space
        - if the action space does not contain the specified action, the action is clipped to the interval edges of
          the action space
        - call ``_step_callback(action)`` to give the user the opportunity to add more functionality
        - execute MuJoCo simulation steps (``mj_step()``). After each simulation step, it is checked whether there are mover or wall
          collisions. In case of a collision, mover_collision or wall_collision will be True and no further simulation
          steps are performed, as a real system would typically stop as well due to position lag errors.
          In addition, ``render()`` can be called after each simulation step to provide a smooth visualization of the movement
          (set ``render_every_cycle=True``).
          The callback ``_mujoco_step_callback(action)`` can be used to add functionality BEFORE the next simulation step is executed.
          This can be useful, for example, to ensure velocity or acceleration limits within each cycle.
        - call ``render()``
        - get return values

        More detailed information about the parameters and return values can be found in the Gymnasium documentation:
        https://gymnasium.farama.org/api/env/#gymnasium.Env.step.

        :param action: the action to apply
        :return:
                - the next observation
                - the immediate reward for taking the action
                - whether a terminal state is reached
                - whether the truncation condition is satisfied
                - auxiliary information contained in the 'info' dictionary
        """
        # make sure that shape is correct and action is within action space
        if not isinstance(action, int):
            assert action.shape == self.action_space.shape, 'action dim != action_space dim'
            if not self.action_space.contains(action):
                logger.warn(f'Action {action} not in action space. Will clip invalid values to interval edges.')
                action = np.clip(action, self.action_space.low, self.action_space.high)

        # custom callback to add more functionality
        self._step_callback(action)

        # integration and collision check
        for _ in range(0, self.num_cycles):
            self._mujoco_step_callback(action)
            # integration
            mujoco.mj_step(self.model, self.data, nstep=1)
            # render every cycle for a smooth visualization of the movement
            if self.render_every_cycle:
                self.render()
            # check wall and mover collision every cycle to ensure that the collisions are detected and all intermediate
            # mover positions are valid and without collisions
            wall_collision = self.check_wall_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            ).any()
            mover_collision = self.check_mover_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            )
            if mover_collision or wall_collision:
                break

        self.render()

        # get next observation
        observation = self._get_obs()
        if isinstance(observation, dict) and 'achieved_goal' in observation.keys() and 'desired_goal' in observation.keys():
            # goal-conditioned RL
            info = self._get_info(mover_collision, wall_collision, observation['achieved_goal'], observation['desired_goal'])
            reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)
            terminated = self.compute_terminated(observation['achieved_goal'], observation['desired_goal'], info)
            truncated = self.compute_truncated(observation['achieved_goal'], observation['desired_goal'], info)
        else:
            info = self._get_info(mover_collision, wall_collision)
            reward = self.compute_reward(info)
            terminated = self.compute_terminated(info)
            truncated = self.compute_truncated(info)
        # check reward shape
        if isinstance(reward, np.ndarray) and reward.shape[0] > 1:
            logger.warn(
                f"Unexpected shape of reward returned by 'env.compute_reward()'. Current shape is: {reward.shape}, \
                  expected shape: (1,)"
            )
        elif isinstance(reward, np.ndarray) and reward.shape[0] == 1:
            reward = reward[0]

        return observation, reward, terminated, truncated, info

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """A callback that should be used to add further functionality to the ``reset()`` method (see documentation of the ``reset()``
        method for more information about when the callback is called).

        :param options: a dictionary that can be used to specify additional reset options, e.g. object parameters; defaults to None
        """
        pass

    def _step_callback(self, action: int | np.ndarray) -> None:
        """A callback that should be used to add further functionality to the ``step()`` method (see documentation of the ``step()``
        method for more information about when the callback is called).

        :param action: the action to apply
        """
        pass

    def _mujoco_step_callback(self, action: int | np.ndarray) -> None:
        """A callback that should be used to add further functionality to the ``step()`` method (see documentation of the ``step()``
        method for more information about when the callback is called).

        :param action: the action to apply
        """
        pass

    @abstractmethod
    def compute_terminated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. This method can be used for both goal-conditioned RL and standard RL.
        Since Hindsight Experience Replay (HER) is commonly used in goal-conditioned RL, this method receives
        the 'achieved_goal' and 'desired_goal' corresponding to the requirements of the HER implementation of stable-baselines3
        (for more information, see https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            goals already achieved (goal-conditioned RL); defaults to None (standard RL)
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            desired goals (goal-conditioned RL); defaults to None (standard RL)
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single bool value or a numpy array of shape (batch_size,) containing Boolean values, where True indicates that
            a terminal state has been reached
        """
        pass

    @abstractmethod
    def compute_truncated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. This method can be used for both goal-conditioned RL and standard RL.
        Since Hindsight Experience Replay (HER) is commonly used in goal-conditioned RL, this method receives
        the 'achieved_goal' and 'desired_goal' corresponding to the requirements of the HER implementation of stable-baselines3
        (for more information, see https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            goals already achieved (goal-conditioned RL); defaults to None (standard RL)
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            desired goals (goal-conditioned RL); defaults to None (standard RL)
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single bool value or a numpy array of shape (batch_size,) containing Boolean values, where True indicates that
            a the truncation condition is satisfied
        """
        pass

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | float:
        """Compute the immediate reward. This method is required by the stable-baselines3 implementation of Hindsight Experience
        Replay (HER) (for more information, see https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            goals already achieved (goal-conditioned RL); defaults to None (standard RL)
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            desired goals (goal-conditioned RL); defaults to None (standard RL)
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        pass

    @abstractmethod
    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a numpy array or a dictionary (dictionary observation space - required by HER implementation of stable-baselines3)
        """
        pass

    @abstractmethod
    def _get_info(
        self,
        mover_collision: bool,
        wall_collision: bool,
        achieved_goal: np.ndarray | None = None,
        desired_goal: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return a dictionary that contains auxiliary information that may depend on the 'achieved_goal' and 'desired_goal' in
        goal-conditioned RL.

        :param mover_collision: whether there is a collision between two movers
        :param wall:collision: whether there is a collision between a mover and a wall
        :param achieved_goal: a numpy array containing the goal which already achieved (goal-conditioned RL) - the shape
            depends on the shape of the observation space; defaults to None
        :param desired_goal: a numpy array containing the desired goal (goal-conditioned RL) - the shape
            depends on the shape of the observation space; defaults to None
        :return: a dictionary with auxiliary information
        """
        pass
