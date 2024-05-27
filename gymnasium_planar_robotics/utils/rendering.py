##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

import numpy as np
import mujoco.viewer
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, BaseRender, OffScreenViewer, WindowViewer
from mujoco import MjData, MjModel
import matplotlib.pyplot as plt
from gymnasium_planar_robotics.utils import rotations_utils
from matplotlib.patches import Rectangle, Circle, Arrow


class MujocoWindowViewer(WindowViewer):
    """A window renderer class for MuJoCo environments with custom overlays.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    """

    def __init__(self, model: MjModel, data: MjData) -> None:
        super().__init__(model=model, data=data)
        self.is_running = True

    def close(self) -> None:
        """Close the viewer."""
        super().close()
        self.is_running = False

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if self._render_every_frame:
            self.add_overlay(topleft, '', '')
        else:
            self.add_overlay(
                topleft,
                f'Run speed = {self._run_speed:.3f} x real time',
                '[S]lower, [F]aster',
            )
        self.add_overlay(
            topleft,
            f'Switch camera (#cams = {self.model.ncam + 1})',
            f'[Tab] (camera ID = {self.cam.fixedcamid})',
        )
        self.add_overlay(topleft, '[C]ontact forces', 'On' if self._contacts else 'Off')
        self.add_overlay(topleft, 'T[r]ansparent', 'On' if self._transparent else 'Off')
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(topleft, 'Stop', '[Space]')
            else:
                self.add_overlay(topleft, 'Start', '[Space]')
                self.add_overlay(topleft, 'Advance simulation by one step', '[right arrow]')
        self.add_overlay(topleft, 'Referenc[e] frames', 'On' if self.vopt.frame == 1 else 'Off')
        self.add_overlay(topleft, '[H]ide Menu', '')
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(topleft, 'Cap[t]ure frame', f'Saved as {fname}')
        else:
            self.add_overlay(topleft, 'Cap[t]ure frame', '')
        self.add_overlay(topleft, 'Toggle geomgroup visibility', '0-4')

        self.add_overlay(bottomleft, 'FPS', f'{int(1 / self._time_per_render)} ')
        self.add_overlay(bottomleft, 'Step', str(round(self.data.time / self.model.opt.timestep)))
        self.add_overlay(bottomleft, 'timestep', f'{self.model.opt.timestep:.5f}')


class MujocoOffScreenViewer(OffScreenViewer):
    """An extension of the Gymnasium OffScreenViewer which allows to also specify the groups of geoms to be rendered by the
    offscreen renderer.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param width: width of the OpenGL rendering context
    :param height: height of the OpenGL rendering context
    :param geomgroup: a numpy array of shape (6,), where each entry is either 0 or 1, specifying the groups of geoms to be
        rendered by the camera, defaults to None (vopt.geomgroup is not changed)
    """

    def __init__(self, model: MjModel, data: MjData, width: int, height: int, geomgroup: np.ndarray | None = None) -> None:
        self._get_opengl_backend(width, height)
        BaseRender.__init__(self, model, data, width, height)

        if geomgroup is not None:
            self.set_geomgroup(geomgroup)

    def set_geomgroup(self, geomgroup: np.ndarray) -> None:
        """Set the groups of geoms that should be rendered.

        :param geomgroup: a numpy array of shape (6,), where each entry is either 0 or 1, specifying the groups of geoms to be
            rendered by the camera
        """
        assert geomgroup.shape == (6,), 'Invalid geomgroup shape. Expected: (6,)'
        assert np.issubdtype(geomgroup.dtype, np.integer), 'Dtype geomgroup is not an integer'
        self.vopt.geomgroup = geomgroup


class MujocoViewerCollection(MujocoRenderer):
    """A manager for all renderers for a MuJoCo environment.
    It provides the possibility to manage one renderer with render_mode 'human' and multiple offscreen renderers.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param default_cam_config: dictionary with attribute values of the viewer's default camera
        (see https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global), defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        model: MjModel,
        data: MjData,
        default_cam_config: dict | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        self.width_no_camera_specified = width_no_camera_specified
        self.height_no_camera_specified = height_no_camera_specified

        self.use_mj_passive_viewer = use_mj_passive_viewer

        super().__init__(model, data, default_cam_config)

    def render(
        self,
        render_mode: str,
        camera_id: int | None = None,
        camera_name: str | None = None,
        width: int = 64,
        height: int = 64,
        geomgroup: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Render a frame of the specified camera.

        :param render_mode: 'human', 'rgb_array' or 'depth_array'
        :param camera_id: the id of the camera used for rendering (only used if render_mode != 'human'), defaults to None
        :param camera_name: the name of the camera used for rendering (only used if render_mode != 'human'), defaults to None.
            You cannot specify both camera_id and camera_name.
        :param width: width of the OpenGL rendering context (only used if render_mode != 'human'), defaults to 64
        :param height: height of the OpenGL rendering context (only used if render_mode != 'human'), defaults to 64
        :param geomgroup: a numpy array of shape (6,), where each entry is either 0 or 1, specifying the groups of geoms to be
            rendered by the camera (only used if render_mode != 'human'), defaults to None (vopt.geomgroup is not changed)
        :raises ValueError: if render_mode != 'human' and ``camera_id`` and ``camera_name`` are specified at the same time
        :return: returns a numpy array if render_mode != 'human', otherwise it returns None (render_mode 'human')
        """
        if render_mode in {
            'rgb_array',
            'depth_array',
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError('Both `camera_id` and `camera_name` cannot be specified at the same time.')

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'
                self.model.vis.global_.offwidth = self.width_no_camera_specified
                self.model.vis.global_.offheight = self.height_no_camera_specified
                width = self.width_no_camera_specified
                height = self.height_no_camera_specified

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

            self._update_viewer(render_mode, camera_id, no_camera_specified, width, height, geomgroup)
            img = self.viewer.render(render_mode=render_mode, camera_id=camera_id)
            return img
        elif render_mode.startswith('human'):
            self._update_viewer(render_mode)
            if self.use_mj_passive_viewer:
                return self.viewer.sync()
            else:
                return self.viewer.render()

    def window_viewer_is_running(self) -> bool:
        """Check whether the window renderer (render_mode 'human') is active, i.e. the window is open.

        :return: True if the window is open, False otherwise
        """
        viewer_name = self._get_viewer_name(render_mode='human')
        viewer = self._viewers.get(viewer_name)
        if viewer is None:
            return False
        if self.use_mj_passive_viewer:
            return viewer.is_running()
        else:
            return viewer.is_running is not None

    def _get_viewer_name(self, render_mode: str, camera_id: int | None = None) -> str:
        """Return the name of the desired viewer. If the render_mode is 'human', viewer_name is 'human', since there can only be one
        renderer for this render_mode. If the render_mode is 'rgb_array' or 'depth_array', the name is composed of the render_mode
        and the camera_id.

        :param render_mode: 'human', 'rgb_array' or 'depth_array'
        :param camera_id: the id of the camera used for rendering (only used if render_mode != 'human'), defaults to None
        :return: returns the name of the desired viewer
        """
        if render_mode.startswith('human'):
            return render_mode
        else:
            assert render_mode in ['rgb_array', 'depth_array'], f'Unkown render mode: {render_mode}'
            return render_mode + f'_{camera_id}'

    def _update_viewer(
        self,
        render_mode: str,
        camera_id: int | None = None,
        no_camera_specified: bool | None = None,
        width: int = 64,
        height: int = 64,
        geomgroup: np.ndarray | None = None,
    ) -> None:
        """Update ``self.viewer``. If the desired viewer does not exist, it initializes the viewer depending on the render mode.

        :param render_mode: 'human', 'rgb_array' or 'depth_array'
        :param camera_id: the id of the camera used for rendering (only used if render_mode != 'human'), defaults to None
        :param no_camera_specified: whether a camera is specified (bool or None, only used if render_mode != 'human'),
            defaults to None
        :param width: width of the OpenGL rendering context (only used if render_mode != 'human'), defaults to 64
        :param height: height of the OpenGL rendering context (only used if render_mode != 'human'), defaults to 64
        :param geomgroup: a numpy array of shape (6,), where each entry is either 0 or 1, specifying the groups of geoms to be
            rendered by the camera (only used if render_mode != 'human'), defaults to None (vopt.geomgroup is not changed)
        """
        viewer_name = self._get_viewer_name(render_mode, camera_id)
        self.viewer = self._viewers.get(viewer_name)

        if render_mode in ['rgb_array', 'depth_array']:
            if self.viewer is None:
                self.viewer = MujocoOffScreenViewer(model=self.model, data=self.data, width=width, height=height, geomgroup=geomgroup)
                self._viewers[viewer_name] = self.viewer

                if no_camera_specified:
                    self._set_cam_config()

            elif geomgroup is not None:
                self.viewer.set_geomgroup(geomgroup)
        else:
            if self.viewer is None:
                if self.use_mj_passive_viewer:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                else:
                    self.viewer = MujocoWindowViewer(self.model, self.data)
                self._set_cam_config()
                self._viewers[viewer_name] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

    def reload_model(self, model: MjModel, data: MjData) -> None:
        """Reload the model and data of each viewer. The intended use of this method is as follows:
        Some environments contain objects with changing parameters, e.g. mass, friction coefficients, size or shape.
        To train a RL agent that generalizes to new objects, it can be desirable to change the object's parameters when ``env.reset()``
        is called. To ensure that the MuJoCo simulation behaves as intended, a new model XML string should be generated and loaded.
        In this case, the viewer also needs to be updated, which is possible with this method.

        :param model: the new mjModel of the MuJoCo environment
        :param data: the new mjData of the MuJoCo environment
        """
        self.model = model
        self.data = data

        for viewer_name in self._viewers.keys():
            if 'rgb_array' in viewer_name or 'depth_array' in viewer_name or not self.use_mj_passive_viewer:
                viewer = self._viewers[viewer_name]
                viewer.model = model
                viewer.data = data
            elif 'human' in viewer_name and self.use_mj_passive_viewer:
                self._viewers[viewer_name].close()
                viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._set_cam_config()
                self._viewers[viewer_name] = viewer


class Matplotlib2DViewer:
    """A class to easily plot the tile and mover configuration along with the mover collision offsets using matplotlib. This class
    should primarily be used for debugging and analysis of specific planning situations. It is not included in the
    MujocoViewerCollection, as it offers fewer possibilities compared to the MuJoCo 3D Viewer (render_mode 'human'). Thus, it must
    be integrated into the environment by the user, e.g. using the ``render_callback()``.

    :param layout_tiles: a numpy array of shape (num_tiles_x, num_tiles_y) indicating where to add a tile (use 1 to add a tile
        and 0 to leave cell empty). The x-axis and y-axis correspond to the axes of the numpy array, so the origin of the base
        frame is in the upper left corner.
    :param num_movers: the number of movers to plot
    :param mover_sizes: a numpy array of shape (3,) or (num_movers,3) specifying the half-sizes of each mover (x,y,z)
    :param mover_colors: a list of strings specifying a color for each mover
    :param tile_size: a numpy array of shape (3,) specifying the half-sizes of a tile (x,y,z)
    :param x_pos_tiles: the x-positions of the tiles specified as a numpy array of shape (num_tiles_x, num_tiles_y)
    :param y_pos_tiles: the y-positions of the tiles specified as a numpy array of shape (num_tiles_x, num_tiles_y)
    :param c_shape: collision shape; can be 'box' or 'circle', defaults to 'circle'
    :param c_size: the size of the collision shape, defaults to 0.11:

        - collision shape 'circle':
            a single float value which corresponds to the radius of the circle, or a numpy array of shape (num_movers,) to specify
            individual values for each mover
        - collision shape 'box':
            a numpy array of shape (2,) to specify x and y half-size of the box, or a numpy array of shape (num_movers, 2) to specify
            individual sizes for each mover
    :param c_size_offset: an additional safety offset that is added to the size of the collision shape, defaults to 0.0.
        Think of this offset as increasing the size of a mover by a safety margin.
    :param arrow_scale: the scaling factor of the arrow length, which displays the current (x,y)-velocity of a mover, defaults
        to 0.3
    :param figure_size: the size of the matplotlib figure, defaults to (7,7)
    """

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        mover_sizes: np.ndarray,
        mover_colors: list[str],
        tile_size: np.ndarray,
        x_pos_tiles: np.ndarray,
        y_pos_tiles: np.ndarray,
        c_shape: str = 'circle',
        c_size: np.ndarray | float = 0.11,
        c_size_offset: float = 0.0,
        arrow_scale: float = 0.3,
        figure_size: tuple = (7, 7),
    ) -> None:
        # mover params
        self.num_movers = num_movers
        self.mover_sizes = mover_sizes
        self.mover_colors = mover_colors
        if len(self.mover_colors) < self.num_movers:
            raise ValueError('The number of specified mover colors does not match the number of movers.')
        # collision params
        self.c_shape = c_shape
        self.c_size = c_size
        self.c_size_offset = c_size_offset

        if self.mover_sizes.shape == (3,):
            self.mover_sizes = np.tile(mover_sizes, reps=(self.num_movers, 1))

        if self.c_shape == 'circle' and isinstance(self.c_size, float):
            self.c_size_arr = np.array([self.c_size] * self.num_movers)
        elif self.c_shape == 'circle' and self.c_size.shape == (self.num_movers,):
            self.c_size_arr = self.c_size.copy()
        elif self.c_shape == 'box' and self.c_size.shape == (2,):
            self.c_size_arr = np.tile(self.c_size, reps=(self.num_movers, 1))
        elif self.c_shape == 'box' and self.c_size.shape == (self.num_movers, 2):
            self.c_size_arr = self.c_size.copy()
        self.c_size_arr_offset = self.c_size_arr + self.c_size_offset

        # 2D plot
        self.arrow_scale = arrow_scale
        self.figure, self.axs = plt.subplots(1, 1, figsize=figure_size)
        # plot tiles (note that x and y axes are swapped similar to an image)
        num_tiles_x = layout_tiles.shape[0]
        num_tiles_y = layout_tiles.shape[1]
        y_ticks = np.linspace(0, num_tiles_x * (tile_size[0] * 2), num=num_tiles_x + 1, endpoint=True)
        x_ticks = np.linspace(0, num_tiles_y * (tile_size[1] * 2), num=num_tiles_y + 1, endpoint=True)
        self.axs.set_yticks(y_ticks)
        self.axs.set_xticks(x_ticks)
        self.axs.set_xlim([np.min(x_ticks), np.max(x_ticks)])
        self.axs.set_ylim([np.min(y_ticks), np.max(y_ticks)])
        self.axs.invert_yaxis()
        self.axs.grid()
        self.axs.set_ylabel('x pos')
        self.axs.set_xlabel('y pos')
        self.axs.set_aspect('equal')
        # visualize missing tiles
        for idx_x in range(0, num_tiles_x):
            for idx_y in range(0, num_tiles_y):
                if not layout_tiles[idx_x, idx_y]:
                    tile_ul = (y_pos_tiles[idx_x, idx_y] - tile_size[1], x_pos_tiles[idx_x, idx_y] - tile_size[0])
                    rect = Rectangle(tile_ul, width=tile_size[1] * 2, height=tile_size[0] * 2, color='silver', zorder=1)
                    self.axs.add_patch(rect)

        self.movers = []
        self.cs = []
        self.cs_offset = []
        self.arrows = []
        self.goals = []

    def render(self, mover_qpos: np.ndarray, mover_qvel: np.ndarray, mover_goals: np.ndarray | None = None) -> None:
        """Render the next frame.

        :param mover_qpos: a numpy array of shape (num_movers,7) containing the qpos (x_p,y_p,z_p,w_o,x_o,y_o,z_o) of each mover
        :param mover_qvel: a numpy array of shape (num_movers,6) containing the qvel (x,y,z,a,b,c) of each mover
        :param mover_goals: None or a numpy array of shape (num_movers,2) containing the (x,y) goal positions of each mover, defaults
            to None. If set to None, no goals are displayed.
        """
        for i in range(0, len(self.movers)):
            self.movers[i].remove()
            self.cs[i].remove()
            self.cs_offset[i].remove()
            self.arrows[i].remove()
            if len(self.goals) > 0:
                self.goals[i].remove()

        self.movers = []
        self.cs = []
        self.cs_offset = []
        self.arrows = []
        self.goals = []

        for idx_mover in range(0, self.num_movers):
            # we assume that the angles about the x and y axes are close to 0
            euler = rotations_utils.quat2euler(quat=mover_qpos[idx_mover, -4:])

            mover_rect = Rectangle(
                (mover_qpos[idx_mover, 1] - self.mover_sizes[idx_mover, 1], mover_qpos[idx_mover, 0] - self.mover_sizes[idx_mover, 0]),
                width=self.mover_sizes[idx_mover, 1] * 2,
                height=self.mover_sizes[idx_mover, 0] * 2,
                angle=euler[-1] * (180 / np.pi),
                rotation_point='center',
                color=self.mover_colors[idx_mover],
                alpha=0.9,
                fill=True,
                zorder=2,
            )
            self.movers.append(self.axs.add_patch(mover_rect))

            arrow = Arrow(
                x=mover_qpos[idx_mover, 1],
                y=mover_qpos[idx_mover, 0],
                dx=self.arrow_scale * mover_qvel[idx_mover, 1],
                dy=self.arrow_scale * mover_qvel[idx_mover, 0],
                width=0.06,
                facecolor=self.mover_colors[idx_mover],
                edgecolor='black',
                lw=0.06,
                zorder=1.5,
            )
            self.arrows.append(self.axs.add_patch(arrow))

            if self.c_shape == 'circle':
                cs = Circle(
                    (mover_qpos[idx_mover, 1], mover_qpos[idx_mover, 0]),
                    self.c_size_arr[idx_mover],
                    color=self.mover_colors[idx_mover],
                    fill=False,
                    zorder=2,
                )
                self.cs.append(self.axs.add_patch(cs))

                cs_offset = Circle(
                    (mover_qpos[idx_mover, 1], mover_qpos[idx_mover, 0]),
                    self.c_size_arr_offset[idx_mover],
                    color=self.mover_colors[idx_mover],
                    fill=False,
                    linestyle='--',
                    zorder=2,
                )
                self.cs_offset.append(self.axs.add_patch(cs_offset))
            else:
                assert self.c_shape == 'box'
                cs = Rectangle(
                    (
                        mover_qpos[idx_mover, 1] - self.c_size_arr[idx_mover, 1],
                        mover_qpos[idx_mover, 0] - self.c_size_arr[idx_mover, 0],
                    ),
                    width=self.c_size_arr[idx_mover, 1] * 2,
                    height=self.c_size_arr[idx_mover, 0] * 2,
                    angle=euler[-1] * (180 / np.pi),
                    rotation_point='center',
                    color=self.mover_colors[idx_mover],
                    fill=False,
                    zorder=2,
                )
                self.cs.append(self.axs.add_patch(cs))

                cs_offset = Rectangle(
                    (
                        mover_qpos[idx_mover, 1] - self.c_size_arr_offset[idx_mover, 1],
                        mover_qpos[idx_mover, 0] - self.c_size_arr_offset[idx_mover, 0],
                    ),
                    width=self.c_size_arr_offset[idx_mover, 1] * 2,
                    height=self.c_size_arr_offset[idx_mover, 0] * 2,
                    angle=euler[-1] * (180 / np.pi),
                    rotation_point='center',
                    color=self.mover_colors[idx_mover],
                    fill=False,
                    linestyle='--',
                    zorder=2,
                )
                self.cs_offset.append(self.axs.add_patch(cs_offset))

            if mover_goals is not None:
                goal = self.axs.plot(
                    mover_goals[idx_mover, 1],
                    mover_goals[idx_mover, 0],
                    color=self.mover_colors[idx_mover],
                    marker='x',
                    ms=10,
                    mew=5,
                    lw=5,
                    zorder=3,
                )[0]
                self.goals.append(goal)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.0001)

    def close(self):
        """Close the figure."""
        plt.close(self.figure)
