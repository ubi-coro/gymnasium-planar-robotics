##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

import numpy as np
import mujoco
from mujoco import MjData, MjModel, mjtObj


def set_actuator_ctrl(model: MjModel, data: MjData, actuator_name: str, value: float) -> None:
    """Set the control inputs for the desired actuator.

    :param model: the MuJoCo model (MjModel)
    :param data: the MuJoCo data structure (MjData)
    :param actuator_name: the name of the actuator for which to set the control input
    :param value: the control input of the actuator
    """
    actuator_id = model.actuator(actuator_name).id
    assert actuator_id != -1, 'Actuator name not found in MuJoCo model.'
    data.ctrl[actuator_id] = value


def get_joint_qacc(model: MjModel, data: MjData, name: str) -> np.ndarray:
    """Return the joint's linear and angular acceleration (qacc) depending on the type of joint.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param name: the name of the joint
    :return: the qacc of the joint
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, 'Joint name not found in MuJoCo model.'
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qacc[start_idx:end_idx].copy()


def get_mujoco_type_names(model: MjModel, obj_type: str, name_pattern: str = '') -> list[str]:
    """Return a list of names of the current model that belong to objects of the specified type and whose names match
    the given name_pattern.

    :param model: mjModel of the MuJoCo environment
    :param obj_type: mjData of the MuJoCo environment
    :param name_pattern: the pattern to search for in the object names, defaults to ''
    :return: a list of object names matching the name_pattern and the obj_type
    """
    mj_model_names = MujocoModelNames(model)
    names_all = getattr(mj_model_names, f'{obj_type}_names')
    names = [name for name in names_all if name_pattern in name]
    return names


##############################################################################################################
# The following functions and classes are completely or at least partially adopted from gymnasium-robotics:  #
# https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/utils/mujoco_utils.py #
##############################################################################################################

MJ_OBJ_TYPES = [
    'mjOBJ_BODY',
    'mjOBJ_JOINT',
    'mjOBJ_GEOM',
    'mjOBJ_SITE',
    'mjOBJ_CAMERA',
    'mjOBJ_ACTUATOR',
    'mjOBJ_SENSOR',
]


def set_joint_qpos(model: MjModel, data: MjData, name: str, value: float | np.ndarray) -> None:
    """Set the joint positions (qpos) of the model.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param name: the name of the joint
    :param value: the new qpos
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, 'Joint name not found in MuJoCo model.'
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (end_idx - start_idx), f'Value has incorrect shape {name}: {value}'
    data.qpos[start_idx:end_idx] = value


def get_joint_qpos(model: MjModel, data: MjData, name: str) -> np.ndarray:
    """Return the joint's position and orientation (qpos) depending on the type of the joint.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param name: the name of the joint
    :return: the qpos of the joint
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, 'Joint name not found in MuJoCo model.'
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qpos[start_idx:end_idx].copy()


def set_joint_qvel(model: MjModel, data: MjData, name: str, value: float | np.ndarray) -> None:
    """Set the joints linear and angular (qvel) of the model.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param name: the name of the joint
    :param value: the new qvel
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, 'Joint name not found in MuJoCo model.'
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 3
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (end_idx - start_idx), f'Value has incorrect shape {name}: {value}'
    data.qvel[start_idx:end_idx] = value


def get_joint_qvel(model: MjModel, data: MjData, name: str) -> np.ndarray:
    """Return the joint's linear and angular velocities (qvel) depending on the type of the joint.

    :param model: mjModel of the MuJoCo environment
    :param data: mjData of the MuJoCo environment
    :param name: the name of the joint
    :return: the qvel of the joint
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, 'Joint name not found in MuJoCo model.'
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qvel[start_idx:end_idx].copy()


def extract_mj_names(model: MjModel, obj_type: mjtObj) -> tuple[tuple[str, ...] | tuple[()], dict[str, int], dict[int, str]]:
    """Extract the names and ids of the given object type from the current MuJoCo model.

    :raises ValueError: if the object type is not supported
    :return: the names, name2id and id2name dictionaries for the current object type
    """
    if obj_type == mujoco.mjtObj.mjOBJ_BODY:
        name_addr = model.name_bodyadr
        n_obj = model.nbody

    elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        name_addr = model.name_jntadr
        n_obj = model.njnt

    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        name_addr = model.name_geomadr
        n_obj = model.ngeom

    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        name_addr = model.name_siteadr
        n_obj = model.nsite

    elif obj_type == mujoco.mjtObj.mjOBJ_LIGHT:
        name_addr = model.name_lightadr
        n_obj = model.nlight

    elif obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
        name_addr = model.name_camadr
        n_obj = model.ncam

    elif obj_type == mujoco.mjtObj.mjOBJ_ACTUATOR:
        name_addr = model.name_actuatoradr
        n_obj = model.nu

    elif obj_type == mujoco.mjtObj.mjOBJ_SENSOR:
        name_addr = model.name_sensoradr
        n_obj = model.nsensor

    elif obj_type == mujoco.mjtObj.mjOBJ_TENDON:
        name_addr = model.name_tendonadr
        n_obj = model.ntendon

    elif obj_type == mujoco.mjtObj.mjOBJ_MESH:
        name_addr = model.name_meshadr
        n_obj = model.nmesh
    else:
        raise ValueError(
            f'`{obj_type}` was passed as the MuJoCo model object type. The MuJoCo model object type can only be of the following '
            + f'mjtObj enum types: {MJ_OBJ_TYPES}.'
        )

    id2name = {i: None for i in range(n_obj)}
    name2id = {}
    for addr in name_addr:
        name = model.names[addr:].split(b'\x00')[0].decode()
        if name:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name

    return tuple(id2name[id] for id in sorted(name2id.values())), name2id, id2name


class MujocoModelNames:
    """Access mjtObj object names and ids of the current MuJoCo model.

    This class supports access to the names and ids of the following mjObj types:

    - mjOBJ_BODY
    - mjOBJ_JOINT
    - mjOBJ_GEOM
    - mjOBJ_SITE
    - mjOBJ_CAMERA
    - mjOBJ_ACTUATOR
    - mjOBJ_SENSOR

    The properties provided for each ``mjObj`` are:

    - ``mjObj`` names: list of the mjObj names in the model of type mjOBJ_FOO
    - ``mjObj`` name2id: dictionary with name of the mjObj as keys and id of the mjObj as values
    - ``mjObj`` id2name: dictionary with id of the mjObj as keys and name of the mjObj as values

    """

    def __init__(self, model: MjModel) -> None:
        """Access mjtObj object names and ids of the current MuJoCo model.

        :param model: mjModel of the MuJoCo environment
        """
        (
            self._body_names,
            self._body_name2id,
            self._body_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_BODY)
        (
            self._joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_JOINT)
        (
            self._geom_names,
            self._geom_name2id,
            self._geom_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_GEOM)
        (
            self._site_names,
            self._site_name2id,
            self._site_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SITE)
        (
            self._camera_names,
            self._camera_name2id,
            self._camera_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_CAMERA)
        (
            self._actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
        (
            self._sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SENSOR)

    @property
    def body_names(self) -> tuple[str, ...] | tuple[()]:
        """Return a tuple of body names in the MuJoCo model.

        :return: body names
        """
        return self._body_names

    @property
    def body_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the bodies in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._body_name2id

    @property
    def body_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the bodies in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._body_id2name

    @property
    def joint_names(self) -> tuple[str, ...] | tuple[()]:
        """Return a tuple of joint names in the MuJoCo model.

        :return: joint names
        """
        return self._joint_names

    @property
    def joint_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the joints in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._joint_name2id

    @property
    def joint_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the joints in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._joint_id2name

    @property
    def geom_names(self) -> tuple[str, ...] | tuple[()]:
        """Return a tuple of geom names in the MuJoCo model.

        :return: geom names
        """
        return self._geom_names

    @property
    def geom_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the geoms in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._geom_name2id

    @property
    def geom_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the geoms in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._geom_id2name

    @property
    def site_names(self) -> tuple[str, ...] | tuple[()]:
        """Return the site names in the MuJoCo model.

        :return: tuple of model site names
        """
        return self._site_names

    @property
    def site_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the sites in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._site_name2id

    @property
    def site_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the sites in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._site_id2name

    @property
    def camera_names(self) -> tuple[str, ...] | tuple[()]:
        """Return a tuple of camera names in the MuJoCo model.

        :return: camera names
        """
        return self._camera_names

    @property
    def camera_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the cameras in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._camera_name2id

    @property
    def camera_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the cameras in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._camera_id2name

    @property
    def actuator_names(self) -> tuple[str, ...] | tuple[()]:
        """Return a tuple of actuator names in the MuJoCo model.

        :return: actuator names
        """
        return self._actuator_names

    @property
    def actuator_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the actuators in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._actuator_name2id

    @property
    def actuator_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the actuators in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._actuator_id2name

    @property
    def sensor_names(self) -> tuple[str, ...] | tuple[()]:
        """Return a tuple of sensor names in the MuJoCo model.

        :return: sensor names
        """
        return self._sensor_names

    @property
    def sensor_name2id(self) -> dict[str, int]:
        """Return a dict containing name-id pairs of the sensors in the MuJoCo model.

        :return: dict with name-id pairs
        """
        return self._sensor_name2id

    @property
    def sensor_id2name(self) -> dict[int, str]:
        """Return a dict containing id-name pairs of the sensors in the MuJoCo model.

        :return: dict with id-name pairs
        """
        return self._sensor_id2name
