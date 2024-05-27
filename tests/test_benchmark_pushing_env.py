##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

import pytest
import numpy as np
from gymnasium_planar_robotics.envs.manipulation.benchmark_pushing_env import BenchmarkPushingEnv


@pytest.mark.parametrize(
    'mover_mass, jerk, num_cycles, test_x, test_y',
    [
        (0.628, 100, 1, True, True),
        (0.628, 100, 1, True, False),
        (0.628, 100, 1, False, True),
        (1.237, 100, 1, True, True),
        (0.628, -100, 1, True, True),
        (0.628, -100, 1, True, False),
        (0.628, -100, 1, False, True),
        (1.237, -100, 1, True, True),
        (0.628, 100, 42, True, True),
        (1.237, 100, 42, True, True),
        (0.628, -100, 42, True, True),
        (1.237, -100, 42, True, True),
    ],
)
def test_jerk_actuator(mover_mass, jerk, num_cycles, test_x, test_y):
    env = BenchmarkPushingEnv(
        mover_params={'mass': mover_mass},
        std_noise=0.0,
        render_mode=None,
        num_cycles=num_cycles,
        v_max=0.01,
        a_max=0.2,
        j_max=150,
        learn_jerk=True,
    )
    mover_start_xy_pos = np.array([[0.48, 0.48]])
    env.reload_model(mover_start_xy_pos=mover_start_xy_pos)

    num_steps = 100
    dt = num_cycles
    timestep = env.model.opt.timestep

    pos_mj_actuator = np.zeros((num_steps, 2))
    velo_mj_actuator = np.zeros((num_steps, 2))
    acc_mj_actuator = np.zeros((num_steps, 2))

    pos_mj_manual = np.zeros((num_steps, 2))
    velo_mj_manual = np.zeros((num_steps, 2))
    acc_mj_manual = np.zeros((num_steps, 2))

    for step in range(0, num_steps):
        if test_x and test_y:
            jerk_arr = np.array([jerk / 2, jerk / 2])
        elif test_x and not test_y:
            jerk_arr = np.array([jerk, 0])
        elif not test_x and test_y:
            jerk_arr = np.array([0, jerk])
        else:
            jerk_arr = np.array([0, 0])

        if step > 0:
            v = velo_mj_manual[step - 1, :2].copy()
            p = pos_mj_manual[step - 1, :2].copy()
            a = acc_mj_manual[step - 1, :2].copy()
        else:
            v = np.zeros(2)
            p = mover_start_xy_pos.copy()
            a = np.zeros(2)

        for _ in range(0, dt):
            next_j = jerk_arr.copy()

            next_a, _ = env.ensure_max_dyn_val(a, env.a_max, next_j)
            v, a_tmp = env.ensure_max_dyn_val(v, env.v_max, next_a)

            a = a_tmp.copy()
            p = timestep * v + p

        # set jerk in env
        env.step(action=jerk_arr)

        pos_mj_manual[step, :2] = p.flatten().copy()
        velo_mj_manual[step, :2] = v.flatten().copy()
        acc_mj_manual[step, :2] = a.flatten().copy()

        # measure position, velocity and acceleration
        pos_mj_actuator[step, :] = env.get_mover_qpos(mover_name=env.mover_names[0], add_noise=False)[:2]
        velo_mj_actuator[step, :] = env.get_mover_qvel(mover_name=env.mover_names[0], add_noise=False)[:2]
        acc_mj_actuator[step, :] = env.get_mover_qacc(mover_name=env.mover_names[0], add_noise=False)[:2]

        norm_velo = np.linalg.norm(velo_mj_actuator[step, :], ord=2)
        norm_acc = np.linalg.norm(acc_mj_actuator[step, :], ord=2)
        assert np.allclose(norm_velo, env.v_max) or norm_velo < env.v_max
        assert np.allclose(norm_acc, env.a_max) or norm_acc < env.a_max

    assert np.allclose(pos_mj_manual, pos_mj_actuator)
    assert np.allclose(velo_mj_manual, velo_mj_actuator)
    assert np.allclose(acc_mj_manual, acc_mj_actuator)


@pytest.mark.parametrize(
    'mover_mass, acc, num_cycles, test_x, test_y',
    [
        (0.628, 0.15, 1, True, True),
        (0.628, 0.15, 1, True, False),
        (0.628, 0.15, 1, False, True),
        (1.237, 0.15, 1, True, True),
        (0.628, -0.15, 1, True, True),
        (0.628, -0.15, 1, True, False),
        (0.628, -0.15, 1, False, True),
        (1.237, -0.15, 1, True, True),
        (0.628, 0.15, 42, True, True),
        (1.237, 0.15, 42, True, True),
        (0.628, -0.15, 42, True, True),
        (1.237, -0.15, 42, True, True),
    ],
)
def test_acceleration_actuator(mover_mass, acc, num_cycles, test_x, test_y):
    env = BenchmarkPushingEnv(
        mover_params={'mass': mover_mass},
        std_noise=0.0,
        render_mode=None,
        num_cycles=num_cycles,
        v_max=0.01,
        a_max=0.2,
        j_max=150,
        learn_jerk=False,
    )
    mover_start_xy_pos = np.array([[0.48, 0.48]])
    env.reload_model(mover_start_xy_pos=mover_start_xy_pos)

    num_steps = 100
    dt = num_cycles
    timestep = env.model.opt.timestep

    pos_mj_actuator = np.zeros((num_steps, 2))
    velo_mj_actuator = np.zeros((num_steps, 2))
    acc_mj_actuator = np.zeros((num_steps, 2))

    pos_mj_manual = np.zeros((num_steps, 2))
    velo_mj_manual = np.zeros((num_steps, 2))
    acc_mj_manual = np.zeros((num_steps, 2))

    for step in range(0, num_steps):
        if test_x and test_y:
            acc_arr = np.array([acc / 2, acc / 2])
        elif test_x and not test_y:
            acc_arr = np.array([acc, 0])
        elif not test_x and test_y:
            acc_arr = np.array([0, acc])
        else:
            acc_arr = np.array([0, 0])

        if step > 0:
            v = velo_mj_manual[step - 1, :2].copy()
            p = pos_mj_manual[step - 1, :2].copy()
            a = acc_mj_manual[step - 1, :2].copy()
        else:
            v = np.zeros(2)
            p = mover_start_xy_pos.copy()
            a = np.zeros(2)

        for _ in range(0, dt):
            next_a = acc_arr.copy()
            v, a_tmp = env.ensure_max_dyn_val(v, env.v_max, next_a)

            a = a_tmp.copy()
            p = timestep * v + p

        # set acc in env
        env.step(action=acc_arr)

        pos_mj_manual[step, :2] = p.flatten().copy()
        velo_mj_manual[step, :2] = v.flatten().copy()
        acc_mj_manual[step, :2] = a.flatten().copy()

        # measure position, velocity and acceleration
        pos_mj_actuator[step, :] = env.get_mover_qpos(mover_name=env.mover_names[0], add_noise=False)[:2]
        velo_mj_actuator[step, :] = env.get_mover_qvel(mover_name=env.mover_names[0], add_noise=False)[:2]
        acc_mj_actuator[step, :] = env.get_mover_qacc(mover_name=env.mover_names[0], add_noise=False)[:2]

        norm_velo = np.linalg.norm(velo_mj_actuator[step, :], ord=2)
        norm_acc = np.linalg.norm(acc_mj_actuator[step, :], ord=2)
        assert np.allclose(norm_velo, env.v_max) or norm_velo < env.v_max
        assert np.allclose(norm_acc, env.a_max) or norm_acc < env.a_max

    assert np.allclose(pos_mj_manual, pos_mj_actuator)
    assert np.allclose(velo_mj_manual, velo_mj_actuator)
    assert np.allclose(acc_mj_manual, acc_mj_actuator)
