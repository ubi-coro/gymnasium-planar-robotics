##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

import pytest
import numpy as np
from gymnasium_planar_robotics.envs.planning.benchmark_planning_env import BenchmarkPlanningEnv


@pytest.mark.parametrize(
    'num_movers, mover_mass, jerk, num_cycles, test_x, test_y',
    [
        (1, 0.628, 100, 1, True, True),
        (1, 0.628, 100, 1, True, False),
        (1, 0.628, 100, 1, False, True),
        (1, 1.237, 100, 1, True, True),
        (1, 0.628, -100, 1, True, True),
        (1, 0.628, -100, 1, True, False),
        (1, 0.628, -100, 1, False, True),
        (1, 1.237, -100, 1, True, True),
        (1, 0.628, 100, 42, True, True),
        (1, 1.237, 100, 42, True, True),
        (1, 0.628, -100, 42, True, True),
        (1, 1.237, -100, 42, True, True),
        (2, 0.628, 100, 42, True, True),
        (2, 0.628, -100, 42, True, True),
        (2, np.array([0.628, 1.237]), 100, 42, True, True),
        (2, np.array([0.628, 1.237]), -100, 42, True, True),
    ],
)
def test_jerk_actuator(num_movers, mover_mass, jerk, num_cycles, test_x, test_y):
    env = BenchmarkPlanningEnv(
        layout_tiles=np.ones((9, 9)),
        num_movers=num_movers,
        show_2D_plot=False,
        mover_colors_2D_plot=['green', 'orange'],
        mover_params={'mass': mover_mass},
        std_noise=0.0,
        render_mode=None,
        num_cycles=num_cycles,
        v_max=0.01,
        a_max=0.2,
        j_max=150,
        learn_jerk=True,
    )

    if num_movers == 1:
        mover_start_xy_pos = np.array([[1.2, 1.2]])
    elif num_movers == 2:
        mover_start_xy_pos = np.array([[0.96, 0.96], [1.2, 1.2]])
    env.goals = env.np_random.uniform(low=env.min_xy_pos, high=env.max_xy_pos, size=(env.num_movers, 2))
    env.reload_model(mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=env.goals)

    num_steps = 100
    dt = num_cycles
    timestep = env.model.opt.timestep

    pos_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    velo_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    acc_mj_actuator = np.zeros((num_steps, 2 * num_movers))

    pos_mj_manual = np.zeros((num_steps, 2 * num_movers))
    velo_mj_manual = np.zeros((num_steps, 2 * num_movers))
    acc_mj_manual = np.zeros((num_steps, 2 * num_movers))

    for step in range(0, num_steps):
        if test_x and test_y:
            jerk_arr = np.array([jerk / 2, jerk / 2] * num_movers)
        elif test_x and not test_y:
            jerk_arr = np.array([jerk, 0] * num_movers)
        elif not test_x and test_y:
            jerk_arr = np.array([0, jerk] * num_movers)
        else:
            jerk_arr = np.array([0, 0] * num_movers)

        for idx_mover in range(0, num_movers):
            if step > 0:
                v = velo_mj_manual[step - 1, idx_mover : idx_mover + 2].copy()
                p = pos_mj_manual[step - 1, idx_mover : idx_mover + 2].copy()
                a = acc_mj_manual[step - 1, idx_mover : idx_mover + 2].copy()
            else:
                v = np.zeros(2)
                p = mover_start_xy_pos[idx_mover, :].copy()
                a = np.zeros(2)

            for _ in range(0, dt):
                next_j = jerk_arr[idx_mover : idx_mover + 2].copy()

                next_a, _ = env.ensure_max_dyn_val(a, env.a_max, next_j)
                v, a_tmp = env.ensure_max_dyn_val(v, env.v_max, next_a)

                a = a_tmp.copy()
                p = timestep * v + p

            pos_mj_manual[step, idx_mover : idx_mover + 2] = p.flatten().copy()
            velo_mj_manual[step, idx_mover : idx_mover + 2] = v.flatten().copy()
            acc_mj_manual[step, idx_mover : idx_mover + 2] = a.flatten().copy()

        # set jerk in env
        env.step(action=jerk_arr)

        # measure position, velocity and acceleration
        for idx_mover in range(0, num_movers):
            pos_mj_actuator[step, idx_mover : idx_mover + 2] = env.get_mover_qpos(
                mover_name=env.mover_names[idx_mover], add_noise=False
            )[:2]
            velo_mj_actuator[step, idx_mover : idx_mover + 2] = env.get_mover_qvel(
                mover_name=env.mover_names[idx_mover], add_noise=False
            )[:2]
            acc_mj_actuator[step, idx_mover : idx_mover + 2] = env.get_mover_qacc(
                mover_name=env.mover_names[idx_mover], add_noise=False
            )[:2]

            norm_velo = np.linalg.norm(velo_mj_actuator[step, idx_mover : idx_mover + 2], ord=2)
            norm_acc = np.linalg.norm(acc_mj_actuator[step, idx_mover : idx_mover + 2], ord=2)
            assert np.allclose(norm_velo, env.v_max) or float(norm_velo) < env.v_max
            assert np.allclose(norm_acc, env.a_max) or float(norm_acc) < env.a_max

    assert np.allclose(pos_mj_manual, pos_mj_actuator)
    assert np.allclose(velo_mj_manual, velo_mj_actuator)
    assert np.allclose(acc_mj_manual, acc_mj_actuator)


@pytest.mark.parametrize(
    'num_movers, mover_mass, acc, num_cycles, test_x, test_y',
    [
        (1, 0.628, 0.15, 1, True, True),
        (1, 0.628, 0.15, 1, True, False),
        (1, 0.628, 0.15, 1, False, True),
        (1, 1.237, 0.15, 1, True, True),
        (1, 0.628, -0.15, 1, True, True),
        (1, 0.628, -0.15, 1, True, False),
        (1, 0.628, -0.15, 1, False, True),
        (1, 1.237, -0.15, 1, True, True),
        (1, 0.628, 0.15, 42, True, True),
        (1, 1.237, 0.15, 42, True, True),
        (1, 0.628, -0.15, 42, True, True),
        (1, 1.237, -0.15, 42, True, True),
        (2, 0.628, 0.15, 42, True, True),
        (2, 0.628, -0.15, 42, True, True),
        (2, np.array([0.628, 1.237]), 0.15, 42, True, True),
        (2, np.array([0.628, 1.237]), -0.15, 42, True, True),
    ],
)
def test_acceleration_actuator(num_movers, mover_mass, acc, num_cycles, test_x, test_y):
    env = BenchmarkPlanningEnv(
        layout_tiles=np.ones((9, 9)),
        num_movers=num_movers,
        show_2D_plot=False,
        mover_colors_2D_plot=[],
        mover_params={'mass': mover_mass},
        std_noise=0.0,
        render_mode=None,
        num_cycles=num_cycles,
        v_max=0.01,
        a_max=0.2,
        j_max=150,
        learn_jerk=False,
    )
    if num_movers == 1:
        mover_start_xy_pos = np.array([[1.2, 1.2]])
    elif num_movers == 2:
        mover_start_xy_pos = np.array([[0.96, 0.96], [1.2, 1.2]])
    env.goals = env.np_random.uniform(low=env.min_xy_pos, high=env.max_xy_pos, size=(env.num_movers, 2))
    env.reload_model(mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=env.goals)

    num_steps = 100
    dt = num_cycles
    timestep = env.model.opt.timestep

    pos_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    velo_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    acc_mj_actuator = np.zeros((num_steps, 2 * num_movers))

    pos_mj_manual = np.zeros((num_steps, 2 * num_movers))
    velo_mj_manual = np.zeros((num_steps, 2 * num_movers))
    acc_mj_manual = np.zeros((num_steps, 2 * num_movers))

    for step in range(0, num_steps):
        if test_x and test_y:
            acc_arr = np.array([acc / 2, acc / 2] * num_movers)
        elif test_x and not test_y:
            acc_arr = np.array([acc, 0] * num_movers)
        elif not test_x and test_y:
            acc_arr = np.array([0, acc] * num_movers)
        else:
            acc_arr = np.array([0, 0] * num_movers)

        for idx_mover in range(0, num_movers):
            if step > 0:
                v = velo_mj_manual[step - 1, idx_mover : idx_mover + 2].copy()
                p = pos_mj_manual[step - 1, idx_mover : idx_mover + 2].copy()
                a = acc_mj_manual[step - 1, idx_mover : idx_mover + 2].copy()
            else:
                v = np.zeros(2)
                p = mover_start_xy_pos[idx_mover, :].copy()
                a = np.zeros(2)

            for _ in range(0, dt):
                next_a = acc_arr[idx_mover : idx_mover + 2].copy()
                v, a_tmp = env.ensure_max_dyn_val(v, env.v_max, next_a)

                a = a_tmp.copy()
                p = timestep * v + p
            pos_mj_manual[step, idx_mover : idx_mover + 2] = p.flatten().copy()
            velo_mj_manual[step, idx_mover : idx_mover + 2] = v.flatten().copy()
            acc_mj_manual[step, idx_mover : idx_mover + 2] = a.flatten().copy()

        # set acc in env
        env.step(action=acc_arr)

        # measure position, velocity and acceleration
        for idx_mover in range(0, num_movers):
            pos_mj_actuator[step, idx_mover : idx_mover + 2] = env.get_mover_qpos(
                mover_name=env.mover_names[idx_mover], add_noise=False
            )[:2]
            velo_mj_actuator[step, idx_mover : idx_mover + 2] = env.get_mover_qvel(
                mover_name=env.mover_names[idx_mover], add_noise=False
            )[:2]
            acc_mj_actuator[step, idx_mover : idx_mover + 2] = env.get_mover_qacc(
                mover_name=env.mover_names[idx_mover], add_noise=False
            )[:2]

            norm_velo = np.linalg.norm(velo_mj_actuator[step, idx_mover : idx_mover + 2], ord=2)
            norm_acc = np.linalg.norm(acc_mj_actuator[step, idx_mover : idx_mover + 2], ord=2)
            assert np.allclose(norm_velo, env.v_max) or float(norm_velo) < env.v_max
            assert np.allclose(norm_acc, env.a_max) or float(norm_acc) < env.a_max

    assert np.allclose(pos_mj_manual, pos_mj_actuator)
    assert np.allclose(velo_mj_manual, velo_mj_actuator)
    assert np.allclose(acc_mj_manual, acc_mj_actuator)
