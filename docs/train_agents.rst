Train Agents
============

You can train your agents using either your environment or an environment contained in the GymPR library. 
Since GymPR is designed, such that all environments follow standard RL APIs, it is possible to use common RL libraries,
which contain implementations of RL algorithms, such as `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_  
or `Tianshou <https://tianshou.org/en/stable/>`_. Since RL typically requires the management of a large number of hyper-parameters, we recommend
the use of frameworks such as `Hydra <https://hydra.cc/>`_ or `hydra-zen <https://mit-ll-responsible-ai.github.io/hydra-zen/>`_ to configure complex 
training and evaluation scenarios.

Training with Stable-Baselines3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following example shows how to train an agent using Stable-Baselines3. To use the example, please install Stable-Baselines3 as 
described in the `documentation <https://stable-baselines3.readthedocs.io/en/master/guide/install.html>`_.

.. note::
    This is a simplified example that is not guaranteed to converge, as the default parameters are used. In addition, this example is not meant to 
    show the use of Hydra or hydra-zen.


.. code-block:: python

    import gymnasium as gym
    from stable_baselines3 import SAC, HerReplayBuffer

    env = gym.make('BenchmarkPushingEnv-v0')
    # copy_info_dict=True, as information about collisions is stored in the info dictionary to avoid 
    # computationally  expensive collision checking calculations when the data is relabeled (HER)
    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'copy_info_dict': True} 
    )
    model.learn(total_timesteps=int(1e6))

