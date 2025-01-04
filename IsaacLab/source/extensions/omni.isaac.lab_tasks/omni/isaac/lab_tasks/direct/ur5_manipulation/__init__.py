import gymnasium as gym
from . import ur5_manipulation
from .ur5_reach.ur5_reach_env import UR5ReachEnv, UR5ReachEnvCfg
from .ur5_reach import reach_agents
from .ur5_push.ur5_push_env import UR5PushEnv, UR5PushEnvCfg
from .ur5_push import push_agents
from .ur5_pick.ur5_pick_env import UR5PickEnv, UR5PickEnvCfg
from .ur5_pick import pick_agents
from .ur5_lift.ur5_lift_env import UR5LiftEnv, UR5LiftEnvCfg
from .ur5_lift import lift_agents
from .reward_utils import *
##
# Register Gym environments.
##
gym.register(
    id="Isaac-UR5-Reach-v0",
    entry_point="omni.isaac.lab_tasks.direct.ur5_manipulation.ur5_reach.ur5_reach_env:UR5ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5ReachEnvCfg,
        "rl_games_cfg_entry_point": f"{reach_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{reach_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{reach_agents.__name__}.rsl_rl_ppo_cfg:ur5PPORunnerCfg"
    },
)

gym.register(
    id="Isaac-UR5-Push-v0",
    entry_point="omni.isaac.lab_tasks.direct.ur5_manipulation.ur5_push.ur5_push_env:UR5PushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5PushEnvCfg,
        "rl_games_cfg_entry_point": f"{push_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{push_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{push_agents.__name__}.rsl_rl_ppo_cfg:ur5PPORunnerCfg"
    },
)

gym.register(
    id="Isaac-UR5-Pick-v0",
    entry_point="omni.isaac.lab_tasks.direct.ur5_manipulation.ur5_pick.ur5_pick_env:UR5PickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5PickEnvCfg,
        "rl_games_cfg_entry_point": f"{pick_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{pick_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{pick_agents.__name__}.rsl_rl_ppo_cfg:ur5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-UR5-Lift-v0",
    entry_point="omni.isaac.lab_tasks.direct.ur5_manipulation.ur5_lift.ur5_lift_env:UR5LiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5LiftEnvCfg,
        "rl_games_cfg_entry_point": f"{lift_agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{lift_agents.__name__}:sb3_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{lift_agents.__name__}.rsl_rl_ppo_cfg:ur5PPORunnerCfg"
    },
)

