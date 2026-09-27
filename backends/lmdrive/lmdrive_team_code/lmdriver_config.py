from vladfuzz_runtime.model_assets import resolve_model_asset


class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    # Model assets live outside source packages and can be relocated as a unit.
    llm_model = str(resolve_model_asset(
        'lmdrive/llava-v1.5-7b', override_env_var='LMDRIVE_LLM_MODEL'
    ))
    preception_model = 'memfuser_baseline_e1d3_return_feature'
    preception_model_ckpt = str(resolve_model_asset(
        'lmdrive/vision-encoder-r50.pth.tar',
        override_env_var='LMDRIVE_VISION_ENCODER',
    ))
    lmdrive_ckpt = str(resolve_model_asset(
        'lmdrive/llava-v1.5-checkpoint.pth',
        override_env_var='LMDRIVE_CHECKPOINT',
    ))

    agent_use_notice = False
    sample_rate = 2

    # Stuck & Force Move
    stuck_threshold = 150
    creep_duration = 5  # Number of frames we will creep forward
    creep_throttle = 0.4

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
