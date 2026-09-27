from vladfuzz_runtime.model_assets import resolve_bert_model, resolve_model_asset


class GlobalConfig:
    """Runtime configuration for the integrated BEVDriver backend."""

    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 20

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 20

    max_throttle = 0.75
    brake_speed = 0.1
    brake_ratio = 1.1
    clip_delta = 0.35

    stuck_threshold = 150
    creep_duration = 5
    creep_throttle = 0.4

    llm_model = str(resolve_model_asset(
        "bevdriver/llama-7b", override_env_var="BEVDRIVER_LLM_MODEL"
    ))
    bert_model = resolve_bert_model("BEVDRIVER_BERT_MODEL")
    encoder_model = "bevdriver_encoder"
    encoder_model_ckpt = str(resolve_model_asset(
        "bevdriver/bev-encoder.pth", override_env_var="BEVDRIVER_ENCODER"
    ))
    bevdriver_ckpt = str(resolve_model_asset(
        "bevdriver/bevdriver-model.pth", override_env_var="BEVDRIVER_CHECKPOINT"
    ))

    agent_use_notice = False
    sample_rate = 1

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
