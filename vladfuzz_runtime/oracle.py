DEFAULT_ORACLE_CHECKS = {
    "collision": True,
    "stuck": True,
    "lane_invasion": True,
    "speeding": False,
    "timeout": True,
    "out_of_bounds": True,
    "other": True,
}

IGNORED_COLLISION_ACTOR_TYPES = {
    "static.road",
}


def collision_actor_type(event):
    other_actor = getattr(event, "other_actor", None)
    return getattr(other_actor, "type_id", "unknown")


def collision_impulse(event):
    impulse = getattr(event, "normal_impulse", None)
    if impulse is None:
        return None
    return {
        "x": getattr(impulse, "x", 0.0),
        "y": getattr(impulse, "y", 0.0),
        "z": getattr(impulse, "z", 0.0),
    }


def should_count_collision(event):
    return collision_actor_type(event) not in IGNORED_COLLISION_ACTOR_TYPES

ALIASES = {
    "crash": "collision",
    "collision": "collision",
    "stuck": "stuck",
    "lane": "lane_invasion",
    "lane_invasion": "lane_invasion",
    "speed": "speeding",
    "speeding": "speeding",
    "timeout": "timeout",
    "target_not_reached": "timeout",
    "not_completed": "timeout",
    "out_of_bounds": "out_of_bounds",
    "out_of_range": "out_of_bounds",
    "other": "other",
}


def parse_oracle_checks(value):
    checks = dict(DEFAULT_ORACLE_CHECKS)
    if value is None or value == "" or value == "default":
        return checks
    if value == "all":
        return {key: True for key in checks}
    if value == "none":
        return {key: False for key in checks}

    checks = {key: False for key in checks}
    for raw_name in value.split(","):
        name = raw_name.strip().lower().replace("-", "_")
        if not name:
            continue
        if name not in ALIASES:
            allowed = ", ".join(sorted(ALIASES))
            raise ValueError(f"Unknown oracle check '{raw_name}'. Allowed values: {allowed}, all, default, none")
        checks[ALIASES[name]] = True
    return checks
