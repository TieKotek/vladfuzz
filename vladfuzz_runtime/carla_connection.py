from dataclasses import dataclass
import os
from typing import Mapping, Optional


@dataclass(frozen=True)
class CarlaConnection:
    host: str
    rpc_port: int
    tm_port: Optional[int]


def _parse_port(environ: Mapping[str, str], name: str, default: int) -> int:
    raw_value = environ.get(name, str(default))
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got {port}")
    return port


def resolve_carla_connection(
    environ: Optional[Mapping[str, str]] = None,
) -> CarlaConnection:
    env = os.environ if environ is None else environ
    if env.get("CARLA_ISOLATED") != "1":
        return CarlaConnection(host="localhost", rpc_port=2000, tm_port=None)

    return CarlaConnection(
        host=env.get("CARLA_HOST", "localhost"),
        rpc_port=_parse_port(env, "CARLA_PORT", 2000),
        tm_port=_parse_port(env, "CARLA_TM_PORT", 8000),
    )
