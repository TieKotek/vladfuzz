"""Classification and signaling for simulator infrastructure failures."""

CARLA_INFRASTRUCTURE_EXIT_CODE = 88


class CarlaInfrastructureError(RuntimeError):
    """A CARLA failure that invalidates the current experiment attempt."""


def _exception_chain(exc):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_carla_infrastructure_error(exc):
    """Return whether an exception indicates an unavailable CARLA simulator."""
    class_names = {"TimeoutException", "SensorReceivedNoData"}
    message_fragments = (
        "while waiting for the simulator",
        "simulator is ready and connected",
        "a sensor took too long to send their data",
        "failed to connect to carla",
        "connection refused",
        "connection reset by peer",
        "rpc error",
    )
    for error in _exception_chain(exc):
        if error.__class__.__name__ in class_names:
            return True
        message = str(error).lower()
        if any(fragment in message for fragment in message_fragments):
            return True
    return False
