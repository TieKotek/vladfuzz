"""Shared agent lifecycle helpers.

Backends can implement ``reset_episode_state`` to clear per-scenario state while
keeping heavyweight model weights resident in memory.
"""


def reset_agent_episode_state(agent) -> None:
    """Reset per-scenario agent state if the backend exposes the standard hook."""
    reset_hook = getattr(agent, "reset_episode_state", None)
    if callable(reset_hook):
        reset_hook()
