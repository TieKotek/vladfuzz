import argparse
import os

import carla


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the CARLA spectator location and nearest driving waypoint."
    )
    parser.add_argument("--host", default=os.getenv("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CARLA_PORT", "2000")))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    location = world.get_spectator().get_location()
    waypoint = world.get_map().get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    print(f"Spectator location: {location}")
    print(f"Nearest driving waypoint: {waypoint.transform if waypoint else 'none'}")


if __name__ == "__main__":
    main()
