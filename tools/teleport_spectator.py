import argparse
import os

import carla


def main() -> None:
    parser = argparse.ArgumentParser(description="Move the CARLA spectator to a world coordinate.")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=100.0)
    parser.add_argument("--pitch", type=float, default=-90.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--host", default=os.getenv("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CARLA_PORT", "2000")))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    spectator = client.get_world().get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=args.x, y=args.y, z=args.z),
            carla.Rotation(pitch=args.pitch, yaw=args.yaw),
        )
    )


if __name__ == "__main__":
    main()
