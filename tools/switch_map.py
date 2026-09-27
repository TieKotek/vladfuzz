import argparse
import os

import carla


def main() -> None:
    parser = argparse.ArgumentParser(description="List or load CARLA maps.")
    parser.add_argument("--map", dest="map_name", help="Map to load, for example Town05.")
    parser.add_argument("--list", action="store_true", help="List available maps and exit.")
    parser.add_argument(
        "--freeze-traffic-lights",
        action="store_true",
        help="Set traffic lights to green and freeze them after loading the map.",
    )
    parser.add_argument("--host", default=os.getenv("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CARLA_PORT", "2000")))
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    if args.list:
        for map_name in client.get_available_maps():
            print(map_name)
        return
    if not args.map_name:
        parser.error("--map is required unless --list is used")

    world = client.load_world(args.map_name)
    if args.freeze_traffic_lights:
        for traffic_light in world.get_actors().filter("traffic.traffic_light*"):
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(1000.0)
            traffic_light.freeze(True)


if __name__ == "__main__":
    main()
