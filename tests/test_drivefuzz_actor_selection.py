import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVEFUZZ_SRC = PROJECT_ROOT / "baselines" / "drivefuzz" / "src"
sys.path.insert(0, str(DRIVEFUZZ_SRC))

import constants as c
import fuzzer


class FakeLocation:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


class FakeWorld:
    def __init__(self, locations):
        self.locations = list(locations)
        self.index = 0

    def get_random_location_from_navigation(self):
        if self.index >= len(self.locations):
            return None
        location = self.locations[self.index]
        self.index += 1
        return location


class DriveFuzzActorSelectionTests(unittest.TestCase):
    def test_no_walkers_restricts_general_mutation_to_vehicles(self):
        random.seed(0)
        conf = SimpleNamespace(
            function="general",
            strategy=c.ALL,
            allow_walkers=False,
        )

        samples = [fuzzer.choose_actor_profile(conf)[0] for _ in range(100)]

        self.assertEqual(set(samples), {c.VEHICLE})

    def test_default_actor_selection_can_still_sample_walkers(self):
        random.seed(0)
        conf = SimpleNamespace(
            function="general",
            strategy=c.ALL,
            allow_walkers=True,
        )

        samples = [fuzzer.choose_actor_profile(conf)[0] for _ in range(100)]

        self.assertIn(c.VEHICLE, samples)
        self.assertIn(c.WALKER, samples)

    def test_autopilot_walker_sampler_uses_navigation_points_near_ego(self):
        ego = FakeLocation(0.0, 0.0)
        world = FakeWorld([
            FakeLocation(1.0, 0.0),     # too close to ego
            FakeLocation(20.0, 0.0),    # valid start
            FakeLocation(21.0, 0.0),    # too close to start
            FakeLocation(35.0, 0.0),    # valid destination
        ])

        start, dest = fuzzer.sample_walker_navigation_pair(world, ego, attempts=4)

        self.assertEqual((start.x, start.y), (20.0, 0.0))
        self.assertEqual((dest.x, dest.y), (35.0, 0.0))

    def test_autopilot_walker_sampler_returns_none_without_navmesh_candidate(self):
        ego = FakeLocation(0.0, 0.0)
        world = FakeWorld([FakeLocation(1000.0, 0.0)])

        start, dest = fuzzer.sample_walker_navigation_pair(world, ego, attempts=2)

        self.assertIsNone(start)
        self.assertIsNone(dest)


if __name__ == "__main__":
    unittest.main()
