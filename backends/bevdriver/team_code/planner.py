from collections import deque

import numpy as np


class RoutePlanner:
    def __init__(self, min_distance, max_distance):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111319.490945])

    def set_route(self, global_plan, gps=False):
        self.route.clear()
        for pos, cmd in global_plan:
            if gps:
                point = np.array([pos["lat"], pos["lon"]])
                point -= self.mean
                point *= self.scale
            else:
                point = np.array([pos.location.x, pos.location.y])
                point -= self.mean
            self.route.append((point, cmd))

    def run_step(self, gps):
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for index in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break
            cumulative_distance += np.linalg.norm(self.route[index][0] - self.route[index - 1][0])
            distance = np.linalg.norm(self.route[index][0] - gps)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = index

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]
