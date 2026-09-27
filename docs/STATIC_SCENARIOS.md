# Static Scenarios

Static scenarios define where a test task may start and finish. They do not
contain model weights or paper results. The 14 scenarios used in the paper are
frozen under `artifact_inputs/static_scenarios/`; they can also be regenerated
from CARLA using the region definitions below.

## Bundled Regions

`configs/scenario_regions.jsonl` contains one JSON object per region. The
bundled set covers intersections, T-junctions, roundabouts, and highway exits
across multiple CARLA towns.

With CARLA running, generate all bundled regions:

```bash
scripts/generate_static_scenarios.sh
```

Regenerated scenarios are written to the ignored `static_scenarios/` working
directory and do not overwrite the frozen copies.

Generate only one region:

```bash
SCENARIO_ID=town01_t_junction_1 scripts/generate_static_scenarios.sh
```

Each region produces these diagnostics under `static_scenarios/`:

```text
<region-id>.json
<region-id>_carla_bev.png
<region-id>_mpl_bev.png
```

Inspect both images before using a custom region. They reveal whether the
rectangle covers the intended roads and whether usable spawn and destination
points were verified.

## Custom Regions

Add a JSONL row with these required fields:

```json
{"id":"my_region","map_name":"Town03","center_x":0,"center_y":0,"extent_x":40,"extent_y":40}
```

`center_x` and `center_y` are CARLA world coordinates. `extent_x` and
`extent_y` are half-width and half-height in meters.

Optional fields are:

| Field | Default | Purpose |
|---|---:|---|
| `scenario_type` | `intersection` | Use `highway` for longitudinal splitting |
| `min_spawn_distance` | `8.0` | Minimum spacing on the same lane |
| `min_global_distance` | `3.0` | Minimum spacing between all candidates |
| `waypoint_sample_dist` | `1.0` | CARLA waypoint sampling interval |
| `exclude_junction` | `false` | Exclude junction waypoints when true |
| `vehicle_filter` | `vehicle.tesla.model3` | Blueprint used to verify spawnability |
| `highway_split_ratios` | `[0.5,0.0,0.5]` | Spawn, gap, and target longitudinal shares |

For a separate file, override `REGIONS`:

```bash
REGIONS=configs/my_regions.jsonl SCENARIO_ID=my_region \
  scripts/generate_static_scenarios.sh
```

CARLA must contain the named map. A custom CARLA map can be used in the same
way after it has been imported into the CARLA installation.

## Finding Region Coordinates

In windowed mode, move the spectator over the desired road and print its world
coordinate:

```bash
python tools/get_spectator.py
```

You can also list maps, switch maps, or position the spectator explicitly:

```bash
python tools/switch_map.py --list
python tools/switch_map.py --map Town03
python tools/teleport_spectator.py --x 5.2 --y -1 --z 100
```

Use the reported `x` and `y` as the new region center, choose conservative
extents, generate the region, and inspect both BEV images before fuzzing.
