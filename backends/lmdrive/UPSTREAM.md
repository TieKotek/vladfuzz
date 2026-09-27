# LMDrive Runtime Source

This directory contains runtime code derived from
<https://github.com/opendilab/LMDrive>, distributed under Apache-2.0. The
license text is included at `../../LICENSES/Apache-2.0.txt`.

VLAD-Fuzz package-qualifies imports, selects the backend through the shared
model registry, resolves weights through `models/`, and supports repeated
closed-loop executions under the shared CARLA scenario runner. Model weights
are not included.

The runtime snapshot entered the research repository in local commit
`0f49a1c49c12a3cb85e660be9d6befcf305d1365`.

Before public release, record the exact matching upstream LMDrive revision and
retain any upstream NOTICE file required by that revision.
