# Third-Party Components

VLAD-Fuzz integrates research code and model assets from several projects. The
clean repository does not redistribute model weights. Users must review and
accept each model's terms at its official download page.

| Component | Upstream | License status |
|---|---|---|
| LMDrive | <https://github.com/opendilab/LMDrive> | Apache-2.0 upstream |
| BEVDriver | <https://github.com/Intelligent-Vehicles-Lab-HM/BEVDriver> | Apache-2.0 upstream |
| SimLingo | <https://github.com/RenzKa/simlingo> | Apache-2.0 upstream; dataset has separate terms |
| DriveFuzz | <https://gitlab.com/s3lab-code/public/drivefuzz> | No top-level license declared upstream; redistribution permission must be confirmed before public release |
| CARLA Leaderboard | <https://github.com/carla-simulator/leaderboard> | See vendored license |
| ScenarioRunner | <https://github.com/carla-simulator/scenario_runner> | See vendored license |

The release must preserve upstream notices and identify local modifications.
SimLingo's dataset is not needed for closed-loop inference and is not included;
users who download it separately must review `LICENCE_Dataset` upstream.

The Apache-2.0 text used by the three backend source distributions is included
at `LICENSES/Apache-2.0.txt`. Backend-specific provenance and local integration
changes are recorded in each backend's `UPSTREAM.md`.

The adapted DriveFuzz runtime is included because it is necessary to reproduce
the paper's scenario-based baseline. Its provenance and modifications are
documented in `baselines/drivefuzz/UPSTREAM.md`. The licenses found in
DriveFuzz's bundled CARLA and Autoware directories do not establish a license
for DriveFuzz's own `src/`; do not publish this repository until redistribution
permission for that source has been confirmed.
