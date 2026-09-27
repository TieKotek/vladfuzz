# BEVDriver Runtime Source

This directory contains runtime code derived from
<https://github.com/Intelligent-Vehicles-Lab-HM/BEVDriver>, distributed under
Apache-2.0. The license text is included at
`../../LICENSES/Apache-2.0.txt`.

VLAD-Fuzz package-qualifies imports, selects the backend through the shared
model registry, resolves weights through `models/`, and adapts the agent to the
shared CARLA scenario runner. Model weights are not included.

The runtime snapshot entered the research repository in local commit
`0f49a1c49c12a3cb85e660be9d6befcf305d1365` and received integration changes in
`c3b6a68e11c4f6063c53a70abcd3ca904c72bb84`.

Before public release, record the exact matching upstream BEVDriver revision
and retain any upstream NOTICE file required by that revision.
