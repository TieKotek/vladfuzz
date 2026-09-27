# SimLingo Runtime Source

This directory contains the SimLingo runtime integration derived from
<https://github.com/RenzKa/simlingo>, which is distributed under Apache-2.0.

VLAD-Fuzz changes package-qualify imports, resolve model assets through the
repository-level `models/` tree, add reusable episode reset behavior, and add
runtime diagnostics required by the shared CARLA execution framework. Model
weights and the separately licensed SimLingo dataset are not included.

The integration entered the research repository in local commit
`0f49a1c49c12a3cb85e660be9d6befcf305d1365` and received runtime stability
changes in `c3b6a68e11c4f6063c53a70abcd3ca904c72bb84`. Before public release, record
the exact matching upstream SimLingo revision in this file.
