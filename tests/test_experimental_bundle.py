import ast
import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from hvaclab.submission.experimental import SUPPORTED, assemble_variant
from hvaclab.submission.build import (
    BASELINE_CELL, NOTEBOOK_CELL, assemble, build_notebook,
)
from test_heat_delivery import building, observation


class ExperimentalBundleTest(unittest.TestCase):
    def test_bundles_are_self_contained_and_preserve_policy_actions(self):
        for key in SUPPORTED:
            with self.subTest(variant=key):
                source, _ = assemble_variant(key)
                template = NOTEBOOK_CELL[NOTEBOOK_CELL.index("import sys"):]
                notebook = build_notebook(source, "", "", "", template=template)
                compile("".join(notebook["cells"][0]["source"]), "candidate.ipynb", "exec")
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith(
                            ("smart_buildings", "hvaclab")))
                namespace = {}
                exec(compile(source, "candidate.py", "exec"), namespace)
                meta = building()
                meta["observation_field_names"].append("ahs_1_outside_air_temperature_sensor")
                assembled = namespace["FinalPolicy"](meta, None)
                reference = resolve_policy(key)(meta, None)
                for outside, temps in [(231, [292, 296, 296, 296]),
                                       (280, [292, 296, 296, 296]),
                                       (310, [299] * 4), (280, [295.5] * 4)]:
                    obs = np.append(observation(temps), outside)
                    left, right = assembled.get_action(obs), reference.get_action(obs)
                    for field in left:
                        np.testing.assert_array_equal(left[field], right[field])
                for hour in (23, 5, 6):
                    obs = np.append(observation([293.5] * 4, hour=hour,
                                                comfort=hour == 6), 280.0)
                    left, right = assembled.get_action(obs), reference.get_action(obs)
                    for field in left:
                        np.testing.assert_array_equal(left[field], right[field])

    def test_default_and_calibration_notebooks_compile_as_complete_cells(self):
        for template in (NOTEBOOK_CELL, BASELINE_CELL):
            notebook = build_notebook(assemble(), "test", "", "", template=template)
            code = "".join(notebook["cells"][0]["source"])
            compile(code, "submission.ipynb", "exec")
            self.assertTrue(code.startswith("from __future__ import annotations"))


if __name__ == "__main__":
    unittest.main()
