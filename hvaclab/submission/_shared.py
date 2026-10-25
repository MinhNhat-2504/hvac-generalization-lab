"""Re-export, so `_final_policy` is importable and checkable on its own.

In the assembled submission this module does not exist: `build.py` strips the
import and `PIDWaterResetPolicy` is already defined earlier in the same file.
Locally the import has to resolve, and it should resolve to exactly the class
the assembly will have concatenated.
"""

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

__all__ = ["PIDWaterResetPolicy"]
