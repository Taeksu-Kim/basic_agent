"""Make ``import agent`` resolve whether this folder is used as a package inside
a project (parent already on ``sys.path``) or checked out standalone as the git
repo root. Put the parent of this folder on the path so ``agent.*`` imports work
when running ``pytest`` from inside the repo. Requires the folder be named ``agent``.
"""

import pathlib
import sys

_parent = str(pathlib.Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
