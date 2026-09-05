"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `unitree_rl_lab` extension. They start
with `Unitree` in their name.
"""

"""Launch Isaac Sim Simulator first."""


import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror=None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True  # noqa: R503

    for info in pkgutil.iter_modules(path, prefix):

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror)


def import_packages():
    # Only register gym IDs (package __init__.py). Env cfg modules import isaaclab.envs
    # and must not be loaded before Isaac Sim AppLauncher starts.
    sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}/source/unitree_rl_lab/unitree_rl_lab/tasks/")
    importlib.import_module("locomotion.robots.go2")
    for extra in ("locomotion.robots.h1", "locomotion.robots.g1.29dof"):
        try:
            importlib.import_module(extra)
        except Exception:
            pass
    sys.path.pop(0)


import_packages()

"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable


def main():
    """Print all environments registered in `unitree_rl_lab` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Unitree RL Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
            # add details to table
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
