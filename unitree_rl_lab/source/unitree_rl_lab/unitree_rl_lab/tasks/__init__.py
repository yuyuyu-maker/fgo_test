##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages.
# Skip mimic (G1 motion-tracking) so Go2 PPO can run on Isaac Lab 2.1 / Isaac Sim 4.5.
_BLACKLIST_PKGS = ["mimic"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
