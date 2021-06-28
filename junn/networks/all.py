"""Helper module to import all packaged networks."""
from .LinkNet import LinkNet
from .MSD import MSD
from .Unet import Unet

# reference them somehow

__networks__ = [Unet, MSD, LinkNet]
