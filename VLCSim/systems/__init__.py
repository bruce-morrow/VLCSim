"""
Package containing the lib and test files for different VLC systems configurations.

Concrete classes on this package must inherit from AbstractSystem and implement its abstract methods.

Exported classes
----------------
AbstractSystem
ReceiverOnPlaneSystem
"""

from systems.lib.abstract_system import AbstractSystem
from systems.lib.receiver_on_plane import ReceiverOnPlaneSystem
