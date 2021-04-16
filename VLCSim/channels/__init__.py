"""
Package containing the lib and test files for different models of optical channels

Concrete classes on this package must inherit from AbstractChannel and implement its abstract methods.

Exported classes
----------------
AbstractChannel
LOSChannel
"""

from channels.lib.abstract_channel import AbstractChannel
from channels.lib.LOS_channel import LOSChannel
