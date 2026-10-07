#!/usr/bin/env python
##############################################################################
#
# (c) 2026 Contributors to diffpy.cmistructure.
# All rights reserved.
#
# File coded by: Members of the diffpy community.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmistructure/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Modules and classes that adapt structure representations to the
ParameterSet interface and automatic structure constraint generation
from space group information."""

from diffpy.cmistructure.sgconstraints import constrain_as_space_group

# package version
from diffpy.cmistructure.version import __version__  # noqa

__all__ = ["constrain_as_space_group", "struToParameterSet"]


def struToParameterSet(name, stru):
    """Creates a ParameterSet from an structure.

    This returns a ParameterSet adapted for the structure depending on its
    type.

    Parameters
    ----------
    stru
        a structure object known by this module
    name
        A name to give the structure.

    Raises TypeError if stru cannot be adapted
    """
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet

    if DiffpyStructureParSet.canAdapt(stru):
        return DiffpyStructureParSet(name, stru)

    from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet

    if ObjCrystCrystalParSet.canAdapt(stru):
        return ObjCrystCrystalParSet(name, stru)

    from diffpy.cmistructure.objcrystparset import ObjCrystMoleculeParSet

    if ObjCrystMoleculeParSet.canAdapt(stru):
        return ObjCrystMoleculeParSet(name, stru)

    from diffpy.cmistructure.cctbxparset import CCTBXCrystalParSet

    if CCTBXCrystalParSet.canAdapt(stru):
        return CCTBXCrystalParSet(name, stru)

    raise TypeError("Unadaptable structure format")


# silence the pyflakes syntax checker
assert __version__ or True

# End of file
