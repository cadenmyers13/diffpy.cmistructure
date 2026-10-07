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

__all__ = ["constrain_as_space_group", "stru_to_parameter_set"]


def stru_to_parameter_set(name, stru):
    """Create a ParameterSet adapted to a structure object.

    The adapter is chosen from the type of `stru`. Supported types are
    diffpy.structure.Structure, pyobjcryst.crystal.Crystal,
    pyobjcryst.molecule.Molecule and cctbx.crystal.special_position_settings.

    Parameters
    ----------
    name : str
        The name to give the structure.
    stru : object
        The structure object to adapt.

    Returns
    -------
    BaseStructureParSet
        The ParameterSet adapting `stru`.

    Raises
    ------
    TypeError
        If `stru` is not one of the supported structure types.
    """
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet

    if DiffpyStructureParSet.can_adapt(stru):
        return DiffpyStructureParSet(name, stru)

    from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet

    if ObjCrystCrystalParSet.can_adapt(stru):
        return ObjCrystCrystalParSet(name, stru)

    from diffpy.cmistructure.objcrystparset import ObjCrystMoleculeParSet

    if ObjCrystMoleculeParSet.can_adapt(stru):
        return ObjCrystMoleculeParSet(name, stru)

    from diffpy.cmistructure.cctbxparset import CCTBXCrystalParSet

    if CCTBXCrystalParSet.can_adapt(stru):
        return CCTBXCrystalParSet(name, stru)

    raise TypeError("Unadaptable structure format")


# silence the pyflakes syntax checker
assert __version__ or True

# End of file
