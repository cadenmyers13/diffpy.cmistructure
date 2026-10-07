#!/usr/bin/env python
##############################################################################
#
# diffpy.srfit      by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Tests space group constraints."""

import unittest

import numpy
import pytest

# ----------------------------------------------------------------------------


def test_ObjCryst_constrain_space_group():
    """Make sure that all Parameters are constrained properly.

    This tests constrainSpaceGroup from
    diffpy.cmistructure.sgconstraints, which is performed automatically
    when an ObjCrystCrystalParSet is created.
    """
    from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet

    pi = numpy.pi

    occryst = makeLaMnO3()
    structure = ObjCrystCrystalParSet(occryst.GetName(), occryst)
    # Make sure we actually create the constraints
    structure._constrain_space_group()
    # Make the space group parameters individually
    structure.space_group_parameters.lattice_parameters
    structure.space_group_parameters.xyz_parameters
    structure.space_group_parameters.adp_parameters

    # Check the orthorhombic lattice
    lattice = structure.get_lattice()
    assert lattice.alpha.const
    assert lattice.beta.const
    assert lattice.gamma.const
    assert pi / 2 == lattice.alpha.get_value()
    assert pi / 2 == lattice.beta.get_value()
    assert pi / 2 == lattice.gamma.get_value()

    assert not lattice.a.const
    assert not lattice.b.const
    assert not lattice.c.const
    assert 0 == len(lattice._constraints)

    # Now make sure the scatterers are constrained properly
    scatterers = structure.get_scatterers()
    la = scatterers[0]
    assert not la.x.const
    assert not la.y.const
    assert la.z.const
    assert 0 == len(la._constraints)

    mn = scatterers[1]
    assert mn.x.const
    assert mn.y.const
    assert mn.z.const
    assert 0 == len(mn._constraints)

    o1 = scatterers[2]
    assert not o1.x.const
    assert not o1.y.const
    assert o1.z.const
    assert 0 == len(o1._constraints)

    o2 = scatterers[3]
    assert not o2.x.const
    assert not o2.y.const
    assert not o2.z.const
    assert 0 == len(o2._constraints)

    # Make sure we can't constrain these
    with pytest.raises(ValueError):
        mn.add_constraint(mn.x, "y")

    with pytest.raises(ValueError):
        mn.add_constraint(mn.y, "z")

    with pytest.raises(ValueError):
        mn.add_constraint(mn.z, "x")

    # Nor can we make them into variables
    from diffpy.srfit.fitbase.fitrecipe import FitRecipe

    f = FitRecipe()
    with pytest.raises(ValueError):
        f.add_variable(mn.x)

    return


def test_DiffPy_constrain_as_space_group(datafile):
    """Test the constrain_as_space_group function."""
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
    from diffpy.cmistructure.sgconstraints import constrain_as_space_group

    structure = makeLaMnO3_P1(datafile)
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)

    space_group_parameters = constrain_as_space_group(
        parameter_set,
        "P b n m",
        scatterers=parameter_set.get_scatterers()[::2],
        constrainadps=True,
    )

    # Make sure that the new parameters were created
    for parameter in space_group_parameters:
        assert parameter is not None
        assert parameter.get_value() is not None

    # Test the unconstrained atoms
    for scatterer in parameter_set.get_scatterers()[1::2]:
        assert not scatterer.x.const
        assert not scatterer.y.const
        assert not scatterer.z.const
        assert not scatterer.U11.const
        assert not scatterer.U22.const
        assert not scatterer.U33.const
        assert not scatterer.U12.const
        assert not scatterer.U13.const
        assert not scatterer.U23.const
        assert 0 == len(scatterer._constraints)

    proxied = [p.par for p in space_group_parameters]

    def _consttest(parameter):
        return parameter.const

    def _constrainedtest(parameter):
        return parameter.constrained

    def _proxytest(parameter):
        return parameter in proxied

    def _alltests(parameter):
        return (
            _consttest(parameter)
            or _constrainedtest(parameter)
            or _proxytest(parameter)
        )

    for index, scatterer in enumerate(parameter_set.get_scatterers()[::2]):
        # Under this scheme, atom 6 is free to vary
        test = False
        for parameter in [scatterer.x, scatterer.y, scatterer.z]:
            test |= _alltests(parameter)
        assert test

        test = False
        for parameter in [
            scatterer.U11,
            scatterer.U22,
            scatterer.U33,
            scatterer.U12,
            scatterer.U13,
            scatterer.U23,
        ]:
            test |= _alltests(parameter)

        assert test

    return


def test_constrain_as_space_group_args(datafile):
    """Test the arguments processing of constrain_as_space_group
    function."""
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
    from diffpy.cmistructure.sgconstraints import constrain_as_space_group
    from diffpy.structure.spacegroups import GetSpaceGroup

    structure = makeLaMnO3_P1(datafile)
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)
    space_group_parameters = constrain_as_space_group(parameter_set, "P b n m")
    space_group = GetSpaceGroup("P b n m")
    parset2 = DiffpyStructureParSet("LMO", makeLaMnO3_P1(datafile))
    sgpars2 = constrain_as_space_group(parset2, space_group)
    list(space_group_parameters)
    list(sgpars2)
    assert space_group_parameters.names == sgpars2.names
    return


def makeLaMnO3_P1(datafile):
    from diffpy.structure import Structure

    structure = Structure()
    structure.read(datafile("LaMnO3.stru"))
    return structure


def makeLaMnO3():
    from pyobjcryst.atom import Atom
    from pyobjcryst.crystal import Crystal
    from pyobjcryst.scatteringpower import ScatteringPowerAtom

    pi = numpy.pi
    # It appears that ObjCryst only supports standard symbols
    crystal = Crystal(5.486341, 5.619215, 7.628206, "P b n m")
    crystal.SetName("LaMnO3")
    # La1
    sp = ScatteringPowerAtom("La1", "La")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.996096, 0.0321494, 0.25, "La1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # Mn1
    sp = ScatteringPowerAtom("Mn1", "Mn")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0, 0.5, 0, "Mn1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # O1
    sp = ScatteringPowerAtom("O1", "O")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.0595746, 0.496164, 0.25, "O1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # O2
    sp = ScatteringPowerAtom("O2", "O")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.720052, 0.289387, 0.0311126, "O2", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)

    return crystal


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
