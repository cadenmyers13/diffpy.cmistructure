#!/usr/bin/env python
##############################################################################
#
# (c) 2010 The Trustees of Columbia University in the City of New York.
# (c) 2026 Contributors to diffpy.cmistructure.
# All rights reserved.
#
# File coded by: Pavol Juhas and members of the diffpy community.
#
# Originally developed in diffpy.srfit by the DANSE Diffraction group and
# Simon J. L. Billinge.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmistructure/graphs/contributors
#
# See LICENSE.rst and LICENSE_DANSE.rst for license information.
#
##############################################################################
"""Tests space group constraints."""

import re

import numpy
import pytest
from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.scatteringpower import ScatteringPowerAtom

from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet
from diffpy.cmistructure.sgconstraints import constrain_as_space_group
from diffpy.srfit.fitbase.fitrecipe import FitRecipe
from diffpy.structure import Structure
from diffpy.structure.spacegroups import get_space_group

# ----------------------------------------------------------------------------


def test_ObjCryst_constrain_space_group():
    """Make sure that all Parameters are constrained properly.

    This tests constrainSpaceGroup from
    diffpy.cmistructure.sgconstraints, which is performed automatically
    when an ObjCrystCrystalParSet is created.
    """
    pi = numpy.pi

    occryst = makeLaMnO3()
    structure = ObjCrystCrystalParSet(occryst.GetName(), occryst)
    # Make sure we actually create the constraints
    structure._constrain_space_group()
    # Make the space group parameters individually
    structure.sgpars.latpars
    structure.sgpars.xyzpars
    structure.sgpars.adppars

    # C1: The orthorhombic lattice of LaMnO3 in P b n m.
    # Expected: The angles are fixed at pi / 2, the lengths are free, and
    # no constraint equations are needed.
    lattice = structure.get_lattice()
    lattice_names = ["a", "b", "c", "alpha", "beta", "gamma"]
    actual_lattice_const = {
        name: getattr(lattice, name).const for name in lattice_names
    }
    expected_lattice_const = {
        "a": False,
        "b": False,
        "c": False,
        "alpha": True,
        "beta": True,
        "gamma": True,
    }
    assert actual_lattice_const == expected_lattice_const
    actual_angles = [
        lattice.alpha.get_value(),
        lattice.beta.get_value(),
        lattice.gamma.get_value(),
    ]
    expected_angles = [pi / 2, pi / 2, pi / 2]
    assert actual_angles == expected_angles
    actual_lattice_constraint_count = len(lattice._constraints)
    expected_lattice_constraint_count = 0
    assert actual_lattice_constraint_count == expected_lattice_constraint_count

    # C2: The scatterers of LaMnO3 on their P b n m sites.
    # Expected: Coordinates on special positions are fixed, the rest are
    # free, and no constraint equations are needed.
    scatterers = structure.get_scatterers()
    la, mn, o1, o2 = scatterers
    actual_xyz_const = {
        "La1": [la.x.const, la.y.const, la.z.const],
        "Mn1": [mn.x.const, mn.y.const, mn.z.const],
        "O1": [o1.x.const, o1.y.const, o1.z.const],
        "O2": [o2.x.const, o2.y.const, o2.z.const],
    }
    expected_xyz_const = {
        "La1": [False, False, True],
        "Mn1": [True, True, True],
        "O1": [False, False, True],
        "O2": [False, False, False],
    }
    assert actual_xyz_const == expected_xyz_const
    actual_constraint_counts = [len(s._constraints) for s in scatterers]
    expected_constraint_counts = [0, 0, 0, 0]
    assert actual_constraint_counts == expected_constraint_counts

    # C3: Fixed coordinates are constrained or made into variables.
    # Expected: A ValueError is raised.
    with pytest.raises(ValueError):
        mn.add_constraint(mn.x, "y")

    with pytest.raises(ValueError):
        mn.add_constraint(mn.y, "z")

    with pytest.raises(ValueError):
        mn.add_constraint(mn.z, "x")

    # Nor can we make them into variables

    f = FitRecipe()
    with pytest.raises(ValueError):
        f.add_variable(mn.x)

    return


def test_DiffPy_constrain_as_space_group(datafile):
    """Test the constrain_as_space_group function."""
    structure = makeLaMnO3_P1(datafile)
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)

    space_group_parameters = constrain_as_space_group(
        parameter_set,
        "P b n m",
        scatterers=parameter_set.get_scatterers()[::2],
        constrainadps=True,
    )

    # C1: The space group Parameters are created.
    # Expected: Every Parameter exists and has a value.
    actual_unset_parameters = [
        parameter
        for parameter in space_group_parameters
        if parameter is None or parameter.get_value() is None
    ]
    expected_unset_parameters = []
    assert actual_unset_parameters == expected_unset_parameters

    # C2: Scatterers that were not passed to constrain_as_space_group.
    # Expected: Their positions and ADPs are free and unconstrained.
    unconstrained = parameter_set.get_scatterers()[1::2]
    actual_free_const = {
        scatterer.name: [
            scatterer.x.const,
            scatterer.y.const,
            scatterer.z.const,
            scatterer.U11.const,
            scatterer.U22.const,
            scatterer.U33.const,
            scatterer.U12.const,
            scatterer.U13.const,
            scatterer.U23.const,
        ]
        for scatterer in unconstrained
    }
    expected_free_const = {
        scatterer.name: [False] * 9 for scatterer in unconstrained
    }
    assert actual_free_const == expected_free_const
    actual_free_constraint_counts = {
        scatterer.name: len(scatterer._constraints)
        for scatterer in unconstrained
    }
    expected_free_constraint_counts = {
        scatterer.name: 0 for scatterer in unconstrained
    }
    assert actual_free_constraint_counts == expected_free_constraint_counts

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

    # C3: Scatterers that were passed to constrain_as_space_group.
    # Expected: At least one position and one ADP Parameter of each is
    # fixed, constrained or proxied by a space group Parameter.
    constrained = parameter_set.get_scatterers()[::2]
    actual_restricted = {
        scatterer.name: [
            any(
                _alltests(parameter)
                for parameter in [scatterer.x, scatterer.y, scatterer.z]
            ),
            any(
                _alltests(parameter)
                for parameter in [
                    scatterer.U11,
                    scatterer.U22,
                    scatterer.U33,
                    scatterer.U12,
                    scatterer.U13,
                    scatterer.U23,
                ]
            ),
        ]
        for scatterer in constrained
    }
    expected_restricted = {
        scatterer.name: [True, True] for scatterer in constrained
    }
    assert actual_restricted == expected_restricted

    return


def test_constrain_as_space_group_args(datafile):
    """Test the arguments processing of constrain_as_space_group
    function."""
    # C1: The space group is given as a symbol or as a SpaceGroup object.
    # Expected: Both create the same space group Parameters.
    structure = makeLaMnO3_P1(datafile)
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)
    symbol_parameters = constrain_as_space_group(parameter_set, "P b n m")
    space_group = get_space_group("P b n m")
    object_parameter_set = DiffpyStructureParSet(
        "LMO", makeLaMnO3_P1(datafile)
    )
    object_parameters = constrain_as_space_group(
        object_parameter_set, space_group
    )
    list(symbol_parameters)
    list(object_parameters)
    actual_names = symbol_parameters.names
    expected_names = object_parameters.names
    assert actual_names == expected_names
    return


def test_constrain_as_space_group_generator(lamno3_generator):
    # Test that constrain_as_space_group accepts a generator in place of
    # its structure parameter set.
    # C1: A generator holding the P1 LaMnO3 parameter set as .phase is
    # constrained to the orthorhombic space group P b n m.
    # Expected: Only the lattice lengths a, b and c of its .phase are left
    # free.
    space_group_parameters = constrain_as_space_group(
        lamno3_generator, "P b n m"
    )
    actual_free_lattice_names = [
        parameter.name for parameter in space_group_parameters.latpars
    ]
    expected_free_lattice_names = ["a", "b", "c"]
    assert actual_free_lattice_names == expected_free_lattice_names


@pytest.mark.parametrize(
    "input_phase, expected_type_name",
    [
        # Test that constrain_as_space_group rejects objects that are
        # neither a structure parameter set nor a generator holding one.
        # C1: A diffpy.structure Structure is passed instead of its
        # parameter set.
        # Expected: A TypeError names the Structure type and explains what
        # to pass instead.
        (Structure(), "Structure"),
        # C2: A file name is passed instead of a parameter set.
        # Expected: A TypeError names the str type and explains what to
        # pass instead.
        ("LaMnO3.stru", "str"),
    ],
)
def test_constrain_as_space_group_bad_phase(input_phase, expected_type_name):

    expected_msg = (
        "Expected a structure parameter set or a generator with a .phase "
        f"attribute, got {expected_type_name}. "
        "Please pass generator.phase or the generator itself."
    )
    with pytest.raises(TypeError, match=re.escape(expected_msg)):
        constrain_as_space_group(input_phase, "P b n m")


# The space group containers keep diffpy.srfit's names, since user scripts
# and parameter paths such as "sgpars.latpars.a" rely on them.
@pytest.mark.parametrize(
    "input_container_attribute, expected_container_name",
    [
        # C1: A container of free Parameters is read from the
        # SpaceGroupParameters, 1. latpars, 2. xyzpars, 3. adppars.
        # Expected: The container is named after the attribute it is read
        # from.
        ("latpars", "latpars"),
        ("xyzpars", "xyzpars"),
        ("adppars", "adppars"),
    ],
)
def test_space_group_parameter_container_names(
    datafile, input_container_attribute, expected_container_name
):

    parameter_set = DiffpyStructureParSet("LaMnO3", makeLaMnO3_P1(datafile))
    space_group_parameters = constrain_as_space_group(parameter_set, "P b n m")
    container = getattr(space_group_parameters, input_container_attribute)
    actual_container_name = container.name
    assert actual_container_name == expected_container_name


def test_objcryst_space_group_parameters_name():
    # C1: The space group Parameters are read from an ObjCrystCrystalParSet.
    # Expected: They are available as .sgpars, in a container named
    # "sgpars".

    occryst = makeLaMnO3()
    structure = ObjCrystCrystalParSet(occryst.GetName(), occryst)
    actual_container_name = structure.sgpars.name
    expected_container_name = "sgpars"
    assert actual_container_name == expected_container_name


def makeLaMnO3_P1(datafile):

    structure = Structure()
    structure.read(datafile("LaMnO3.stru"))
    return structure


def makeLaMnO3():

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
