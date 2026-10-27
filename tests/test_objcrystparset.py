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
"""Tests for diffpy.cmistructure package."""

import numpy
import pytest
from pyobjcryst.spacegroup import SpaceGroup

from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet
from diffpy.structure import spacegroups

# ----------------------------------------------------------------------------


def test_implicit_bond_angle_restraints(c60_crystal, c60_molecule):
    """Test the structure with implicit bond angles."""
    occryst = c60_crystal
    ocmol = c60_molecule

    # Add some bond angles to the molecule
    ocmol.AddBondAngle(ocmol[0], ocmol[5], ocmol[8], 1.1, 0.1, 0.1)
    ocmol.AddBondAngle(ocmol[0], ocmol[7], ocmol[44], 1.3, 0.1, 0.1)

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60
    m.wrap_restraints()

    # C1: Two restraints are added to the molecule.
    # Expected: The molecule holds both restraints.
    actual_restraint_count = len(m._restraints)
    expected_restraint_count = 2
    assert actual_restraint_count == expected_restraint_count

    # C2: The restraint penalties are evaluated.
    # Expected: They equal the pyobjcryst log-likelihoods.
    res0, res1 = m._restraints
    actual_penalties = set([res0.penalty(), res1.penalty()])
    angles = ocmol.GetBondAngleList()
    expected_penalties = set(
        [angles[0].GetLogLikelihood(), angles[1].GetLogLikelihood()]
    )
    assert actual_penalties == expected_penalties

    return


def test_objcryst_parset(c60_crystal, c60_molecule):
    """Test the structure conversion."""
    occryst = c60_crystal
    ocmol = c60_molecule
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    actual_name = crystal.name
    expected_name = "bucky"
    assert actual_name == expected_name

    def _testCrystal():
        # Test the lattice
        actual_lattice = [
            crystal.a.value,
            crystal.b.get_value(),
            crystal.c.get_value(),
            crystal.alpha.get_value(),
            crystal.beta.get_value(),
            crystal.gamma.get_value(),
        ]
        expected_lattice = [
            occryst.a,
            occryst.b,
            occryst.c,
            occryst.alpha,
            occryst.beta,
            occryst.gamma,
        ]
        assert actual_lattice == pytest.approx(expected_lattice)
        return

    def _testMolecule():
        # Test position, occupancy and orientation
        actual_molecule = [
            m.x.get_value(),
            m.y.get_value(),
            m.z.get_value(),
            m.occ.get_value(),
            m.q0.get_value(),
            m.q1.get_value(),
            m.q2.get_value(),
            m.q3.get_value(),
        ]
        expected_molecule = [
            ocmol.X,
            ocmol.Y,
            ocmol.Z,
            ocmol.Occupancy,
            ocmol.Q0,
            ocmol.Q1,
            ocmol.Q2,
            ocmol.Q3,
        ]
        assert actual_molecule == pytest.approx(expected_molecule)

        # Check the atoms thoroughly
        actual_elements = [a.element for a in m.atoms]
        expected_elements = [
            ocmol[i].GetScatteringPower().GetSymbol()
            for i in range(len(ocmol))
        ]
        assert actual_elements == expected_elements
        actual_atoms = [
            [
                a.x.get_value(),
                a.y.get_value(),
                a.z.get_value(),
                a.occ.get_value(),
                a.Biso.get_value(),
            ]
            for a in m.atoms
        ]
        expected_atoms = [
            pytest.approx(
                [
                    ocmol[i].X,
                    ocmol[i].Y,
                    ocmol[i].Z,
                    ocmol[i].Occupancy,
                    ocmol[i].GetScatteringPower().Biso,
                ]
            )
            for i in range(len(ocmol))
        ]
        assert actual_atoms == expected_atoms
        return

    # C1: The ParameterSet has just been created from the crystal.
    # Expected: The Parameters match the pyobjcryst values.
    _testCrystal()
    _testMolecule()

    # C2: Values are changed through pyobjcryst.
    # Expected: The Parameters follow the changes.
    ocmol[0].X *= 1.1
    ocmol[0].Occupancy *= 1.1
    ocmol[0].GetScatteringPower().Biso *= 1.1
    ocmol.Q0 *= 1.1
    occryst.a *= 1.1

    _testCrystal()
    _testMolecule()

    # C3: Values are changed through the ParameterSet.
    # Expected: The pyobjcryst objects follow the changes.
    crystal.c60.C44.x.set_value(1.1)
    crystal.c60.C44.occ.set_value(1.1)
    crystal.c60.C44.Biso.set_value(1.1)
    crystal.c60.q3.set_value(1.1)
    crystal.a.set_value(1.1)

    _testCrystal()
    _testMolecule()
    return


def test_implicit_bond_length_restraints(c60_crystal, c60_molecule):
    """Test the structure with implicit bond lengths."""
    occryst = c60_crystal
    ocmol = c60_molecule

    # Add some bonds to the molecule
    ocmol.AddBond(ocmol[0], ocmol[5], 3.3, 0.1, 0.1)
    ocmol.AddBond(ocmol[0], ocmol[7], 3.3, 0.1, 0.1)

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60
    m.wrap_restraints()

    # C1: Two restraints are added to the molecule.
    # Expected: The molecule holds both restraints.
    actual_restraint_count = len(m._restraints)
    expected_restraint_count = 2
    assert actual_restraint_count == expected_restraint_count

    # C2: The restraint penalties are evaluated.
    # Expected: They equal the pyobjcryst log-likelihoods.
    res0, res1 = m._restraints
    actual_penalties = set([res0.penalty(), res1.penalty()])
    bonds = ocmol.GetBondList()
    expected_penalties = set(
        [bonds[0].GetLogLikelihood(), bonds[1].GetLogLikelihood()]
    )
    assert actual_penalties == expected_penalties

    return


def test_implicit_dihedral_angle_restraints(c60_crystal, c60_molecule):
    """Test the structure with implicit dihedral angles."""
    occryst = c60_crystal
    ocmol = c60_molecule

    # Add some bond angles to the molecule
    ocmol.AddDihedralAngle(
        ocmol[0], ocmol[5], ocmol[8], ocmol[41], 1.1, 0.1, 0.1
    )
    ocmol.AddDihedralAngle(
        ocmol[0], ocmol[7], ocmol[44], ocmol[2], 1.3, 0.1, 0.1
    )

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60
    m.wrap_restraints()

    # C1: Two restraints are added to the molecule.
    # Expected: The molecule holds both restraints.
    actual_restraint_count = len(m._restraints)
    expected_restraint_count = 2
    assert actual_restraint_count == expected_restraint_count

    # C2: The restraint penalties are evaluated.
    # Expected: They equal the pyobjcryst log-likelihoods.
    res0, res1 = m._restraints
    actual_penalties = set([res0.penalty(), res1.penalty()])
    angles = ocmol.GetDihedralAngleList()
    expected_penalties = set(
        [angles[0].GetLogLikelihood(), angles[1].GetLogLikelihood()]
    )
    assert actual_penalties == expected_penalties

    return


def test_wrap_stretch_mode_parameters(c60_crystal, c60_molecule):
    # Case: The molecule has two bond lengths and one bond angle in the
    # chain C0-C1-C2, and its stretch modes are built.
    # Expected: Each bond length and bond angle stretch mode is wrapped
    # as a Parameter named after its atoms. pyobjcryst builds one mode
    # for each side of a bond or angle, so every name appears in both
    # atom orders.
    ocmol = c60_molecule
    ocmol.AddBond(ocmol[0], ocmol[1], 1.37, 0.1, 0.1)
    ocmol.AddBond(ocmol[1], ocmol[2], 6.9, 0.1, 0.1)
    ocmol.AddBondAngle(ocmol[0], ocmol[1], ocmol[2], 1.57, 0.1, 0.1)
    ocmol.BuildStretchModeBondLength()
    ocmol.BuildStretchModeBondAngle()
    molecule = ObjCrystCrystalParSet("bucky", c60_crystal).c60
    parameter_names_before = set(molecule._parameters)
    molecule.wrap_stretch_mode_parameters()
    actual_names = sorted(set(molecule._parameters) - parameter_names_before)
    expected_names = [
        "ba_C0_C1_C2",
        "ba_C2_C1_C0",
        "bl_C0_C1",
        "bl_C1_C0",
        "bl_C1_C2",
        "bl_C2_C1",
    ]
    assert actual_names == expected_names


def test_explicit_bond_length_restraints(c60_crystal, c60_molecule):
    """Test the structure with explicit bond lengths."""
    occryst = c60_crystal
    ocmol = c60_molecule

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    # make some bond angle restraints
    res0 = m.restrain_bond_length(m.atoms[0], m.atoms[5], 3.3, 0.1, 0.1)
    res1 = m.restrain_bond_length(m.atoms[0], m.atoms[7], 3.3, 0.1, 0.1)

    # C1: Two restraints are added to the molecule.
    # Expected: The molecule holds both restraints.
    actual_restraint_count = len(m._restraints)
    expected_restraint_count = 2
    assert actual_restraint_count == expected_restraint_count

    # C2: The restraint penalties are evaluated.
    # Expected: They equal the pyobjcryst log-likelihoods.
    bonds = ocmol.GetBondList()
    actual_bond_count = len(bonds)
    expected_bond_count = 2
    assert actual_bond_count == expected_bond_count
    actual_penalties = [res0.penalty(), res1.penalty()]
    expected_penalties = [b.GetLogLikelihood() for b in bonds]
    assert actual_penalties == expected_penalties

    return


def test_show_explicit_molecule_restraints(c60_crystal, capturestdout):
    # C1: A bond length, a bond angle and a dihedral angle restraint are
    # added through the ObjCrystMoleculeParSet.
    # Expected: The Restraints section lists each restraint by its atom
    # names with its target value, sigma, delta and scaling.
    crystal = ObjCrystCrystalParSet("bucky", c60_crystal)
    molecule = crystal.c60
    atoms = molecule.atoms
    molecule.restrain_bond_length(atoms[0], atoms[5], 3.3, 0.1, 0.2)
    molecule.restrain_bond_angle(atoms[0], atoms[5], atoms[7], 1.0, 0.1, 0.2)
    molecule.restrain_dihedral_angle(
        atoms[0], atoms[5], atoms[7], atoms[8], 2.0, 0.1, 0.2
    )
    lines = capturestdout(crystal.show, textwidth=0).splitlines()
    actual_restraint_lines = set(lines[lines.index("Restraints") + 2 :])
    expected_restraint_lines = {
        "C0-C5: length = 3.300000, sigma = 0.100000, delta = 0.200000, "
        "scaled = False",
        "C0-C5-C7: angle = 1.000000, sigma = 0.100000, delta = 0.200000, "
        "scaled = False",
        "C0-C5-C7-C8: angle = 2.000000, sigma = 0.100000, "
        "delta = 0.200000, scaled = False",
    }
    assert actual_restraint_lines == expected_restraint_lines


def test_show_implicit_molecule_restraints(
    c60_crystal, c60_molecule, capturestdout
):
    # C1: A bond length, a bond angle and a dihedral angle restraint are
    # defined on the pyobjcryst Molecule and wrapped with
    # wrap_restraints.
    # Expected: The Restraints section lists each restraint by its atom
    # names with its target value, sigma, delta and scaling.
    ocmol = c60_molecule
    ocmol.AddBond(ocmol[0], ocmol[5], 3.3, 0.1, 0.2)
    ocmol.AddBondAngle(ocmol[0], ocmol[5], ocmol[7], 1.0, 0.1, 0.2)
    ocmol.AddDihedralAngle(
        ocmol[0], ocmol[5], ocmol[7], ocmol[8], 2.0, 0.1, 0.2
    )
    crystal = ObjCrystCrystalParSet("bucky", c60_crystal)
    crystal.c60.wrap_restraints()
    lines = capturestdout(crystal.show, textwidth=0).splitlines()
    actual_restraint_lines = set(lines[lines.index("Restraints") + 2 :])
    expected_restraint_lines = {
        "C0-C5: length = 3.300000, sigma = 0.100000, delta = 0.200000, "
        "scaled = False",
        "C0-C5-C7: angle = 1.000000, sigma = 0.100000, delta = 0.200000, "
        "scaled = False",
        "C0-C5-C7-C8: angle = 2.000000, sigma = 0.100000, "
        "delta = 0.200000, scaled = False",
    }
    assert actual_restraint_lines == expected_restraint_lines


def test_explicit_bond_angle_restraints(c60_crystal, c60_molecule):
    """Test the structure with explicit bond angles.

    Note that this cannot work with co-linear points as the direction of
    rotation cannot be defined in this case.
    """
    occryst = c60_crystal
    ocmol = c60_molecule

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    # restrain some bond angles
    res0 = m.restrain_bond_angle(
        m.atoms[0], m.atoms[5], m.atoms[8], 3.3, 0.1, 0.1
    )
    res1 = m.restrain_bond_angle(
        m.atoms[0], m.atoms[7], m.atoms[44], 3.3, 0.1, 0.1
    )

    # C1: Two restraints are added to the molecule.
    # Expected: The molecule holds both restraints.
    actual_restraint_count = len(m._restraints)
    expected_restraint_count = 2
    assert actual_restraint_count == expected_restraint_count

    # C2: The restraint penalties are evaluated.
    # Expected: They equal the pyobjcryst log-likelihoods.
    actual_penalties = set([res0.penalty(), res1.penalty()])
    angles = ocmol.GetBondAngleList()
    expected_penalties = set(
        [angles[0].GetLogLikelihood(), angles[1].GetLogLikelihood()]
    )
    assert actual_penalties == expected_penalties

    return


def test_explicit_dihedral_angle_restraints(c60_crystal, c60_molecule):
    """Test the structure with explicit dihedral angles."""
    occryst = c60_crystal
    ocmol = c60_molecule

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    # Restrain some dihedral angles.
    res0 = m.restrain_dihedral_angle(
        m.atoms[0], m.atoms[5], m.atoms[8], m.atoms[41], 1.1, 0.1, 0.1
    )
    res1 = m.restrain_dihedral_angle(
        m.atoms[0], m.atoms[7], m.atoms[44], m.atoms[2], 1.1, 0.1, 0.1
    )

    # C1: Two restraints are added to the molecule.
    # Expected: The molecule holds both restraints.
    actual_restraint_count = len(m._restraints)
    expected_restraint_count = 2
    assert actual_restraint_count == expected_restraint_count

    # C2: The restraint penalties are evaluated.
    # Expected: They equal the pyobjcryst log-likelihoods.
    actual_penalties = set([res0.penalty(), res1.penalty()])
    angles = ocmol.GetDihedralAngleList()
    expected_penalties = set(
        [angles[0].GetLogLikelihood(), angles[1].GetLogLikelihood()]
    )
    assert actual_penalties == expected_penalties

    return


def test_explicit_bond_length_parameter(c60_crystal):
    """Test adding bond length parameters to the molecule."""
    occryst = c60_crystal

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    a0 = m.atoms[0]
    a7 = m.atoms[7]
    a20 = m.atoms[20]

    # Add a parameter
    p1 = m.add_bond_length_parameter("C07", a0, a7)
    # Have another atom tag along for the ride
    p1.add_atoms([a20])

    xyz0 = numpy.array([a0.x.get_value(), a0.y.get_value(), a0.z.get_value()])
    xyz7 = numpy.array([a7.x.get_value(), a7.y.get_value(), a7.z.get_value()])
    xyz20 = numpy.array(
        [a20.x.get_value(), a20.y.get_value(), a20.z.get_value()]
    )

    dd = xyz0 - xyz7
    d0 = numpy.dot(dd, dd) ** 0.5
    # C1: A bond length Parameter is added.
    # Expected: Its value is the current bond length.
    actual_length = p1.get_value()
    expected_length = d0
    assert actual_length == pytest.approx(expected_length, abs=1e-6)

    # Record the unit direction of change for later
    u = dd / d0

    # C2: The bond length Parameter is stretched by 5%.
    # Expected: The Parameter and the measured bond length both take
    # the new value, the first atom stays put, and the second and
    # tag-along atoms move along the bond.
    scale = 1.05
    p1.set_value(scale * d0)

    actual_length = p1.get_value()
    expected_length = scale * d0
    assert actual_length == pytest.approx(expected_length, abs=1e-6)

    xyz0a = numpy.array([a0.x.get_value(), a0.y.get_value(), a0.z.get_value()])
    xyz7a = numpy.array([a7.x.get_value(), a7.y.get_value(), a7.z.get_value()])
    xyz20a = numpy.array(
        [a20.x.get_value(), a20.y.get_value(), a20.z.get_value()]
    )

    dda = xyz0a - xyz7a
    d1 = numpy.dot(dda, dda) ** 0.5

    actual_measured_length = d1
    expected_measured_length = scale * d0
    assert actual_measured_length == pytest.approx(
        expected_measured_length, abs=1e-6
    )

    actual_xyz0 = xyz0a.tolist()
    expected_xyz0 = xyz0.tolist()
    assert actual_xyz0 == expected_xyz0

    actual_xyz7 = xyz7a
    expected_xyz7 = xyz7 + (1 - scale) * d0 * u
    assert actual_xyz7 == pytest.approx(expected_xyz7, abs=1e-5)

    actual_xyz20 = xyz20a
    expected_xyz20 = xyz20 + (1 - scale) * d0 * u
    assert actual_xyz20 == pytest.approx(expected_xyz20, abs=1e-6)

    return


def test_explicit_bond_angle_parameter(c60_crystal):
    """Test adding bond angle parameters to the molecule."""
    occryst = c60_crystal

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    a0 = m.atoms[0]
    a7 = m.atoms[7]
    a20 = m.atoms[20]
    a25 = m.atoms[25]

    xyz0 = numpy.array([a0.x.get_value(), a0.y.get_value(), a0.z.get_value()])
    xyz7 = numpy.array([a7.x.get_value(), a7.y.get_value(), a7.z.get_value()])
    xyz20 = numpy.array(
        [a20.x.get_value(), a20.y.get_value(), a20.z.get_value()]
    )
    xyz25 = numpy.array(
        [a25.x.get_value(), a25.y.get_value(), a25.z.get_value()]
    )

    v1 = xyz7 - xyz0
    d1 = numpy.dot(v1, v1) ** 0.5
    v2 = xyz7 - xyz20
    d2 = numpy.dot(v2, v2) ** 0.5

    angle0 = numpy.arccos(numpy.dot(v1, v2) / (d1 * d2))

    # Add a parameter
    p1 = m.add_bond_angle_parameter("C0720", a0, a7, a20)
    # Have another atom tag along for the ride
    p1.add_atoms([a25])

    # C1: A bond angle Parameter is added.
    # Expected: Its value is the current bond angle.
    actual_angle = p1.get_value()
    expected_angle = angle0
    assert actual_angle == pytest.approx(expected_angle, abs=1e-6)

    # C2: The bond angle Parameter is stretched by 5%.
    # Expected: The Parameter and the measured angle both take the new
    # value, and only the third and tag-along atoms move.
    scale = 1.05
    p1.set_value(scale * angle0)

    actual_angle = p1.get_value()
    expected_angle = scale * angle0
    assert actual_angle == pytest.approx(expected_angle, abs=1e-6)

    xyz0a = numpy.array([a0.x.get_value(), a0.y.get_value(), a0.z.get_value()])
    xyz7a = numpy.array([a7.x.get_value(), a7.y.get_value(), a7.z.get_value()])
    xyz20a = numpy.array(
        [a20.x.get_value(), a20.y.get_value(), a20.z.get_value()]
    )
    xyz25a = numpy.array(
        [a25.x.get_value(), a25.y.get_value(), a25.z.get_value()]
    )

    v1a = xyz7a - xyz0a
    d1a = numpy.dot(v1a, v1a) ** 0.5
    v2a = xyz7a - xyz20a
    d2a = numpy.dot(v2a, v2a) ** 0.5

    angle1 = numpy.arccos(numpy.dot(v1a, v2a) / (d1a * d2a))

    actual_measured_angle = angle1
    expected_measured_angle = scale * angle0
    assert actual_measured_angle == pytest.approx(
        expected_measured_angle, abs=1e-6
    )

    actual_moved = {
        "C0": not numpy.array_equal(xyz0, xyz0a),
        "C7": not numpy.array_equal(xyz7, xyz7a),
        "C20": not numpy.array_equal(xyz20, xyz20a),
        "C25": not numpy.array_equal(xyz25, xyz25a),
    }
    expected_moved = {"C0": False, "C7": False, "C20": True, "C25": True}
    assert actual_moved == expected_moved

    return


def test_explicit_dihedral_angle_parameter(c60_crystal):
    """Test adding dihedral angle parameters to the molecule."""
    occryst = c60_crystal

    # make our crystal
    crystal = ObjCrystCrystalParSet("bucky", occryst)
    m = crystal.c60

    a0 = m.atoms[0]
    a7 = m.atoms[7]
    a20 = m.atoms[20]
    a25 = m.atoms[25]
    a33 = m.atoms[33]

    xyz0 = numpy.array([a0.x.get_value(), a0.y.get_value(), a0.z.get_value()])
    xyz7 = numpy.array([a7.x.get_value(), a7.y.get_value(), a7.z.get_value()])
    xyz20 = numpy.array(
        [a20.x.get_value(), a20.y.get_value(), a20.z.get_value()]
    )
    xyz25 = numpy.array(
        [a25.x.get_value(), a25.y.get_value(), a25.z.get_value()]
    )
    xyz33 = numpy.array(
        [a33.x.get_value(), a33.y.get_value(), a33.z.get_value()]
    )

    v12 = xyz0 - xyz7
    v23 = xyz7 - xyz20
    v34 = xyz20 - xyz25
    v123 = numpy.cross(v12, v23)
    v234 = numpy.cross(v23, v34)

    d123 = numpy.dot(v123, v123) ** 0.5
    d234 = numpy.dot(v234, v234) ** 0.5
    angle0 = -numpy.arccos(numpy.dot(v123, v234) / (d123 * d234))

    # Add a parameter
    p1 = m.add_dihedral_angle_parameter("C072025", a0, a7, a20, a25)
    # Have another atom tag along for the ride
    p1.add_atoms([a33])

    # C1: A dihedral angle Parameter is added.
    # Expected: Its value is the current dihedral angle.
    actual_angle = p1.get_value()
    expected_angle = angle0
    assert actual_angle == pytest.approx(expected_angle, abs=1e-6)

    # C2: The dihedral angle Parameter is stretched by 5%.
    # Expected: The Parameter and the measured angle both take the new
    # value, and only the fourth and tag-along atoms move.
    scale = 1.05
    p1.set_value(scale * angle0)

    actual_angle = p1.get_value()
    expected_angle = scale * angle0
    assert actual_angle == pytest.approx(expected_angle, abs=1e-6)

    xyz0a = numpy.array([a0.x.get_value(), a0.y.get_value(), a0.z.get_value()])
    xyz7a = numpy.array([a7.x.get_value(), a7.y.get_value(), a7.z.get_value()])
    xyz20a = numpy.array(
        [a20.x.get_value(), a20.y.get_value(), a20.z.get_value()]
    )
    xyz25a = numpy.array(
        [a25.x.get_value(), a25.y.get_value(), a25.z.get_value()]
    )
    xyz33a = numpy.array(
        [a33.x.get_value(), a33.y.get_value(), a33.z.get_value()]
    )

    v12a = xyz0a - xyz7a
    v23a = xyz7a - xyz20a
    v34a = xyz20a - xyz25a
    v123a = numpy.cross(v12a, v23a)
    v234a = numpy.cross(v23a, v34a)

    d123a = numpy.dot(v123a, v123a) ** 0.5
    d234a = numpy.dot(v234a, v234a) ** 0.5
    angle1 = -numpy.arccos(numpy.dot(v123a, v234a) / (d123a * d234a))
    actual_measured_angle = angle1
    expected_measured_angle = scale * angle0
    assert actual_measured_angle == pytest.approx(
        expected_measured_angle, abs=1e-6
    )

    actual_moved = {
        "C0": not numpy.array_equal(xyz0, xyz0a),
        "C7": not numpy.array_equal(xyz7, xyz7a),
        "C20": not numpy.array_equal(xyz20, xyz20a),
        "C25": not numpy.array_equal(xyz25, xyz25a),
        "C33": not numpy.array_equal(xyz33, xyz33a),
    }
    expected_moved = {
        "C0": False,
        "C7": False,
        "C20": False,
        "C25": True,
        "C33": True,
    }
    assert actual_moved == expected_moved

    return


def get_objcryst_parset_space_group(space_group):
    """Make an ObjCrystCrystalParSet with the proper space group."""
    sgobjcryst = SpaceGroup(space_group.short_name)
    sgnew = ObjCrystCrystalParSet._create_space_group(sgobjcryst)
    return sgnew


def get_diffpy_space_group_hash(space_group):
    lines = [str(space_group.number % 1000)] + sorted(
        map(str, space_group.iter_symops())
    )
    s = "\n".join(lines)
    return s


def are_space_groups_equivalent(sg1, sg2):
    """Check to see if two space group objects are the same."""
    hash1 = get_diffpy_space_group_hash(sg1)
    hash2 = get_diffpy_space_group_hash(sg2)
    return hash1 == hash2


# FIXME: only about 50% of the spacegroups pass the assertion
# test disabled even if cctbx is installed
def xtest_create_space_group():
    """Check all sgtbx space groups for proper conversion to
    SpaceGroup."""
    from cctbx import sgtbx

    for smbls in sgtbx.space_group_symbol_iterator():
        shn = smbls.hermann_mauguin()
        short_name = shn.replace(" ", "")
        if spacegroups.is_space_group_identifier(short_name):
            space_group = spacegroups.get_space_group(shn)
            sgnew = get_objcryst_parset_space_group(space_group)
            actual_equivalent = are_space_groups_equivalent(space_group, sgnew)
            expected_equivalent = True
            assert actual_equivalent == expected_equivalent
    return
