#!/usr/bin/env python
##############################################################################
#
# diffpy.srfit      by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Code to set space group constraints for a crystal structure."""

import re

import numpy

from diffpy.srfit.fitbase.parameter import ParameterProxy
from diffpy.srfit.fitbase.recipeorganizer import RecipeContainer

__all__ = ["constrain_as_space_group"]


def constrain_as_space_group(
    phase,
    spacegroup,
    scatterers=None,
    sgoffset=[0, 0, 0],
    constrainlat=True,
    constrainadps=True,
    adpsymbols=None,
    isosymbol="Uiso",
):
    """Constrain a P1 structure to a space group.

    This applies space group constraints to a structure ParameterSet with
    P1 symmetry. The passed scatterers are explicitly constrained to the
    specified space group, and the ADPs and lattice may be constrained as
    well. New Parameters used in the constraints are created within the
    returned SpaceGroupParameters object. Constraints are created in the
    ParameterSet that contains the constrained Parameter. This erases any
    constraints or constant flags on the scatterers, lattice or ADPs that
    are to be constrained.

    Parameters
    ----------
    phase : BaseStructureParSet
        The structure ParameterSet to constrain.
    spacegroup : int, str or diffpy.structure.spacegroups.SpaceGroup
        The space group number, symbol or SpaceGroup instance.
    scatterers : list of ParameterSet, optional
        The scatterer ParameterSets to constrain. If None (default), all
        scatterers returned by ``phase.get_scatterers()`` are constrained.
    sgoffset : list of float, optional
        The offset of the space group origin (default [0, 0, 0]).
    constrainlat : bool, optional
        The flag indicating whether to constrain the lattice (default
        True).
    constrainadps : bool, optional
        The flag indicating whether to constrain the ADPs (default True).
    adpsymbols : list of str, optional
        The ADP names. By default this is
        diffpy.structure.symmetryutilities.stdUsymbols (U11, U22, etc.).
        The names must be given in the same order as stdUsymbols.
    isosymbol : str, optional
        The name of the isotropic ADP (default "Uiso"). If None,
        isotropic ADPs are constrained via the anisotropic ADPs.

    Returns
    -------
    SpaceGroupParameters
        The free Parameters of the structure that remain after applying
        the space group constraints.

    Notes
    -----
    The lattice constraints are applied as follows.

    Triclinic
        No constraints.
    Monoclinic
        alpha and beta are fixed to 90 unless alpha != beta and
        alpha == gamma, in which case alpha and gamma are fixed to 90.
    Orthorhombic
        alpha, beta and gamma are fixed to 90.
    Tetragonal
        b is constrained to a and alpha, beta and gamma are fixed to 90.
    Trigonal
        If gamma == 120, then b is constrained to a, alpha and beta are
        fixed to 90 and gamma is fixed to 120. Otherwise, b and c are
        constrained to a, and beta and gamma are fixed to alpha.
    Hexagonal
        b is constrained to a, alpha and beta are fixed to 90 and gamma
        is fixed to 120.
    Cubic
        b and c are constrained to a, and alpha, beta and gamma are fixed
        to 90.
    """
    from diffpy.structure.spacegroups import GetSpaceGroup, SpaceGroup

    sg = spacegroup
    if not isinstance(spacegroup, SpaceGroup):
        sg = GetSpaceGroup(spacegroup)
    sgp = _constrain_as_space_group(
        phase,
        sg,
        scatterers,
        sgoffset,
        constrainlat,
        constrainadps,
        adpsymbols,
        isosymbol,
    )

    return sgp


def _constrain_as_space_group(
    phase,
    sg,
    scatterers=None,
    sgoffset=[0, 0, 0],
    constrainlat=True,
    constrainadps=True,
    adpsymbols=None,
    isosymbol="Uiso",
):
    """Restricted interface to constrain_as_space_group.

    Arguments: As constrain_as_space_group, except
    -----------------------------------------------
    sg
        diffpy.structure.spacegroups.SpaceGroup instance
    """
    from diffpy.structure.symmetryutilities import stdUsymbols

    if scatterers is None:
        scatterers = phase.get_scatterers()
    if adpsymbols is None:
        adpsymbols = stdUsymbols

    sgp = SpaceGroupParameters(
        phase,
        sg,
        scatterers,
        sgoffset,
        constrainlat,
        constrainadps,
        adpsymbols,
        isosymbol,
    )

    return sgp


# End constrain_as_space_group


class BaseSpaceGroupParameters(RecipeContainer):
    """Base class for holding space group Parameters.

    This class stores the variable Parameters of a structure, leaving out
    those that are constrained or fixed by the space group. It has the same
    Parameter attribute access as a ParameterSet, which makes it easy to
    access the free variables of a structure when scripting.

    Attributes
    ----------
    name : str
        The name of this container (default "sgpars").
    """

    def __init__(self, name="sgpars"):
        """Initialize the space group Parameter container.

        Parameters
        ----------
        name : str, optional
            The name of this container (default "sgpars").
        """
        RecipeContainer.__init__(self, name)
        return

    def add_parameter(self, par, check=True):
        """Store a Parameter.

        Parameters
        ----------
        par : Parameter
            The Parameter to be stored.
        check : bool, optional
            The flag indicating whether to check for an existing Parameter
            of the same name (default True).

        Raises
        ------
        ValueError
            If the Parameter has no name, or if `check` is True and a
            Parameter of the same name has already been stored.
        """
        # Store the Parameter
        RecipeContainer._add_object(self, par, self._parameters, check)
        return


# End class BaseSpaceGroupParameters


class SpaceGroupParameters(BaseSpaceGroupParameters):
    """Create and hold the free Parameters of a space group constraint.

    This class stores the variable Parameters of a structure, leaving out
    those that are constrained or fixed by the space group, and does the
    work of constrain_as_space_group. It has the same Parameter attribute
    access as a ParameterSet.

    Attributes
    ----------
    name : str
        The name of this container, always "sgpars".
    phase : BaseStructureParSet
        The constrained structure ParameterSet.
    sg : diffpy.structure.spacegroups.SpaceGroup
        The space group of the constraints.
    sgoffset : list of float
        The offset of the space group origin.
    scatterers : list of ParameterSet
        The constrained scatterer ParameterSets.
    constrainlat : bool
        The flag indicating whether the lattice is constrained.
    constrainadps : bool
        The flag indicating whether the ADPs are constrained.
    adpsymbols : list of str
        The ADP names.
    isosymbol : str or None
        The name of the isotropic ADP.
    xyzpars : BaseSpaceGroupParameters
        The free xyz Parameters, created on first access.
    latpars : BaseSpaceGroupParameters
        The free lattice Parameters, created on first access.
    adppars : BaseSpaceGroupParameters
        The free ADP Parameters, created on first access.
    """

    def __init__(
        self,
        phase,
        sg,
        scatterers,
        sgoffset,
        constrainlat,
        constrainadps,
        adpsymbols,
        isosymbol,
    ):
        """Initialize the space group Parameters.

        The constraints are not applied until the Parameters are first
        accessed.

        Parameters
        ----------
        phase : BaseStructureParSet
            The structure ParameterSet to be constrained.
        sg : diffpy.structure.spacegroups.SpaceGroup
            The space group of the constraints.
        scatterers : list of ParameterSet
            The scatterer ParameterSets to constrain.
        sgoffset : list of float
            The offset of the space group origin.
        constrainlat : bool
            The flag indicating whether to constrain the lattice.
        constrainadps : bool
            The flag indicating whether to constrain the ADPs.
        adpsymbols : list of str
            The ADP names, in the same order as
            diffpy.structure.symmetryutilities.stdUsymbols.
        isosymbol : str or None
            The name of the isotropic ADP. If None, isotropic ADPs are
            constrained via the anisotropic ADPs.
        """
        BaseSpaceGroupParameters.__init__(self)
        self._latpars = None
        self._xyzpars = None
        self._adppars = None

        self._parsets = {}
        self._manage(self._parsets)

        self.phase = phase
        self.sg = sg
        self.sgoffset = sgoffset
        self.scatterers = scatterers
        self.constrainlat = constrainlat
        self.constrainadps = constrainadps
        self.adpsymbols = adpsymbols
        self.isosymbol = isosymbol

        return

    def __iter__(self):
        """Iterate over top-level parameters."""
        if (
            self._latpars is None
            or self._xyzpars is None
            or self._adppars is None
        ):
            self._make_constraints()
        return RecipeContainer.__iter__(self)

    latpars = property(lambda self: self._get_lat_pars())

    def _get_lat_pars(self):
        """Accessor for _latpars."""
        if self._latpars is None:
            self._constrain_lattice()
        return self._latpars

    xyzpars = property(lambda self: self._get_xyz_pars())

    def _get_xyz_pars(self):
        """Accessor for _xyzpars."""
        positions = []
        for scatterer in self.scatterers:
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])
        if self._xyzpars is None:
            self._constrain_xyzs(positions)
        return self._xyzpars

    adppars = property(lambda self: self._get_adp_pars())

    def _get_adp_pars(self):
        """Accessor for _adppars."""
        positions = []
        for scatterer in self.scatterers:
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])
        if self._adppars is None:
            self._constrain_adps(positions)
        return self._adppars

    def _make_constraints(self):
        """Constrain the structure to the space group.

        This works as described by the constrain_as_space_group method.
        """
        # Start by clearing the constraints
        self._clear_constraints()

        scatterers = self.scatterers

        # Prepare positions
        positions = []
        for scatterer in scatterers:
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])

        self._constrain_lattice()
        self._constrain_xyzs(positions)
        self._constrain_adps(positions)

        return

    def _clear_constraints(self):
        """Clear old constraints.

        This only clears constraints where new ones are going to be
        applied.
        """
        phase = self.phase
        scatterers = self.scatterers
        isosymbol = self.isosymbol
        adpsymbols = self.adpsymbols

        # Clear xyz
        for scatterer in scatterers:

            for par in [scatterer.x, scatterer.y, scatterer.z]:
                if scatterer.is_constrained(par):
                    scatterer.remove_constraint(par)
                par.set_constant(False)

        # Clear the lattice
        if self.constrainlat:

            lattice = phase.get_lattice()
            latpars = [
                lattice.a,
                lattice.b,
                lattice.c,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
            ]
            for par in latpars:
                if lattice.is_constrained(par):
                    lattice.remove_constraint(par)
                par.set_constant(False)

        # Clear ADPs
        if self.constrainadps:
            for scatterer in scatterers:
                if isosymbol:
                    par = scatterer.get(isosymbol)
                    if par is not None:
                        if scatterer.is_constrained(par):
                            scatterer.remove_constraint(par)
                        par.set_constant(False)

                for pname in adpsymbols:
                    par = scatterer.get(pname)
                    if par is not None:
                        if scatterer.is_constrained(par):
                            scatterer.remove_constraint(par)
                        par.set_constant(False)

        return

    def _constrain_lattice(self):
        """Constrain the lattice parameters."""
        if not self.constrainlat:
            return

        phase = self.phase
        sg = self.sg

        lattice = phase.get_lattice()
        system = sg.crystal_system
        if not system:
            system = "Triclinic"
        system = system.title()
        # This makes the constraints
        f = _constraint_map[system]
        f(lattice)

        # Now get the unconstrained, non-constant lattice pars and store them.
        self._latpars = BaseSpaceGroupParameters("latpars")
        latpars = [
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
        ]
        pars = [p for p in latpars if not p.const and not p.constrained]
        for par in pars:
            # FIXME - the original parameter will still appear as
            # constrained.
            newpar = self.__add_par(par.name, par)
            self._latpars.add_parameter(newpar)

        return

    def _constrain_xyzs(self, positions):
        """Constrain the positions.

        Parameters
        ----------
        positions
            The coordinates of the scatterers.
        """
        from diffpy.structure.symmetryutilities import SymmetryConstraints

        sg = self.sg
        sgoffset = self.sgoffset

        # We do this without ADPs here so we can skip much complication. See
        # the _constrain_adps method for details.
        g = SymmetryConstraints(sg, positions, sgoffset=sgoffset)

        scatterers = self.scatterers
        self._xyzpars = BaseSpaceGroupParameters("xyzpars")

        # Make proxies to the free xyz parameters
        xyznames = [name[:1] + "_" + name[1:] for name, val in g.pospars]
        for pname in xyznames:
            name, idx = pname.rsplit("_", 1)
            idx = int(idx)
            par = scatterers[idx].get(name)
            newpar = self.__add_par(pname, par)
            self._xyzpars.add_parameter(newpar)

        # Constrain non-free xyz parameters
        fpos = g.positionFormulas(xyznames)
        for idx, tmp in enumerate(zip(scatterers, fpos)):
            scatterer, fp = tmp

            # Extract the constraint equation from the formula
            for parname, formula in fp.items():
                _makeconstraint(
                    parname, formula, scatterer, idx, self._parameters
                )

        return

    def _constrain_adps(self, positions):
        """Constrain the ADPs.

        Parameters
        ----------
        positions
            The coordinates of the scatterers.
        """
        from diffpy.structure.symmetryutilities import (
            SymmetryConstraints,
            stdUsymbols,
        )

        if not self.constrainadps:
            return

        sg = self.sg
        sgoffset = self.sgoffset
        scatterers = self.scatterers
        isosymbol = self.isosymbol
        adpsymbols = self.adpsymbols
        adpmap = dict(zip(stdUsymbols, adpsymbols))
        self._adppars = BaseSpaceGroupParameters("adppars")

        # Prepare ADPs. Note that not all scatterers have constrainable ADPs.
        # For example, MoleculeParSet from objcryststructure does not. We
        # discard those.
        nonadps = []
        Uijs = []
        for sidx, scatterer in enumerate(scatterers):

            pars = [scatterer.get(symb) for symb in adpsymbols]

            if None in pars:
                nonadps.append(sidx)
                continue

            Uij = numpy.zeros((3, 3), dtype=float)
            for idx, par in enumerate(pars):
                i, j = _idxtoij[idx]
                Uij[i, j] = Uij[j, i] = par.get_value()

            Uijs.append(Uij)

        # Discard any positions for the nonadps
        positions = list(positions)
        nonadps.reverse()
        [positions.pop(idx) for idx in nonadps]

        # Now we can create symmetry constraints without having to worry about
        # the nonadps
        g = SymmetryConstraints(sg, positions, Uijs, sgoffset=sgoffset)

        adpnames = [adpmap[name[:3]] + "_" + name[3:] for name, val in g.Upars]

        # Make proxies to the free adp parameters. We start by filtering out
        # the isotropic ones so we can use the isotropic parameter.
        isoidx = []
        isonames = []
        for pname in adpnames:
            name, idx = pname.rsplit("_", 1)
            idx = int(idx)
            # Check for isotropic ADPs
            scatterer = scatterers[idx]
            if isosymbol and g.Uisotropy[idx] and idx not in isoidx:
                isoidx.append(idx)
                par = scatterer.get(isosymbol)
                if par is not None:
                    parname = "%s_%i" % (isosymbol, idx)
                    newpar = self.__add_par(parname, par)
                    self._adppars.add_parameter(newpar)
                    isonames.append(newpar.name)
            else:
                par = scatterer.get(name)
                if par is not None:
                    newpar = self.__add_par(pname, par)
                    self._adppars.add_parameter(newpar)

        # Constrain dependent isotropics
        for idx, isoname in zip(isoidx[:], isonames):
            for j in g.coremap[idx]:
                if j == idx:
                    continue
                isoidx.append(j)
                scatterer = scatterers[j]
                scatterer.add_constraint(
                    isosymbol, isoname, params=self._parameters
                )

        fadp = g.UFormulas(adpnames)

        # Constrain dependent anisotropics. We use the fact that an
        # anisotropic cannot be dependent on an isotropic.
        for idx, tmp in enumerate(zip(scatterers, fadp)):
            if idx in isoidx:
                continue
            scatterer, fa = tmp
            # Extract the constraint equation from the formula
            for stdparname, formula in fa.items():
                pname = adpmap[stdparname]
                _makeconstraint(
                    pname, formula, scatterer, idx, self._parameters
                )

    def __add_par(self, parname, par):
        """Constrain a parameter via proxy with a specified name.

        Parameters
        ----------
        par
            Parameter to constrain
        idx
            Index to identify scatterer from which par comes
        """
        newpar = ParameterProxy(parname, par)
        self.add_parameter(newpar)
        return newpar


# End class SpaceGroupParameters

# crystal system rules
# ref: Benjamin, W. A., Introduction to crystallography,
# New York (1969), p.60


def _constrain_triclinic(lattice):
    """Make constraints for Triclinic systems."""
    return


def _constrain_monoclinic(lattice):
    """Make constraints for Monoclinic systems.

    alpha and beta are fixed to 90 unless alpha != beta and alpha ==
    gamma, in which case alpha and gamma are constrained to 90.
    """
    afactor = 1
    if lattice.angunits == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.alpha.set_constant(True, ang90)
    beta = lattice.beta.get_value()
    gamma = lattice.gamma.get_value()

    if ang90 != beta and ang90 == gamma:
        lattice.gamma.set_constant(True, ang90)
    else:
        lattice.beta.set_constant(True, ang90)
    return


def _constrain_orthorhombic(lattice):
    """Make constraints for Orthorhombic systems.

    alpha, beta and gamma are constrained to 90
    """
    afactor = 1
    if lattice.angunits == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang90)
    return


def _constrain_tetragonal(lattice):
    """Make constraints for Tetragonal systems.

    b is constrained to a and alpha, beta and gamma are constrained to
    90.
    """
    afactor = 1
    if lattice.angunits == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang90)
    lattice.add_constraint(lattice.b, lattice.a)
    return


def _constrain_trigonal(lattice):
    """Make constraints for Trigonal systems.

    If gamma == 120, then b is constrained to a, alpha and beta are
    constrained to 90 and gamma is constrained to 120. Otherwise, b and
    c are constrained to a, beta and gamma are constrained to alpha.
    """
    afactor = 1
    if lattice.angunits == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    ang120 = 120.0 * afactor
    if lattice.gamma.get_value() == ang120:
        lattice.add_constraint(lattice.b, lattice.a)
        lattice.alpha.set_constant(True, ang90)
        lattice.beta.set_constant(True, ang90)
        lattice.gamma.set_constant(True, ang120)
    else:
        lattice.add_constraint(lattice.b, lattice.a)
        lattice.add_constraint(lattice.c, lattice.a)
        lattice.add_constraint(lattice.beta, lattice.alpha)
        lattice.add_constraint(lattice.gamma, lattice.alpha)
    return


def _constrain_hexagonal(lattice):
    """Make constraints for Hexagonal systems.

    b is constrained to a, alpha and beta are constrained to 90 and
    gamma is constrained to 120.
    """
    afactor = 1
    if lattice.angunits == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    ang120 = 120.0 * afactor
    lattice.add_constraint(lattice.b, lattice.a)
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang120)
    return


def _constrain_cubic(lattice):
    """Make constraints for Cubic systems.

    b and c are constrained to a, alpha, beta and gamma are constrained
    to 90.
    """
    afactor = 1
    if lattice.angunits == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.add_constraint(lattice.b, lattice.a)
    lattice.add_constraint(lattice.c, lattice.a)
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang90)
    return


# This is used to map the correct crystal system to the proper constraint
# function.
_constraint_map = {
    "Triclinic": _constrain_triclinic,
    "Monoclinic": _constrain_monoclinic,
    "Orthorhombic": _constrain_orthorhombic,
    "Tetragonal": _constrain_tetragonal,
    "Trigonal": _constrain_trigonal,
    "Hexagonal": _constrain_hexagonal,
    "Cubic": _constrain_cubic,
}


def _makeconstraint(parname, formula, scatterer, idx, ns={}):
    """Constrain a parameter according to a formula.

    Parameters
    ----------
    parname
        Name of parameter
    formula
        Constraint formula
    scatterer
        scatterer containing par of parname
    idx
        Index to identify scatterer from which par comes
    ns
        namespace to draw extra names from (default {})

    Returns
    -------
    par
        Returns the parameter if it is free.
    """
    par = scatterer.get(parname)

    if par is None:
        return

    compname = "%s_%i" % (parname, idx)

    # Check to see if this parameter is free
    pat = r"%s *([+-] *\d+)?$" % compname
    if re.match(pat, formula):
        return par

    # Check to see if it is a constant
    fval = _get_float(formula)
    if fval is not None:
        par.set_constant()
        return

    # If we got here, then we have a constraint equation
    # Fix any division issues
    formula = formula.replace("/", "*1.0/")
    scatterer.add_constraint(par, formula, params=ns)
    return


def _get_float(formula):
    """Get a float from a formula string, or None if this is not
    possible."""
    try:
        return eval(formula)
    except NameError:
        return None


# Constants needed above
_idxtoij = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
deg2rad = numpy.pi / 180
rad2deg = 1.0 / deg2rad


# End of file
