"""Fusion reaction physics models and constants."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from process.core import constants
from process.core.model import DataStructure
from process.data_structure.physics_variables import PhysicsData
from process.models.physics.plasma_profiles import PlasmaProfile
from process.models.physics.p11b_cross_sections import (
    P11B_GAMOW_ENERGY_MEV,
    P11B_REDUCED_MASS_MEV,
    get_p11b_cross_section_model,
)
from process.models.physics.reactivity import (
    as_reactivity_model,
    interpolate_reactivity_loglog,
    maxwellian_reactivity,
    profile_reactivity_integrand,
    tabulate_reactivity,
)

logger = logging.getLogger(__name__)


REACTION_CONSTANTS_DT = {
    "bg": 34.3827,
    "mrc2": 1.124656e6,
    "cc1": 1.17302e-9,
    "cc2": 1.51361e-2,
    "cc3": 7.51886e-2,
    "cc4": 4.60643e-3,
    "cc5": 1.35000e-2,
    "cc6": -1.06750e-4,
    "cc7": 1.36600e-5,
}

REACTION_CONSTANTS_DHE3 = {
    "bg": 68.7508,
    "mrc2": 1.124572e6,
    "cc1": 5.51036e-10,
    "cc2": 6.41918e-3,
    "cc3": -2.02896e-3,
    "cc4": -1.91080e-5,
    "cc5": 1.35776e-4,
    "cc6": 0.0,
    "cc7": 0.0,
}

REACTION_CONSTANTS_DD1 = {
    "bg": 31.3970,
    "mrc2": 0.937814e6,
    "cc1": 5.43360e-12,
    "cc2": 5.85778e-3,
    "cc3": 7.68222e-3,
    "cc4": 0.0,
    "cc5": -2.96400e-6,
    "cc6": 0.0,
    "cc7": 0.0,
}

REACTION_CONSTANTS_DD2 = {
    "bg": 31.3970,
    "mrc2": 0.937814e6,
    "cc1": 5.65718e-12,
    "cc2": 3.41267e-3,
    "cc3": 1.99167e-3,
    "cc4": 0.0,
    "cc5": 1.05060e-5,
    "cc6": 0.0,
    "cc7": 0.0,
}


class FusionReactionRate:
    """Calculate the fusion reaction rate for each reaction case (DT, DHE3, DD1, DD2).

    This class provides methods to numerically integrate over the plasma cross-section
    to find the core plasma fusion power for different fusion reactions. The reactions
    considered are:
        - Deuterium-Tritium (D-T)
        - Deuterium-Helium-3 (D-3He)
        - Deuterium-Deuterium (D-D) first branch
        - Deuterium-Deuterium (D-D) second branch

    The class uses the Bosch-Hale parametrization to compute the volumetric fusion
    reaction rate <sigma v> and integrates over the plasma cross-section to find the
    core plasma fusion power.

    Attributes
    ----------
         plasma_profile (PlasmaProfile): The parameterized temperature and density
         profiles of the plasma.
         sigmav_dt_average (float): Average fusion reaction rate <sigma v> for D-T.
         dhe3_power_density (float): Fusion power density produced by the D-3He reaction.
         dd_power_density (float): Fusion power density produced by the D-D reactions.
         dt_power_density (float): Fusion power density produced by the D-T reaction.
         alpha_power_density (float): Power density of alpha particles produced.
         pden_non_alpha_charged_mw (float): Power density of charged particles produced.
         neutron_power_density (float): Power density of neutrons produced.
         fusion_rate_density (float): Fusion reaction rate density.
         alpha_rate_density (float): Alpha particle production rate density.
         proton_rate_density (float): Proton production rate density.
         f_dd_branching_trit (float): The rate of tritium producing D-D reactions to
         3He ones.

    References
    ----------
        - H.-S. Bosch and G. M. Hale, “Improved formulas for fusion cross-sections
          and thermal reactivities,” Nuclear Fusion, vol. 32, no. 4, pp. 611-631,
          Apr. 1992, doi: https://doi.org/10.1088/0029-5515/32/4/i07.
    """

    def __init__(self, plasma_profile: PlasmaProfile, data: DataStructure):
        """
        Initialize the FusionReactionRate class with the given plasma profile.

        Parameters
        ----------
        plasma_profile:
            The parameterized temperature and density profiles of the plasma.


        """
        self.plasma_profile = plasma_profile
        self.data = data

        self.sigmav_dt_average = 0.0
        self.dhe3_power_density = 0.0
        self.dd_power_density = 0.0
        self.dt_power_density = 0.0
        self.alpha_power_density = 0.0
        self.pden_non_alpha_charged_mw = 0.0
        self.neutron_power_density = 0.0
        self.fusion_rate_density = 0.0
        self.alpha_rate_density = 0.0
        self.proton_rate_density = 0.0
        self.f_dd_branching_trit = 0.0
        self.p11b_power_density = 0.0
        self.p11b_rate_density = 0.0
        self.p11b_alpha_rate_density = 0.0

    def deuterium_branching(self, ion_temperature: float) -> float:
        """Calculate the relative rate of tritium producing D-D reactions to 3He ones
        based on the volume averaged ion temperature

        Parameters
        ----------
        ion_temperature :
            float

        Notes
        -----
        For ion temperatures between 0.5 keV and 200 keV.
        The deviation of the fit from the R-matrix branching ratio is always smaller
        than 0.5%.

        References
        ----------
            - H.-S. Bosch and G. M. Hale, “Improved formulas for fusion cross-sections
              and thermal reactivities,” Nuclear Fusion, vol. 32, no. 4, pp. 611-631,
              Apr. 1992, doi: https://doi.org/10.1088/0029-5515/32/4/i07.
        """
        # Divide by 2 to get the branching ratio for the D-D reaction that produces
        # tritium as the output is just the ratio of the two normalized cross sections
        self.f_dd_branching_trit = (
            1.02934
            - 8.3264e-3 * ion_temperature
            + 1.7631e-4 * ion_temperature**2
            - 1.8201e-6 * ion_temperature**3
            + 6.9855e-9 * ion_temperature**4
        ) / 2.0

    def dt_reaction(self):
        """D + T --> 4He + n reaction

        This method calculates the fusion reaction rate and power density for the
        deuterium-tritium (D-T) fusion reaction. It uses the Bosch-Hale parametrization
        to compute the volumetric fusion reaction rate <sigma v> and integrates over
        the plasma cross-section to find the core plasma fusion power.

        The method updates the following attributes:
            - self.sigmav_dt_average: Average fusion reaction rate <sigma v> for D-T.
            - self.dt_power_density: Fusion power density produced by the D-T reaction.
            - self.alpha_power_density: Power density of alpha particles produced.
            - self.pden_non_alpha_charged_mw: Power density of charged particles
              produced.
            - self.neutron_power_density: Power density of neutrons produced.
            - self.fusion_rate_density: Fusion reaction rate density.
            - self.alpha_rate_density: Alpha particle production rate density.
            - self.proton_rate_density: Proton production rate density.


        """
        # Skip when either fuel fraction is zero. Bosch-Hale can return NaN at
        # extreme Ti (outside the fit range); IEEE NaN*0 is still NaN and would
        # pollute neutron / total fusion power even for pure p-11B.
        if (
            self.data.physics.f_plasma_fuel_deuterium <= 0.0
            or self.data.physics.f_plasma_fuel_tritium <= 0.0
        ):
            self.sigmav_dt_average = 0.0
            self.dt_power_density = 0.0
            self.data.physics.fusrat_plasma_dt_profile = np.zeros_like(
                self.plasma_profile.neprofile.profile_y
            )
            return

        # Initialize Bosch-Hale constants for the D-T reaction
        dt = BoschHaleConstants(**REACTION_CONSTANTS_DT)

        self.data.physics.fusrat_plasma_dt_profile = (
            bosch_hale_reactivity(
                (
                    self.data.physics.temp_plasma_ion_vol_avg_kev
                    / self.data.physics.temp_plasma_electron_vol_avg_kev
                )
                * self.plasma_profile.teprofile.profile_y,
                dt,
            )
            * self.data.physics.f_plasma_fuel_deuterium
            * self.data.physics.f_plasma_fuel_tritium
            * (
                self.plasma_profile.neprofile.profile_y
                * (
                    self.data.physics.nd_plasma_fuel_ions_vol_avg
                    / self.data.physics.nd_plasma_electrons_vol_avg
                )
            )
            ** 2
        )

        # Calculate the fusion reaction rate integral using Simpson's rule
        sigmav = integrate.simpson(
            fusion_rate_integral(
                self.plasma_profile, dt, physics_data=self.data.physics
            ),
            x=self.plasma_profile.neprofile.profile_x,
            dx=self.plasma_profile.neprofile.profile_dx,
        )

        # Store the average fusion reaction rate
        self.sigmav_dt_average = sigmav

        # Reaction energy in MegaJoules [MJ]
        reaction_energy = constants.D_T_ENERGY / 1.0e6

        # Calculate the fusion power density produced [MW/m^3]
        fusion_power_density = (
            sigmav
            * reaction_energy
            * (
                self.data.physics.f_plasma_fuel_deuterium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
            * (
                self.data.physics.f_plasma_fuel_tritium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
        )

        # Power densities for different particles [MW/m^3]
        # Alpha particle gets approximately 20% of the fusion power
        alpha_power_density = (
            1.0 - constants.DT_NEUTRON_ENERGY_FRACTION
        ) * fusion_power_density
        pden_non_alpha_charged_mw = 0.0
        neutron_power_density = (
            constants.DT_NEUTRON_ENERGY_FRACTION * fusion_power_density
        )

        # Calculate the fusion rate density [reactions/m^3/second]
        fusion_rate_density = fusion_power_density / reaction_energy
        alpha_rate_density = fusion_rate_density
        proton_rate_density = 0.0

        # Update the cumulative D-T power density
        self.dt_power_density = fusion_power_density

        # Sum the fusion rates for all particles
        self.sum_fusion_rates(
            alpha_power_density,
            pden_non_alpha_charged_mw,
            neutron_power_density,
            fusion_rate_density,
            alpha_rate_density,
            proton_rate_density,
        )

    def dhe3_reaction(self):
        """D + 3He --> 4He + p reaction

        This method calculates the fusion reaction rate and power density for the
        deuterium-helium-3 (D-3He) fusion reaction. It uses the Bosch-Hale
        parametrization to compute the volumetric fusion reaction rate <sigma v> and
        integrates over the plasma cross-section to find the core plasma fusion power.

        The method updates the following attributes:
            - self.dhe3_power_density: Fusion power density produced by the D-3He
              reaction.
            - self.alpha_power_density: Power density of alpha particles produced.
            - self.pden_non_alpha_charged_mw: Power density of charged particles
              produced.
            - self.neutron_power_density: Power density of neutrons produced.
            - self.fusion_rate_density: Fusion reaction rate density.
            - self.alpha_rate_density: Alpha particle production rate density.
            - self.proton_rate_density: Proton production rate density.


        """
        if (
            self.data.physics.f_plasma_fuel_deuterium <= 0.0
            or self.data.physics.f_plasma_fuel_helium3 <= 0.0
        ):
            self.dhe3_power_density = 0.0
            self.data.physics.fusrat_plasma_dhe3_profile = np.zeros_like(
                self.plasma_profile.neprofile.profile_y
            )
            return

        # Initialize Bosch-Hale constants for the D-3He reaction
        dhe3 = BoschHaleConstants(**REACTION_CONSTANTS_DHE3)

        # Calculate the fusion reaction rate integral using Simpson's rule
        sigmav = integrate.simpson(
            fusion_rate_integral(
                self.plasma_profile, dhe3, physics_data=self.data.physics
            ),
            x=self.plasma_profile.neprofile.profile_x,
            dx=self.plasma_profile.neprofile.profile_dx,
        )

        self.data.physics.fusrat_plasma_dhe3_profile = (
            bosch_hale_reactivity(
                (
                    self.data.physics.temp_plasma_ion_vol_avg_kev
                    / self.data.physics.temp_plasma_electron_vol_avg_kev
                )
                * self.plasma_profile.teprofile.profile_y,
                dhe3,
            )
            * self.data.physics.f_plasma_fuel_deuterium
            * self.data.physics.f_plasma_fuel_helium3
            * (
                self.plasma_profile.neprofile.profile_y
                * (
                    self.data.physics.nd_plasma_fuel_ions_vol_avg
                    / self.data.physics.nd_plasma_electrons_vol_avg
                )
            )
            ** 2
        )

        # Reaction energy in MegaJoules [MJ]
        reaction_energy = constants.D_HELIUM_ENERGY / 1.0e6

        # Calculate the fusion power density produced [MW/m^3]
        fusion_power_density = (
            sigmav
            * reaction_energy
            * (
                self.data.physics.f_plasma_fuel_deuterium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
            * (
                self.data.physics.f_plasma_fuel_helium3
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
        )

        # Power densities for different particles [MW/m^3]
        # Alpha particle gets approximately 20% of the fusion power
        alpha_power_density = (
            1.0 - constants.DHELIUM_PROTON_ENERGY_FRACTION
        ) * fusion_power_density
        pden_non_alpha_charged_mw = (
            constants.DHELIUM_PROTON_ENERGY_FRACTION * fusion_power_density
        )
        neutron_power_density = 0.0

        # Calculate the fusion rate density [reactions/m^3/second]
        fusion_rate_density = fusion_power_density / reaction_energy
        alpha_rate_density = fusion_rate_density
        proton_rate_density = fusion_rate_density  # Proton production rate [m^3/second]

        # Update the cumulative D-3He power density
        self.dhe3_power_density = fusion_power_density

        # Sum the fusion rates for all particles
        self.sum_fusion_rates(
            alpha_power_density,
            pden_non_alpha_charged_mw,
            neutron_power_density,
            fusion_rate_density,
            alpha_rate_density,
            proton_rate_density,
        )

    def dd_helion_reaction(self):
        """D + D --> 3He + n reaction

        This method calculates the fusion reaction rate and power density for the
        deuterium-deuterium (D-D) fusion reaction, specifically the branch that produces
        helium-3 (3He) and a neutron (n). It uses the Bosch-Hale parametrization
        to compute the volumetric fusion reaction rate <sigma v> and integrates over
        the plasma cross-section to find the core plasma fusion power.

        The method updates the following attributes:
            - self.dd_power_density: Fusion power density produced by the D-D reaction.
            - self.alpha_power_density: Power density of alpha particles produced.
            - self.pden_non_alpha_charged_mw: Power density of charged particles
              produced.
            - self.neutron_power_density: Power density of neutrons produced.
            - self.fusion_rate_density: Fusion reaction rate density.
            - self.alpha_rate_density: Alpha particle production rate density.
            - self.proton_rate_density: Proton production rate density.


        """
        if self.data.physics.f_plasma_fuel_deuterium <= 0.0:
            self.data.physics.fusrat_plasma_dd_helion_profile = np.zeros_like(
                self.plasma_profile.neprofile.profile_y
            )
            return

        # Initialize Bosch-Hale constants for the D-D reaction
        dd1 = BoschHaleConstants(**REACTION_CONSTANTS_DD1)

        # Calculate the fusion reaction rate integral using Simpson's rule
        sigmav = integrate.simpson(
            fusion_rate_integral(
                self.plasma_profile,
                dd1,
                physics_data=self.data.physics,
            ),
            x=self.plasma_profile.neprofile.profile_x,
            dx=self.plasma_profile.neprofile.profile_dx,
        )

        self.data.physics.fusrat_plasma_dd_helion_profile = (
            bosch_hale_reactivity(
                (
                    self.data.physics.temp_plasma_ion_vol_avg_kev
                    / self.data.physics.temp_plasma_electron_vol_avg_kev
                )
                * self.plasma_profile.teprofile.profile_y,
                dd1,
            )
            * self.data.physics.f_plasma_fuel_deuterium
            * self.data.physics.f_plasma_fuel_deuterium
            * (
                self.plasma_profile.neprofile.profile_y
                * (
                    self.data.physics.nd_plasma_fuel_ions_vol_avg
                    / self.data.physics.nd_plasma_electrons_vol_avg
                )
            )
            ** 2
        )

        # Reaction energy in MegaJoules [MJ]
        reaction_energy = constants.DD_HELIUM_ENERGY / 1.0e6

        # Calculate the fusion power density produced [MW/m^3]
        # The power density is scaled by the branching ratio to simulate the different
        # product pathways
        fusion_power_density = (
            sigmav
            * reaction_energy
            * (1.0 - self.f_dd_branching_trit)
            * (
                self.data.physics.f_plasma_fuel_deuterium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
            * (
                self.data.physics.f_plasma_fuel_deuterium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
        )

        # Power densities for different particles [MW/m^3]
        # Neutron particle gets approximately 75% of the fusion power
        alpha_power_density = 0.0
        pden_non_alpha_charged_mw = (
            1.0 - constants.DD_NEUTRON_ENERGY_FRACTION
        ) * fusion_power_density
        neutron_power_density = (
            constants.DD_NEUTRON_ENERGY_FRACTION * fusion_power_density
        )

        # Calculate the fusion rate density [reactions/m^3/second]
        fusion_rate_density = fusion_power_density / reaction_energy
        alpha_rate_density = 0.0
        proton_rate_density = 0.0

        # Update the cumulative D-D power density
        self.dd_power_density += fusion_power_density

        # Sum the fusion rates for all particles
        self.sum_fusion_rates(
            alpha_power_density,
            pden_non_alpha_charged_mw,
            neutron_power_density,
            fusion_rate_density,
            alpha_rate_density,
            proton_rate_density,
        )

    def dd_triton_reaction(self):
        """D + D --> T + p reaction

        This method calculates the fusion reaction rate and power density for the
        deuterium-deuterium (D-D) fusion reaction, specifically the branch that produces
        tritium (T) and a proton (p). It uses the Bosch-Hale parametrization
        to compute the volumetric fusion reaction rate <sigma v> and integrates over
        the plasma cross-section to find the core plasma fusion power.

        The method updates the following attributes:
            - self.dd_power_density: Fusion power density produced by the D-D reaction.
            - self.alpha_power_density: Power density of alpha particles produced.
            - self.pden_non_alpha_charged_mw: Power density of charged particles
              produced.
            - self.neutron_power_density: Power density of neutrons produced.
            - self.fusion_rate_density: Fusion reaction rate density.
            - self.alpha_rate_density: Alpha particle production rate density.
            - self.proton_rate_density: Proton production rate density.


        """
        if self.data.physics.f_plasma_fuel_deuterium <= 0.0:
            self.data.physics.fusrat_plasma_dd_triton_profile = np.zeros_like(
                self.plasma_profile.neprofile.profile_y
            )
            return

        # Initialize Bosch-Hale constants for the D-D reaction
        dd2 = BoschHaleConstants(**REACTION_CONSTANTS_DD2)

        # Calculate the fusion reaction rate integral using Simpson's rule
        sigmav = integrate.simpson(
            fusion_rate_integral(
                self.plasma_profile,
                dd2,
                physics_data=self.data.physics,
            ),
            x=self.plasma_profile.neprofile.profile_x,
            dx=self.plasma_profile.neprofile.profile_dx,
        )

        self.data.physics.fusrat_plasma_dd_triton_profile = (
            bosch_hale_reactivity(
                (
                    self.data.physics.temp_plasma_ion_vol_avg_kev
                    / self.data.physics.temp_plasma_electron_vol_avg_kev
                )
                * self.plasma_profile.teprofile.profile_y,
                dd2,
            )
            * self.data.physics.f_plasma_fuel_deuterium
            * self.data.physics.f_plasma_fuel_deuterium
            * (
                self.plasma_profile.neprofile.profile_y
                * (
                    self.data.physics.nd_plasma_fuel_ions_vol_avg
                    / self.data.physics.nd_plasma_electrons_vol_avg
                )
            )
            ** 2
        )

        # Reaction energy in MegaJoules [MJ]
        reaction_energy = constants.DD_TRITON_ENERGY / 1.0e6

        # Calculate the fusion power density produced [MW/m^3]
        # The power density is scaled by the branching ratio to simulate the different
        # product pathways
        fusion_power_density = (
            sigmav
            * reaction_energy
            * self.f_dd_branching_trit
            * (
                self.data.physics.f_plasma_fuel_deuterium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
            * (
                self.data.physics.f_plasma_fuel_deuterium
                * self.data.physics.nd_plasma_fuel_ions_vol_avg
            )
        )

        # Power densities for different particles [MW/m³]
        alpha_power_density = 0.0
        pden_non_alpha_charged_mw = fusion_power_density
        neutron_power_density = 0.0

        # Calculate the fusion rate density [reactions/m³/s]
        fusion_rate_density = fusion_power_density / reaction_energy
        alpha_rate_density = 0.0
        # Proton production rate [particles/m³/s]
        proton_rate_density = fusion_rate_density

        # Update the cumulative D-D power density
        self.dd_power_density += fusion_power_density

        # Sum the fusion rates for all particles
        self.sum_fusion_rates(
            alpha_power_density,
            pden_non_alpha_charged_mw,
            neutron_power_density,
            fusion_rate_density,
            alpha_rate_density,
            proton_rate_density,
        )

    def p11b_reaction(self):
        """p + 11B --> 3 alpha reaction.

        Calculates the p-11B fusion power density using the Maxwellian reactivity
        from :func:`p11b_reactivity` and the volume-averaged proton and boron-11
        densities. The density profile shape is taken to be the same as the
        electron density profile, consistent with the treatment of the other
        fusion reactions in this class.

        All of the reaction energy (8.683 MeV) is carried by three alpha
        particles. The power is therefore added to the alpha particle fusion
        power, and three alpha particles are added to the alpha production rate
        per reaction. No neutron power is produced.

        Notes
        -----
        - Boron-11 is specified through ``f_nd_boron11_electrons`` and protons
          through ``f_nd_protium_electrons`` (both relative to the electron
          density); they are not part of the D/T/3He fuel fractions.
        - The p-11B alpha production rate feeds ``fusden_plasma_alpha`` and hence
          ``fusden_alpha_total`` (used for the alpha confinement time). The
          thermal alpha ash density is still an input (``f_nd_alpha_thermal_electron``)
          and is not solved from the p-11B production rate.
        - The alpha energy split between ions and electrons and the
          ``f_p_alpha_plasma_deposited`` factor are applied to the combined alpha
          power, as for D-T alphas.
        """
        n_proton = self.data.physics.nd_plasma_protons_vol_avg
        n_boron = self.data.physics.nd_plasma_boron11_vol_avg

        # Nothing to do if either reactant is absent (default for D-T cases)
        if n_proton <= 0.0 or n_boron <= 0.0:
            return

        # Ion temperature profile, same convention as the Bosch-Hale reactions
        ion_temperature_profile = (
            self.data.physics.temp_plasma_ion_vol_avg_kev
            / self.data.physics.temp_plasma_electron_vol_avg_kev
        ) * self.plasma_profile.teprofile.profile_y

        # Normalised density profile (shape only, volume average removed)
        density_profile_normalised = (
            1.0 / self.data.physics.nd_plasma_electrons_vol_avg
        ) * self.plasma_profile.neprofile.profile_y

        # Volume-averaged <sigma v>: Maxwellian σ×f table for the selected
        # cross-section model (i_p11b_xsec_model), same 2ρ (n/⟨n⟩)² weight as
        # fusion_rate_integral (Bosch–Hale path).
        model_id = int(self.data.physics.i_p11b_xsec_model)

        def _reactivity(t_kev):
            return p11b_reactivity(t_kev, model=model_id)

        sigmav = integrate.simpson(
            profile_reactivity_integrand(
                self.plasma_profile.teprofile.profile_x,
                ion_temperature_profile,
                density_profile_normalised,
                as_reactivity_model(_reactivity),
            ),
            x=self.plasma_profile.neprofile.profile_x,
            dx=self.plasma_profile.neprofile.profile_dx,
        )

        # Reaction energy [MJ]
        reaction_energy = constants.P_B11_ENERGY / 1.0e6

        # Fusion power density [MW/m^3]
        fusion_power_density = sigmav * n_proton * n_boron * reaction_energy

        # Power densities for the different products [MW/m^3].
        # Same structure as dt_reaction(), but with no neutron branch:
        # all of the energy is carried by the three alphas.
        alpha_power_density = (
            constants.P_B11_ALPHA_ENERGY_FRACTION * fusion_power_density
        )
        pden_non_alpha_charged_mw = 0.0
        neutron_power_density = (
            constants.P_B11_NEUTRON_ENERGY_FRACTION * fusion_power_density
        )

        # Reaction rate density [reactions/m^3/s]
        fusion_rate_density = fusion_power_density / reaction_energy

        # Alpha production rate density [alphas/m^3/s].
        # Three alphas per reaction, each carrying P_B11_ALPHA_ENERGY (~2.894 MeV).
        alpha_rate_density = constants.P_B11_ALPHA_COUNT * fusion_rate_density

        self.p11b_power_density = fusion_power_density
        self.p11b_rate_density = fusion_rate_density
        self.p11b_alpha_rate_density = alpha_rate_density

        # Sum the fusion rates for all particles
        self.sum_fusion_rates(
            alpha_power_density,
            pden_non_alpha_charged_mw,
            neutron_power_density,
            fusion_rate_density,
            alpha_rate_density,
            0.0,
        )

    def sum_fusion_rates(
        self,
        alpha_power_add: float,
        charged_power_add: float,
        neutron_power_add: float,
        fusion_rate_add: float,
        alpha_rate_add: float,
        proton_rate_add: float,
    ):
        """Sum the fusion rate at the end of each reaction.

        This method updates the cumulative fusion power densities and reaction rates
        for alpha particles, charged particles, neutrons, and protons.

        Parameters
        ----------
        alpha_power_add :
            Alpha particle fusion power per unit volume [MW/m³].
        charged_power_add :
            Other charged particle fusion power per unit volume [MW/m³].
        neutron_power_add :
            Neutron fusion power per unit volume [MW/m³]
        fusion_rate_add :
            Fusion reaction rate per unit volume [reactions/m³/s].
        alpha_rate_add :
            Alpha particle production rate per unit volume [particles/m³/s].
        proton_rate_add :
            Proton production rate per unit volume [particles/m³/s].



        """
        self.alpha_power_density += alpha_power_add
        self.pden_non_alpha_charged_mw += charged_power_add
        self.neutron_power_density += neutron_power_add
        self.fusion_rate_density += fusion_rate_add
        self.alpha_rate_density += alpha_rate_add
        self.proton_rate_density += proton_rate_add

    def calculate_fusion_rates(self):
        """Initiate all the fusion rate calculations.

        This method sequentially calculates the fusion reaction rates and power
        densities for the following reactions:
            - Deuterium-Tritium (D-T)
            - Deuterium-Helium-3 (D-3He)
            - Deuterium-Deuterium (D-D) first branch
            - Deuterium-Deuterium (D-D) second branch

        It updates the instance attributes for the cumulative power densities and
        reaction rates
        for alpha particles, charged particles, neutrons, and protons.


        """
        self.dt_reaction()
        self.dhe3_reaction()
        self.dd_helion_reaction()
        self.dd_triton_reaction()
        self.p11b_reaction()

    def set_physics_variables(self):
        """Set the required physics variables in the physics_variables and
        physics modules.

        This method updates the global physics variables and module variables with the
        current instance's fusion power densities and reaction rates.


        """
        self.data.physics.pden_plasma_alpha_mw = self.alpha_power_density
        self.data.physics.pden_non_alpha_charged_mw = self.pden_non_alpha_charged_mw
        self.data.physics.pden_plasma_neutron_mw = self.neutron_power_density
        self.data.physics.fusden_plasma = self.fusion_rate_density
        self.data.physics.fusden_plasma_alpha = self.alpha_rate_density
        self.data.physics.proton_rate_density = self.proton_rate_density
        self.data.physics.sigmav_dt_average = self.sigmav_dt_average
        self.data.physics.dt_power_density_plasma = self.dt_power_density
        self.data.physics.dhe3_power_density = self.dhe3_power_density
        self.data.physics.dd_power_density = self.dd_power_density
        self.data.physics.f_dd_branching_trit = self.f_dd_branching_trit
        self.data.physics.pden_plasma_p11b_mw = self.p11b_power_density
        self.data.physics.fusden_plasma_p11b = self.p11b_rate_density
        self.data.physics.fusden_plasma_p11b_alpha = self.p11b_alpha_rate_density


@dataclass
class BoschHaleConstants:
    """DataClass which holds the constants required for the Bosch Hale calculation
    for a given fusion reaction.
    """

    bg: float
    mrc2: float
    cc1: float
    cc2: float
    cc3: float
    cc4: float
    cc5: float
    cc6: float
    cc7: float


def fusion_rate_integral(
    plasma_profile: PlasmaProfile,
    reaction_constants: BoschHaleConstants,
    physics_data: PhysicsData,
) -> np.ndarray:
    """Evaluate the integrand for the fusion power integration.

    Uses the Bosch–Hale closed-form ⟨σv⟩(T) backend. For reactions that need
    an explicit Maxwellian σ(E) average (e.g. p–¹¹B), call
    :func:`profile_reactivity_integrand` with a :class:`~process.models.physics.reactivity.ReactivityModel`
    built from :func:`~process.models.physics.reactivity.maxwellian_reactivity`
    instead.

    Parameters
    ----------
    plasma_profile :
        Parameterised temperature and density profiles.
    reaction_constants :
        Bosch-Hale reaction constants.
    physics_data: PhysicsData
        physics dataclass

    Returns
    -------
    :
        np.ndarray: Integrand for the fusion power.

    References
    ----------
        - H.-S. Bosch and G. M. Hale, “Improved formulas for fusion cross-sections and
        thermal reactivities,” Nuclear Fusion, vol. 32, no. 4, pp. 611-631, Apr. 1992,
        doi: https://doi.org/10.1088/0029-5515/32/4/i07.
    """
    # Since the electron temperature profile is only calculated directly, we scale the
    # ion temperature profile by the ratio of the volume averaged ion to electron
    # temperature
    ion_temperature_profile = (
        physics_data.temp_plasma_ion_vol_avg_kev
        / physics_data.temp_plasma_electron_vol_avg_kev
    ) * plasma_profile.teprofile.profile_y

    # Set each point in the density profile as a fraction of the volume averaged density
    density_profile_normalised = (
        1.0 / physics_data.nd_plasma_electrons_vol_avg
    ) * plasma_profile.neprofile.profile_y

    # Bosch–Hale parametric ⟨σv⟩(T); same numbers as before, via the shared
    # profile weight so DT / DHe3 / DD and σ×f channels share one integrand.
    return profile_reactivity_integrand(
        plasma_profile.teprofile.profile_x,
        ion_temperature_profile,
        density_profile_normalised,
        as_reactivity_model(
            lambda t: bosch_hale_reactivity(t, reaction_constants)
        ),
    )


def bosch_hale_reactivity(
    ion_temperature_profile: np.ndarray, reaction_constants: BoschHaleConstants
) -> np.ndarray:
    """Calculate the volumetric fusion reaction rate 〈sigmav〉 (m³/s) for one of four
    nuclear reactions using the Bosch-Hale parametrization.

    The valid range of the fit is 0.2 keV < t < 100 keV except for D-3He where it is
    0.5 keV < t < 190 keV.

    Reactions:
        1. D-T reaction
        2. D-3He reaction
        3. D-D 1st reaction
        4. D-D 2nd reaction

    Parameters
    ----------
    ion_temperature_profile :
        Plasma ion temperature profile in keV.
    reaction_constants :
        Bosch-Hale reaction constants.

    Returns
    -------
    :
        np.ndarray: Volumetric fusion reaction rate 〈sigmav〉 in m^3/s for each point in
        the ion temperature profile.

    References
    ----------
        - H.-S. Bosch and G. M. Hale, “Improved formulas for fusion cross-sections and
        thermal reactivities,” Nuclear Fusion, vol. 32, no. 4, pp. 611-631, Apr. 1992,
        doi: https://doi.org/10.1088/0029-5515/32/4/i07.
    """
    theta1 = (
        ion_temperature_profile
        * (
            reaction_constants.cc2
            + ion_temperature_profile
            * (reaction_constants.cc4 + ion_temperature_profile * reaction_constants.cc6)
        )
        / (
            1.0
            + ion_temperature_profile
            * (
                reaction_constants.cc3
                + ion_temperature_profile
                * (
                    reaction_constants.cc5
                    + ion_temperature_profile * reaction_constants.cc7
                )
            )
        )
    )
    theta = ion_temperature_profile / (1.0 - theta1)

    xi = ((reaction_constants.bg**2) / (4.0 * theta)) ** (1 / 3)

    # Volumetric reaction rate / reactivity 〈sigmav〉 (m³/s)
    # Original form is in [cm³/s], so multiply by 1.0e-6 to convert to [m³/s]
    sigmav = (
        1.0e-6
        * reaction_constants.cc1
        * theta
        * np.sqrt(xi / (reaction_constants.mrc2 * ion_temperature_profile**3))
        * np.exp(-3.0 * xi)
    )

    # if t = 0, sigmav = 0. Use this mask to set sigmav to zero.
    t_mask = ion_temperature_profile == 0.0  # noqa: RUF069
    sigmav[t_mask] = 0.0

    # Outside the Bosch-Hale fit range (or with pathological profiles) xi/theta
    # can become negative and produce NaN/Inf. Never propagate those.
    sigmav = np.nan_to_num(sigmav, nan=0.0, posinf=0.0, neginf=0.0)

    # Return np.ndarray of sigmav for each point in the ion temperature profile
    return sigmav


def set_fusion_powers(
    f_alpha_electron: float,
    f_alpha_ion: float,
    p_beam_alpha_mw: float,
    pden_non_alpha_charged_mw: float,
    pden_plasma_neutron_mw: float,
    vol_plasma: float,
    pden_plasma_alpha_mw: float,
    f_p_alpha_plasma_deposited: float,
) -> tuple:
    """Computes various fusion power metrics based on the provided
    plasma parameters.

    Parameters
    ----------
    f_alpha_electron :
        float
    f_alpha_ion :
        float
    p_beam_alpha_mw :
        float
    pden_non_alpha_charged_mw :
        float
    pden_plasma_neutron_mw :
        float
    vol_plasma :
        float
    pden_plasma_alpha_mw :
        float
    f_p_alpha_plasma_deposited: float
        Fraction of alpha power deposited in plasma

    Returns
    -------
    :
        tuple: A tuple containing the following elements:
        - pden_neutron_total_mw (float): Neutron fusion power per unit volume from
          plasma and beams [MW/m³].
        - p_plasma_alpha_mw (float): Alpha fusion power from only the plasma [MW].
        - p_alpha_total_mw (float): Total alpha fusion power from plasma and beams [MW].
        - p_plasma_neutron_mw (float): Neutron fusion power from only the plasma [MW].
        - p_neutron_total_mw (float): Total neutron fusion power from plasma and
          beams [MW].
        - p_non_alpha_charged_mw (float): Other total charged particle fusion power [MW].
        - pden_alpha_total_mw (float): Alpha power per unit volume, from beams and
          plasma [MW/m³].
        - f_pden_alpha_electron_mw (float): Alpha power per unit volume to
          electrons [MW/m³].
        - f_pden_alpha_ions_mw (float): Alpha power per unit volume to ions [MW/m³].
        - p_charged_particle_mw (float): Charged particle fusion power [MW].
        - p_fusion_total_mw (float): Total fusion power [MW].

    References
    ----------
        - N.A. Uckan and ITER Physics Group, 'ITER Physics Design Guidelines: 1989'
        - ITER Documentation Series No.10, IAEA/ITER/DS/10, IAEA, Vienna, 1990
    """
    # Alpha power

    # Calculate alpha power produced just by the plasma
    p_plasma_alpha_mw = pden_plasma_alpha_mw * vol_plasma

    # Add neutral beam alpha power / volume
    pden_alpha_total_mw = pden_plasma_alpha_mw + (p_beam_alpha_mw / vol_plasma)

    # Total alpha power
    p_alpha_total_mw = pden_alpha_total_mw * vol_plasma

    # Neutron Power

    # Calculate neutron power produced just by the plasma
    p_plasma_neutron_mw = pden_plasma_neutron_mw * vol_plasma

    # Add extra neutron power from beams
    pden_neutron_total_mw = pden_plasma_neutron_mw + (
        (
            (
                constants.DT_NEUTRON_ENERGY_FRACTION
                / (1.0 - constants.DT_NEUTRON_ENERGY_FRACTION)
            )
            * p_beam_alpha_mw
        )
        / vol_plasma
    )

    # Total neutron power
    p_neutron_total_mw = pden_neutron_total_mw * vol_plasma

    # Charged particle power

    # Total non-alpha charged particle power
    p_non_alpha_charged_mw = pden_non_alpha_charged_mw * vol_plasma

    # Charged particle fusion power
    p_charged_particle_mw = p_alpha_total_mw + p_non_alpha_charged_mw

    # Total fusion power
    p_fusion_total_mw = p_alpha_total_mw + p_neutron_total_mw + p_non_alpha_charged_mw

    # Alpha power to electrons and ions (used with electron
    # and ion power balance equations only)
    # No consideration of pden_non_alpha_charged_mw here.
    f_pden_alpha_ions_mw = f_p_alpha_plasma_deposited * pden_alpha_total_mw * f_alpha_ion
    f_pden_alpha_electron_mw = (
        f_p_alpha_plasma_deposited * pden_alpha_total_mw * f_alpha_electron
    )

    return (
        pden_neutron_total_mw,
        p_plasma_alpha_mw,
        p_alpha_total_mw,
        p_plasma_neutron_mw,
        p_neutron_total_mw,
        p_non_alpha_charged_mw,
        pden_alpha_total_mw,
        f_pden_alpha_electron_mw,
        f_pden_alpha_ions_mw,
        p_charged_particle_mw,
        p_fusion_total_mw,
    )


def beam_fusion(
    beamfus0: float,
    betbm0: float,
    b_plasma_total: float,
    c_beam_total: float,
    nd_plasma_electrons_vol_avg: float,
    nd_plasma_fuel_ions_vol_avg: float,
    ion_electron_coulomb_log: float,
    e_beam_kev: float,
    f_deuterium_plasma: float,
    f_tritium_plasma: float,
    f_beam_tritium: float,
    temp_plasma_electron_density_weighted_kev: float,
    vol_plasma: float,
    n_charge_plasma_effective_mass_weighted_vol_avg: float,
) -> tuple[float, float, float]:
    """Calculate neutral beam slowing-down properties and beam-target fusion.

    This function computes the neutral beam beta contribution, hot beam ion density,
    and alpha power generated from beam-target fusion reactions in the plasma.

    Parameters
    ----------
    beamfus0 :
        Multiplier for beam target fusion calculation.
    betbm0 :
        Leading coefficient for neutral beam beta fraction.
    b_plasma_total :
        Total magnetic field (T).
    c_beam_total :
        Total neutral beam current (A).
    nd_plasma_electrons_vol_avg :
        Volume-averaged electron density (m^-3).
    nd_plasma_fuel_ions_vol_avg :
        Volume-averaged fuel ion density (m^-3).
    ion_electron_coulomb_log :
        Ion-electron Coulomb logarithm.
    e_beam_kev :
        Neutral beam injection energy (keV).
    f_deuterium_plasma :
        Deuterium fraction in the bulk plasma.
    f_tritium_plasma :
        Tritium fraction in the bulk plasma.
    f_beam_tritium :
        Tritium fraction in the injected beam.
    temp_plasma_electron_density_weighted_kev :
        Density weighted average electron temperature (keV).
    vol_plasma :
        Plasma volume (m^3).
    n_charge_plasma_effective_mass_weighted_vol_avg :
        Mass-weighted effective plasma charge.

    Returns
    -------
    beta_beam :
        Neutral beam beta contribution.
    n_beam :
        Hot beam ion density (m^-3).
    p_alpha_beam :
        Alpha power from beam-target fusion (MW).

    Notes
    -----
        - The function uses the Bosch-Hale parametrization to evaluate fusion reactivity.
        - Beam ion slowing-down properties are computed using Coulomb collision physics.
        - Fusion power is obtained by integrating the beam-target reaction rate over
        the fast-ion distribution.

    References
    ----------
        - H.-S. Bosch and G. M. Hale, “Improved formulas for fusion cross-sections and
        thermal reactivities,” Nuclear Fusion, vol. 32, no. 4, pp. 611-631, Apr. 1992,
        doi: https://doi.org/10.1088/0029-5515/32/4/i07.

        - J. W. Sheffield, “The physics of magnetic fusion reactors,”
        Rev. Mod. Phys., vol. 66, no. 3, pp. 1015-1103, Jul. 1994, Jul. 1994,
        doi: https://doi.org/10.1103/revmodphys.66.1015.

        - Deng Baiquan and G. A. Emmert, “Fast ion pressure in fusion plasma,”
        Nuclear Fusion and Plasma Physics, vol. 9, no. 3, pp. 136-141, 2022.
        Available: https://fti.neep.wisc.edu/fti.neep.wisc.edu/pdf/fdm718.pdf
    """
    # Beam ion slowing down time given by Deng Baiquan and G. A. Emmert 1987
    t_beam_slow = (
        1.99e19
        * (
            constants.M_DEUTERON_AMU * (1.0 - f_beam_tritium)
            + (constants.M_TRITON_AMU * f_beam_tritium)
        )
        * (temp_plasma_electron_density_weighted_kev**1.5 / nd_plasma_electrons_vol_avg)
        / ion_electron_coulomb_log
    )

    # Critical energy for electron/ion slowing down of the beam ion
    # (deuterium and tritium neutral beams, respectively) (keV)
    # Taken from J.W Sheffield, “The physics of magnetic fusion reactors,”
    critical_energy_deuterium = (
        14.8
        * constants.M_DEUTERON_AMU
        * temp_plasma_electron_density_weighted_kev
        * n_charge_plasma_effective_mass_weighted_vol_avg ** (2 / 3)
        * (ion_electron_coulomb_log + 4.0)
        / ion_electron_coulomb_log
    )
    critical_energy_tritium = critical_energy_deuterium * (
        constants.M_TRITON_AMU / constants.M_DEUTERON_AMU
    )

    # Deuterium and tritium ion densities
    nd_plasma_deuterium = nd_plasma_fuel_ions_vol_avg * f_deuterium_plasma
    nd_plasma_tritium = nd_plasma_fuel_ions_vol_avg * f_tritium_plasma

    beam_state = beam_slowing_down_state(
        e_beam_kev,
        critical_energy_deuterium,
        critical_energy_tritium,
        t_beam_slow,
        f_beam_tritium,
        c_beam_total,
        vol_plasma,
    )

    sigv_beam_deuterium = beam_reaction_rate_coefficient(
        constants.M_DEUTERON_AMU,
        beam_state.deuterium_critical_energy_speed,
        e_beam_kev,
    )

    sigv_beam_tritium = beam_reaction_rate_coefficient(
        constants.M_TRITON_AMU,
        beam_state.tritium_critical_energy_speed,
        e_beam_kev,
    )

    reaction_rate_beam_deuterium_dt = beam_target_reaction_rate(
        beam_state.deuterium_beam_density,
        nd_plasma_tritium,
        sigv_beam_deuterium,
        vol_plasma,
    )
    p_alpha_beam_deuterium_dt_mw = alpha_power_beam(
        reaction_rate_beam_deuterium_dt,
    )

    reaction_rate_beam_tritium_dt = beam_target_reaction_rate(
        beam_state.tritium_beam_density,
        nd_plasma_deuterium,
        sigv_beam_tritium,
        vol_plasma,
    )
    p_alpha_beam_tritium_dt_mw = alpha_power_beam(
        reaction_rate_beam_tritium_dt,
    )

    # Neutral beam alpha power
    p_beam_alpha_mw = beamfus0 * (
        p_alpha_beam_deuterium_dt_mw + p_alpha_beam_tritium_dt_mw
    )

    # Neutral beam beta
    beta_beam = (
        betbm0
        * 4.03e-22
        * (2 / 3)
        * beam_state.nd_beam_hot
        * beam_state.e_beam_deposited_kev
        / (b_plasma_total**2)
    )

    return beta_beam, beam_state.nd_beam_hot, p_beam_alpha_mw


@dataclass(frozen=True)
class BeamSlowingDownState:
    """Beam slowing-down state and hot-ion properties.

    Attributes
    ----------
    deuterium_beam_density :
        Steady-state deuterium beam ion density (m^-3).
    tritium_beam_density :
        Steady-state tritium beam ion density (m^-3).
    deuterium_critical_energy_speed :
        Critical speed of deuterium beam ions corresponding to the
        critical energy (m/s).
    tritium_critical_energy_speed :
        Critical speed of tritium beam ions corresponding to the
        critical energy (m/s).
    nd_beam_hot :
        Total hot beam ion density (m^-3).
    e_beam_deposited_kev :
        Density-weighted average deposited beam ion energy (keV).
    """

    deuterium_beam_density: float
    tritium_beam_density: float
    deuterium_critical_energy_speed: float
    tritium_critical_energy_speed: float
    nd_beam_hot: float
    e_beam_deposited_kev: float


def beam_slowing_down_state(
    e_beam_kev: float,
    critical_energy_deuterium: float,
    critical_energy_tritium: float,
    t_beam_slow: float,
    f_beam_tritium: float,
    c_beam_total: float,
    vol_plasma: float,
) -> BeamSlowingDownState:
    """Calculate beam slowing-down state and hot ion properties.

    Estimates the steady-state hot D and T beam-ion densities from beam
    source rate times slowing-down residence time, computes species critical
    velocities from critical energies, evaluates a slowing-down-distribution
    fast-ion pressure for each species, and converts that pressure into a
    pressure-equivalent mean fast-ion energy.

    Parameters
    ----------
    e_beam_kev :
        Beam energy (keV).
    critical_energy_deuterium :
        Critical energy for electron/ion slowing down of deuterium beam ions (keV).
    critical_energy_tritium :
        Critical energy for electron/ion slowing down of tritium beam ions (keV).
    t_beam_slow :
        Beam ion slowing-down time on electrons (s).
    f_beam_tritium :
        Tritium fraction in the injected beam (0.0 = pure deuterium beam).
    c_beam_total :
        Total neutral beam current (A).
    vol_plasma :
        Plasma volume (m^3).

    Returns
    -------
    :
        Beam slowing-down state containing deuterium and tritium hot beam-ion
        densities, deuterium and tritium critical speeds, total hot beam-ion
        density, and average deposited beam-ion energy.

    Notes
    -----
        - The critical energy for electron/ion slowing down of the beam ion is calculated
        for both deuterium and tritium neutral beams.
        - The function integrates the hot beam fusion reaction rate integrand over the
        range of beam velocities up to the critical velocity.

    References
    ----------
        - H.-S. Bosch and G. M. Hale, “Improved formulas for fusion cross-sections
        and thermal reactivities,” Nuclear Fusion, vol. 32, no. 4, pp. 611-631,
        Apr. 1992, doi: https://doi.org/10.1088/0029-5515/32/4/i07.

        - Deng Baiquan and G. A. Emmert, “Fast ion pressure in fusion plasma,”
        Nuclear Fusion and Plasma Physics, vol. 9, no. 3, pp. 136-141, 2022,
        Available: https://fti.neep.wisc.edu/fti.neep.wisc.edu/pdf/fdm718.pdf

        - Wesson, J. (2011) Tokamaks. 4th Edition, 2011 Oxford Science Publications,
        International Series of Monographs on Physics, Volume 149.

        - J. W. Sheffield, “The physics of magnetic fusion reactors,” vol. 66, no. 3,
        pp. 1015-1103, Jul. 1994, doi: https://doi.org/10.1103/revmodphys.66.1015.
    """
    # D and T beam current fractions
    beam_current_deuterium = c_beam_total * (1.0 - f_beam_tritium)
    beam_current_tritium = c_beam_total * f_beam_tritium

    # At a critical energy the rate of loss to the ions becomes equal to that to
    # the electrons, and at lower energies the loss to the ions predominates.

    # Ratio of beam energy to critical energy for deuterium
    beam_energy_ratio_deuterium = e_beam_kev / critical_energy_deuterium

    # Calculate the characterstic time for the deuterium ions to slow down to the
    # thermal energy, eg E = 0.
    characteristic_deuterium_t_beam_slow = (
        t_beam_slow / 3.0 * np.log(1.0 + (beam_energy_ratio_deuterium) ** 1.5)
    )

    deuterium_beam_density = (
        beam_current_deuterium
        * characteristic_deuterium_t_beam_slow
        / (constants.ELECTRON_CHARGE * vol_plasma)
    )

    # Ratio of beam energy to critical energy for tritium
    beam_energy_ratio_tritium = e_beam_kev / critical_energy_tritium

    # Calculate the characterstic time for the tritium to slow down to the thermal
    # energy, eg E = 0.
    # Wesson, J. (2011) Tokamaks.
    characteristic_tritium_t_beam_slow = (
        t_beam_slow / 3.0 * np.log(1.0 + (beam_energy_ratio_tritium) ** 1.5)
    )

    tritium_beam_density = (
        beam_current_tritium
        * characteristic_tritium_t_beam_slow
        / (constants.ELECTRON_CHARGE * vol_plasma)
    )

    nd_beam_hot = deuterium_beam_density + tritium_beam_density

    # Find the speed of the deuterium particle when it has the critical energy.
    # Re-arrange kinetic energy equation to find speed. Non-relativistic.
    deuterium_critical_energy_speed = np.sqrt(
        2.0
        * constants.KILOELECTRON_VOLT
        * critical_energy_deuterium
        / (constants.ATOMIC_MASS_UNIT * constants.M_DEUTERON_AMU)
    )

    # Find the speed of the tritium particle when it has the critical energy.
    # Re-arrange kinetic energy equation to find speed. Non-relativistic.
    tritium_critical_energy_speed = np.sqrt(
        2.0
        * constants.KILOELECTRON_VOLT
        * critical_energy_tritium
        / (constants.ATOMIC_MASS_UNIT * constants.M_TRITON_AMU)
    )

    # Source term representing the number of ions born per unit time per unit volume.
    # D.Baiquan et.al.  “Fast ion pressure in fusion plasma,” Nuclear Fusion and
    # Plasma Physics, vol. 9, no. 3, pp. 136-141, 2022,
    # Available: https://fti.neep.wisc.edu/fti.neep.wisc.edu/pdf/fdm718.pdf

    source_deuterium = beam_current_deuterium / (constants.ELECTRON_CHARGE * vol_plasma)

    source_tritium = beam_current_tritium / (constants.ELECTRON_CHARGE * vol_plasma)

    pressure_coeff_deuterium = (
        constants.M_DEUTERON_AMU
        * constants.ATOMIC_MASS_UNIT
        * t_beam_slow
        * deuterium_critical_energy_speed**2
        * source_deuterium
        / (constants.KILOELECTRON_VOLT * 3.0)
    )
    pressure_coeff_tritium = (
        constants.M_TRITON_AMU
        * constants.ATOMIC_MASS_UNIT
        * t_beam_slow
        * tritium_critical_energy_speed**2
        * source_tritium
        / (constants.KILOELECTRON_VOLT * 3.0)
    )

    # Fast Ion Pressure
    # This is the same form as the ideal gas law pressure, P=1/3 * nmv²
    deuterium_pressure = pressure_coeff_deuterium * fast_ion_pressure_integral(
        e_beam_kev, critical_energy_deuterium
    )
    tritium_pressure = pressure_coeff_tritium * fast_ion_pressure_integral(
        e_beam_kev, critical_energy_tritium
    )

    # Beam deposited energy
    # Find the energy from the ideal gas pressure, P=1/3 * nmv² = 2/3 * n<E>
    deuterium_deposited_energy = (
        1.5 * deuterium_pressure / deuterium_beam_density
        if deuterium_beam_density > 0.0
        else 0.0
    )
    tritium_deposited_energy = (
        1.5 * tritium_pressure / tritium_beam_density
        if tritium_beam_density > 0.0
        else 0.0
    )

    e_beam_deposited_kev = (
        (
            deuterium_beam_density * deuterium_deposited_energy
            + tritium_beam_density * tritium_deposited_energy
        )
        / nd_beam_hot
        if nd_beam_hot > 0.0
        else 0.0
    )

    return BeamSlowingDownState(
        deuterium_beam_density=deuterium_beam_density,
        tritium_beam_density=tritium_beam_density,
        deuterium_critical_energy_speed=deuterium_critical_energy_speed,
        tritium_critical_energy_speed=tritium_critical_energy_speed,
        nd_beam_hot=nd_beam_hot,
        e_beam_deposited_kev=e_beam_deposited_kev,
    )


def fast_ion_pressure_integral(e_beam_kev: float, critical_energy: float) -> float:
    """Evaluate the dimensionless fast-ion pressure integral.

    This function returns the closed-form dimensionless integral factor
    appearing in the analytic fast-ion pressure expression for a
    monoenergetic source with a classical slowing-down distribution.

    Parameters
    ----------
    e_beam_kev :
        Beam birth energy (keV).
    critical_energy :
        Critical energy for electron/ion slowing down (keV).

    Returns
    -------
    :
        Dimensionless pressure integral factor.

    Notes
    -----
        - The function uses the ratio of the beam energy to the critical energy to
          compute the hot ion energy parameter.
        - The calculation involves logarithmic and arctangent functions to account for
          the energy distribution of the hot ions.

    References
    ----------
        - Deng Baiquan and G. A. Emmert, “Fast ion pressure in fusion plasma,”
        Nuclear Fusion and Plasma Physics,
        vol. 9, no. 3, pp. 136-141, 2022,
        Available: https://fti.neep.wisc.edu/fti.neep.wisc.edu/pdf/fdm718.pdf

        - W.A Houlberg, “Thermalization of an Energetic Heavy Ion in a Multi-species
        Plasma,” University of Wisconsin Fusion Technology Institute,
        Report UWFDM-103 1974,
        Available: https://fti.neep.wisc.edu/fti.neep.wisc.edu/pdf/fdm103.pdf
    """
    xcs = e_beam_kev / critical_energy
    xc = np.sqrt(xcs)

    t1 = xcs / 2.0
    t2 = np.log((xcs + 2.0 * xc + 1.0) / (xcs - xc + 1.0)) / 6.0

    xarg = (2.0 * xc - 1.0) / np.sqrt(3.0)
    t3 = np.arctan(xarg) / np.sqrt(3.0)
    t4 = (1 / np.sqrt(3.0)) * np.arctan(1 / np.sqrt(3.0))

    return t1 + t2 - t3 - t4


def beam_target_reaction_rate(
    nd_beam_ion: float,
    nd_target_ion: float,
    sigv_beam: float,
    vol_plasma: float,
) -> float:
    """Calculate the total beam-target fusion reaction rate.

    This function computes the total number of beam-target fusion reactions
    occurring per second in the plasma.

    The total beam-target reaction rate is given by:
        reaction_rate = nd_beam_ion * nd_target_ion * sigv_beam * vol_plasma

    Unit analysis:
        (m^-3) * (m^-3) * (m^3/s) * (m^3) = s^-1

    Parameters
    ----------
    nd_beam_ion :
        Hot beam ion density (m^-3).
    nd_target_ion :
        Thermal target ion density (m^-3).
    sigv_beam :
        Beam-target fusion reactivity (m^3/s).
    vol_plasma :
        Plasma volume (m^3).

    Returns
    -------
    :
        Total beam-target fusion reaction rate (s^-1).

    Notes
    -----
    Any spatial/profile effects should be introduced through the
    supplied nd_target_ion.
    """
    return nd_beam_ion * nd_target_ion * sigv_beam * vol_plasma


def alpha_power_beam(
    beam_target_reaction_rate: float,
) -> float:
    """Calculate alpha power from beam-target fusion.

    This function converts the total beam-target fusion reaction rate into
    alpha particle power.

    The alpha power is given by:
        P_alpha = beam_target_reaction_rate * DT_ALPHA_ENERGY

    Unit analysis:
        (s^-1) * (J) = W

    The result is converted from watts to megawatts (MW).

    Parameters
    ----------
    beam_target_reaction_rate :
        Total beam-target fusion reaction rate (s^-1).

    Returns
    -------
    :
        Alpha power from beam-target fusion (MW).
    """
    return beam_target_reaction_rate * constants.DT_ALPHA_ENERGY / 1.0e6


def beam_reaction_rate_coefficient(
    relative_mass_ion: float, critical_velocity: float, beam_energy_kev: float
) -> float:
    """Calculate an effective beam-target fusion rate coefficient.

    This function evaluates an effective fusion rate coefficient for a
    slowing-down fast-ion population interacting with a thermal target
    plasma. The beam ions are assumed to follow a classical slowing-down
    distribution characterised by the critical velocity, and the fusion
    contribution is obtained by integrating a beam-energy-dependent fusion
    cross-section over the slowing-down distribution.

    Parameters
    ----------
    relative_mass_ion :
        Relative atomic mass of the ion (e.g. approx 2.0 for D, 3.0 for T).
    critical_velocity :
        Critical velocity for electron/ion slowing down of the beam ion [m/s].
    beam_energy_kev :
        Neutral beam energy [keV].

    Returns
    -------
    :
        Hot beam fusion rate coefficient [m^3/s].

    Notes
    -----
    - The function returns a reactivity rate coefficient, not the reaction rate itself.
    - The integration variable is the beam speed normalised to the critical speed.
    - The underlying beam fusion cross-section is evaluated in cm^2 and converted
      internally to m^2 so that this function returns an SI rate coefficient.
    - The integration is performed using scipy.integrate.quad.

    References
    ----------
    - S. Niikura and M. Nagami, "Improvement of fusion reactivity and fusion
      power multiplication factor in the presence of fast ions," Fusion
      Engineering and Design, vol. 12, pp. 467-480, 1990.
    """
    # Find the speed of the beam particle when it has the critical energy.
    # Re-arrange kinetic energy equation to find speed. Non-relativistic.
    beam_velocity = np.sqrt(
        (beam_energy_kev * constants.KILOELECTRON_VOLT)
        * 2.0
        / (relative_mass_ion * constants.ATOMIC_MASS_UNIT)
    )

    relative_velocity = beam_velocity / critical_velocity
    integral_coefficient = 3.0 * critical_velocity / np.log(1.0 + relative_velocity**3)

    fusion_integral_cm2 = integrate.quad(
        _hot_beam_fusion_reaction_rate_integrand,
        0.0,
        relative_velocity,
        args=(critical_velocity,),
    )[0]

    # _hot_beam_fusion_cross_section returns cm^2, so the integrated quantity is in
    # cm^2. Convert to m^2 here so the final rate coefficient is returned in m^3/s.
    fusion_integral_m2 = fusion_integral_cm2 * 1.0e-4

    return integral_coefficient * fusion_integral_m2


def _hot_beam_fusion_reaction_rate_integrand(
    velocity_ratio: float, critical_velocity: float
) -> float:
    """Evaluate the integrand for the beam-target fusion rate coefficient.

    This function returns the slowing-down-weighted integrand used to
    compute the effective beam-target fusion rate coefficient for a fast-ion
    population. The integration variable is the beam speed normalised to the
    critical speed, and the integrand combines a slowing-down weighting with
    the beam fusion cross-section evaluated at the corresponding beam energy.

    Parameters
    ----------
    velocity_ratio :
        Ratio of beam velocity to the critical velocity.
    critical_velocity :
        Critical velocity for electron/ion slowing down of the beam ion (m/s).


    Returns
    -------
    :
        Value of the integrand for the hot beam fusion reaction rate.

    Notes
    -----
    - The factor `velocity_ratio**3 / (1.0 + velocity_ratio**3)` represents
      the slowing-down weighting in the normalised velocity variable.
    - The beam energy passed to `_beam_fusion_cross_section` is constructed
      from the instantaneous beam speed and expressed in keV/amu.
    - This function is an internal helper used by
      `beam_reaction_rate_coefficient`.

    References
    ----------
    - S. Niikura and M. Nagami, "Improvement of fusion reactivity and fusion
      power multiplication factor in the presence of fast ions," Fusion
      Engineering and Design, vol. 12, pp. 467-480, 1990.

    """
    intgeral_term = (velocity_ratio**3) / (1.0 + velocity_ratio**3)

    # critical_velocity : critical velocity for electron/ion slowing down of beam
    # ion (m/s)
    beam_velcoity = critical_velocity * velocity_ratio

    # Calculate the beam kinetic energy per amu and normalise to keV
    xvcs = beam_velcoity**2 * constants.ATOMIC_MASS_UNIT / (constants.KILOELECTRON_VOLT)

    # Calculate the fusion reaction cross-section from beam kinetic energy
    cross_section = _beam_fusion_cross_section(xvcs)

    return intgeral_term * cross_section


def _beam_fusion_cross_section(vrelsq: float) -> float:
    """Calculate the beam fusion cross-section from beam energy.

    This function computes the fusion reaction cross-section based on the
    square of the speed of the beam ion (keV/amu). The functional form of
    the cross-section is in terms of the equivalent deuterium energy, i.e.,
    for a tritium beam at 500 keV the energy used in the cross-section
    function is 333 keV.

    Parameters
    ----------
    vrelsq :
        Square of the speed of the beam ion (keV/amu).

    Returns
    -------
    :
        Fusion reaction cross-section (cm²).

    Notes
    -----
    - The cross-section is limited at low and high beam energies.
    - For beam kinetic energy less than 10 keV, the cross-section is set to 1.0e-27 cm².
    - For beam kinetic energy greater than 10,000 keV, the cross-section is set to
      8.0e-26 cm².
    - The cross-section is calculated using a functional form with parameters a1 to a5.
    - The exact provenance of this particular fit and its coefficients has
      not yet been traced in the present code documentation.
    """
    a1 = 45.95
    a2 = 5.02e4
    a3 = 1.368e-2
    a4 = 1.076
    a5 = 4.09e2

    # Beam kinetic energy
    e_beam_kev = 0.5 * constants.M_DEUTERON_AMU * vrelsq

    # Set limits on cross-section at low and high beam energies
    if e_beam_kev < 10.0:
        return 1.0e-27
    if e_beam_kev > 1.0e4:
        return 8.0e-26
    t1 = a2 / (1.0 + (a3 * e_beam_kev - a4) ** 2) + a5
    t2 = e_beam_kev * (np.exp(a1 / np.sqrt(e_beam_kev)) - 1.0)
    return 1.0e-24 * t1 / t2


# =============================================================================
# Proton - Boron-11 reactivity
# =============================================================================
#
# p + 11B --> 3 alpha + 8.683 MeV
#
# Cross-section models live in :mod:`process.models.physics.p11b_cross_sections`
# (plugin registry; default Wang et al. 2026). Unlike Bosch–Hale (closed-form
# ⟨σv⟩ fits for DT/DHe3/DD), p–¹¹B uses the shared Maxwellian σ×f integrator
# in :mod:`process.models.physics.reactivity`:
#     <sigma v> = sqrt(8 kT / (pi mu)) * integral_0^inf sigma(x kT) x e^-x dx.
#
# Reactivity tables are cached per ``i_p11b_xsec_model`` on 0.5–3000 keV.

_P11B_T_MIN_KEV = 0.5
_P11B_T_MAX_KEV = 3000.0
_P11B_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_P11B_T_CLIP_WARNED: set[int] = set()


def p11b_s_factor(energy_mev, model=None):
    """Astrophysical S-factor of p-11B [MeV b].

    Parameters
    ----------
    energy_mev :
        Centre-of-mass energy [MeV], scalar or array.
    model :
        ``i_p11b_xsec_model`` index, model key, or
        :class:`~process.models.physics.p11b_cross_sections.P11BCrossSectionModel`.
        Default = Wang 2026.

    Returns
    -------
    :
        S-factor [MeV b], zero outside the model's fitted energy range.
    """
    return get_p11b_cross_section_model(model).s_factor(energy_mev)


def p11b_cross_section(energy_mev, model=None, *, warn_out_of_range: bool = True):
    """p-11B fusion cross-section [m^2] for centre-of-mass energy in MeV."""
    return get_p11b_cross_section_model(model).cross_section(
        energy_mev, warn_out_of_range=warn_out_of_range
    )


def _p11b_reactivity_direct(t_kev, model=None):
    """Maxwellian p-11B reactivity [m^3/s] via σ(E)×Maxwellian quadrature."""
    xsec = get_p11b_cross_section_model(model)

    def _sigma(energy_mev):
        return xsec.cross_section(energy_mev, warn_out_of_range=False)

    return maxwellian_reactivity(
        t_kev, _sigma, xsec.reduced_mass_mev, n_quad=160
    )


def _p11b_reactivity_table(model=None):
    """Build (and cache per model) the p-11B reactivity interpolation table."""
    xsec = get_p11b_cross_section_model(model)
    model_id = xsec.index
    if model_id not in _P11B_TABLES:

        def _sigma(energy_mev):
            return xsec.cross_section(energy_mev, warn_out_of_range=False)

        _P11B_TABLES[model_id] = tabulate_reactivity(
            _sigma,
            xsec.reduced_mass_mev,
            _P11B_T_MIN_KEV,
            _P11B_T_MAX_KEV,
            n_points=400,
            n_quad=160,
        )
    return _P11B_TABLES[model_id]


def clear_p11b_reactivity_tables() -> None:
    """Drop cached reactivity tables (for tests / model reloads)."""
    _P11B_TABLES.clear()
    _P11B_T_CLIP_WARNED.clear()


def p11b_reactivity(t_kev, model=None):
    """Maxwellian p-11B reactivity <sigma v> [m^3/s].

    Uses the shared :func:`~process.models.physics.reactivity.maxwellian_reactivity`
    integrator (velocity distribution × cross-section at given T), tabulated
    per cross-section model for optimiser speed.

    Parameters
    ----------
    t_kev :
        Ion temperature [keV], scalar or array. Values outside the tabulated
        0.5-3000 keV range are clipped (with a one-shot warning per model).
    model :
        ``i_p11b_xsec_model`` index / key / model instance. Default = Wang 2026.

    Returns
    -------
    :
        Reactivity [m^3/s].
    """
    xsec = get_p11b_cross_section_model(model)
    temperatures, reactivities = _p11b_reactivity_table(xsec)
    query = np.asarray(t_kev, dtype=float)
    if np.any(query < temperatures[0]) or np.any(query > temperatures[-1]):
        if xsec.index not in _P11B_T_CLIP_WARNED:
            _P11B_T_CLIP_WARNED.add(xsec.index)
            logger.warning(
                "p-11B reactivity model %s (%s): ion temperature clipped to "
                "[%.1f, %.1f] keV tabulated range.",
                xsec.index,
                xsec.key,
                temperatures[0],
                temperatures[-1],
            )
    return interpolate_reactivity_loglog(t_kev, temperatures, reactivities)
