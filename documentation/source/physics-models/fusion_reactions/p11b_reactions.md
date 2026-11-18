# Proton–Boron-11 Fusion

## Overview

PROCESS can calculate the p-¹¹B reaction in addition to D-T, D-³He and D-D:

$$
\mathrm{p} + {}^{11}\mathrm{B} \longrightarrow 3\,^{4}\mathrm{He} + 8.683\ \mathrm{MeV}
$$

All of the reaction energy is carried by charged particles (three alpha
particles), so the reaction produces **no neutron power**. The 8.683 MeV
Q-value is used from the evaluated nuclear data; it is stored as
`constants.P_B11_ENERGY`.

Unlike D-T and D-³He, p-¹¹B reactants are not part of the D/T/³He fuel
fractions. They are specified separately:

| Input | Meaning | Default |
|---|---|---|
| `f_nd_protium_electrons` | proton density / electron density | 0.0 |
| `f_nd_boron11_electrons` | boron-11 density / electron density | 0.0 |

Both species are included in the quasi-neutrality calculation: boron-11
contributes a charge of 5 to the ion charge sum, so the D/T/³He fuel density
is reduced accordingly. Boron-11 is also added to the total ion density,
the ion mass and Z<sub>eff</sub> (through 25·n<sub>B11</sub>/n<sub>e</sub>).

## Cross-section models | `i_p11b_xsec_model`

S-factor / σ(E) models are registered in
`process.models.physics.p11b_cross_sections` and selected by input
`i_p11b_xsec_model` (default **0**):

| Index | Key | Reference | Notes |
|---|---|---|---|
| 0 | `wang_2026` | arXiv:2601.00241 | Default; includes 4.7 MeV resonance |
| 1 | `nevins_swain_2000` | Nucl. Fusion 40, 865 | Historical; E ≤ 3.5 MeV |
| 2 | `tentori_belloni_2023` | Nucl. Fusion 63, 086001 | Fit to Sikora–Weller + Buck |
| 3 | `sikora_weller_lower` | J. Fusion Energy 35, 538 | Tentori S × 0.9 lower envelope; **not** a design baseline |

Energies above a model's fitted `E_max` return σ = 0 and emit a one-shot
warning (no silent extrapolation). Reactivity tables are cached **per model**.

## Reactivity | `p11b_reactivity()`

Unlike D–T / D–³He / D–D (closed-form Bosch–Hale ⟨σv⟩ fits), p–¹¹B uses the
shared Maxwellian **velocity-distribution × cross-section** integrator in
`process.models.physics.reactivity.maxwellian_reactivity` (Gauss–Laguerre over
σ(E) at a given ion temperature). Values are tabulated on 0.5–3000 keV and
served through `p11b_reactivity(t, model=...)` for optimiser speed. Both thermal
channels share the same profile weight `profile_reactivity_integrand`
(`2ρ ⟨σv⟩ (n/⟨n⟩)²`).

The default (Wang et al. 2026) Maxwellian reactivity uses the astrophysical
S-factor parameterisation:

$$
\sigma(E) = \frac{S(E)}{E}\exp\left(-\sqrt{\frac{E_G}{E}}\right),
\qquad E_G = 22.589\ \mathrm{MeV}
$$

with the piecewise fit

$$
S(E)=
\begin{cases}
S_1(E), & E \le 0.4\ \mathrm{MeV},\\
S_2(E), & 0.4 < E \le 0.7\ \mathrm{MeV},\\
S_3(E), & 0.7 < E \le 10\ \mathrm{MeV},
\end{cases}
$$

where S₁ is a polynomial plus the 148 keV Breit-Wigner resonance, S₂ is a
polynomial, and S₃ is a sum of five Breit-Wigner resonances (0.62, 1.39,
2.49, 3.53 and 4.70 MeV). The coefficients are given in Table 1 of the
reference.

The reactivity is then

$$
\langle\sigma v\rangle =
\sqrt{\frac{8kT}{\pi\mu}}
\int_0^\infty \sigma(xkT)\,x\,e^{-x}\,\mathrm{d}x,
$$

with the p-¹¹B reduced mass μc² = 859.526 MeV. It is tabulated on
0.5–3000 keV and interpolated in log–log space, which keeps the model fast
enough for the optimiser.

The fit is valid for centre-of-mass energies up to 10 MeV; results at ion
temperatures far above ~1 MeV should be treated with caution.

## Reaction | `p11b_reaction()`

The reaction power density is

$$
P_{p\text{-}^{11}\text{B}} =
\langle\sigma v\rangle\,
n_p\, n_{B11}\,
E_{p\text{-}^{11}\text{B}},
$$

integrated over the same profile shape as the other reactions.

### Relation to the D-T alpha treatment

The p-¹¹B channel uses exactly the same bookkeeping as D-T, with a different
energy fraction and a different number of alphas per reaction:

| Quantity | D-T | p-¹¹B |
|---|---|---|
| Energy fraction to alphas | $1-f_{\text{neutron}}=0.204$ | $1-f_{\text{neutron}}=1.0$ |
| Alphas per reaction | 1 | 3 |
| Energy per alpha | 3.5 MeV | 8.683/3 = 2.894 MeV |
| Neutron power | 79.6% of $P_{\text{fus}}$ | 0 |

In code this is

```python
# D-T (dt_reaction)
alpha_power_density = (1.0 - constants.DT_NEUTRON_ENERGY_FRACTION) * fusion_power_density
alpha_rate_density  = fusion_rate_density

# p-11B (p11b_reaction)
alpha_power_density = constants.P_B11_ALPHA_ENERGY_FRACTION * fusion_power_density   # 1.0
alpha_rate_density  = constants.P_B11_ALPHA_COUNT * fusion_rate_density              # 3.0
```

with the constants

| Constant | Value | Meaning |
|---|---|---|
| `P_B11_ALPHA_COUNT` | 3.0 | alphas per reaction |
| `P_B11_NEUTRON_ENERGY_FRACTION` | 0.0 | no neutron branch |
| `P_B11_ALPHA_ENERGY_FRACTION` | 1.0 | all energy to alphas |
| `P_B11_ALPHA_ENERGY` | 2.894 MeV | energy per alpha |

All of the reaction energy is therefore added to the **alpha particle** fusion
power (`pden_plasma_alpha_mw`), and three alpha particles are added to the alpha
production rate per reaction. The contribution appears in:

- `p_alpha_total_mw` / `pden_alpha_total_mw` (and hence the fast-alpha beta),
- `fusden_plasma_alpha` / `fusden_alpha_total` (alpha confinement time),
- `p_fusion_total_mw` and the plasma power balance.

No neutron power is produced. The `f_p_alpha_plasma_deposited` factor and the
ion/electron alpha energy split are applied to the combined alpha power, as for
D-T alphas.

Output variables:

| Variable | Description |
|---|---|
| `pden_plasma_p11b_mw` | p-¹¹B fusion power density [MW/m³] |
| `p_plasma_p11b_mw` | p-¹¹B fusion power [MW] |
| `fusden_plasma_p11b` | p-¹¹B reaction rate density [reactions/m³/s] |
| `fusden_plasma_p11b_alpha` | p-¹¹B alpha production rate density = 3× reaction rate [α/m³/s] |
| `nd_plasma_boron11_vol_avg` | volume-averaged boron-11 density [m⁻³] |

## Example input

```ini
! p-11B fuel (fractions of the electron density)
f_nd_protium_electrons  = 0.05
f_nd_boron11_electrons  = 0.10
f_nd_beam_electron      = 0.0

! keep a small D/T background for the remaining fuel charge
f_plasma_fuel_deuterium = 0.5
f_plasma_fuel_tritium   = 0.5
f_plasma_fuel_helium3   = 0.0
```

## Limitations

- **Boron line radiation is not included** in the impurity radiation model
  (boron is not one of the 14 impurity species). Bremsstrahlung still sees
  the increased Z<sub>eff</sub>, but line radiation from boron is missing, which
  matters for the p-¹¹B energy balance.
- **Helium ash density is still an input.** The p-¹¹B alpha *production rate* is
  included in `fusden_plasma_alpha`, but the thermal alpha density
  (`nd_plasma_alphas_thermal_vol_avg`) is still set by
  `f_nd_alpha_thermal_electron` and is not solved from the p-¹¹B source.
- The alpha energy split between ions and electrons uses the D-T alpha
  model (equation for `f_alpha_electron`), evaluated at the electron
  temperature; for the 2.89 MeV p-¹¹B alphas this is an approximation.
- **Tritium breeding, neutron wall load and shielding** models still assume a
  D-T device. For a pure p-¹¹B study these must be switched off or replaced.
- A dedicated p-¹¹B design point needs ion temperatures of order 100 keV or
  more; a DEMO-like input file will only produce a small p-¹¹B contribution.
- The proton density is shared with the D-³He proton production model; for
  p-¹¹B studies set `f_nd_protium_electrons` to the desired fuel proton fraction
  and keep D-³He reactions negligible.

## References

- Wang et al., "Revisiting p-¹¹B Fusion: Updated Cross-sections, Reactivity,
  and Energy Balance", arXiv:2601.00241 (2026).
- W. M. Nevins and R. Swain, "The thermonuclear fusion rate coefficient for
  p-¹¹B reactions", *Nuclear Fusion*, 40(4), 865–872 (2000).
