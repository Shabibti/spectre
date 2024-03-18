// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Gforce.hpp"

#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Gforce::Gforce(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<BoundaryCorrection> Gforce::get_clone() const {
  return std::make_unique<Gforce>(*this);
}

void Gforce::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

double Gforce::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_covector,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3,
                  Frame::Inertial>& /*spatial_velocity_one_form*/,

    const Scalar<DataVector>& /*rest_mass_density*/,
    const Scalar<DataVector>& /*electron_fraction*/,
    const Scalar<DataVector>& /*temperature*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*spatial_velocity*/,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>&
    /*equation_of_state*/) {
  {
    // Compute max abs char speed
    Scalar<DataVector>& shift_dot_normal = *packaged_tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_abs_char_speed) =
          max(abs(-get(lapse) - get(shift_dot_normal) -
                  get(*normal_dot_mesh_velocity)),
              abs(get(lapse) - get(shift_dot_normal) -
                  get(*normal_dot_mesh_velocity)));
    } else {
      get(*packaged_abs_char_speed) =
          max(abs(-get(lapse) - get(shift_dot_normal)),
              abs(get(lapse) - get(shift_dot_normal)));
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  normal_dot_flux(packaged_normal_dot_flux_tilde_d, normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);

  *packaged_normal_covector = normal_covector;

  return max(get(*packaged_abs_char_speed));
}

void Gforce::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector_ext,
    const dg::Formulation dg_formulation) {
  Scalar<DataVector> tilde_d_LW{};
  Scalar<DataVector> tilde_ye_LW{};
  Scalar<DataVector> tilde_tau_LW{};
  tnsr::i<DataVector, 3, Frame::Inertial> tilde_s_LW{};
  tnsr::I<DataVector, 3, Frame::Inertial> tilde_b_LW{};
  Scalar<DataVector> tilde_phi_LW{};

  const auto omega_g = 0.5;

  get(tilde_d_LW) =
      0.5 * (get(tilde_d_ext) + get(tilde_d_int)) -
      0.5 * (1.0 / max(get(abs_char_speed_int), get(abs_char_speed_ext))) *
          (-get(normal_dot_flux_tilde_d_int) -
           get(normal_dot_flux_tilde_d_ext));
  get(tilde_ye_LW) =
      0.5 * (get(tilde_ye_ext) + get(tilde_ye_int)) -
      0.5 * (1.0 / max(get(abs_char_speed_int), get(abs_char_speed_ext))) *
          (-get(normal_dot_flux_tilde_ye_int) -
           get(normal_dot_flux_tilde_ye_ext));
  get(tilde_tau_LW) =
      0.5 * (get(tilde_tau_ext) + get(tilde_tau_int)) -
      0.5 * (1.0 / max(get(abs_char_speed_int), get(abs_char_speed_ext))) *
          (-get(normal_dot_flux_tilde_tau_int) -
           get(normal_dot_flux_tilde_tau_ext));
  get(tilde_phi_LW) =
      0.5 * (get(tilde_phi_ext) + get(tilde_phi_int)) -
      0.5 * (1.0 / max(get(abs_char_speed_int), get(abs_char_speed_ext))) *
          (-get(normal_dot_flux_tilde_phi_int) -
           get(normal_dot_flux_tilde_phi_ext));
  for (size_t i = 0; i < 3; ++i) {
    tilde_s_LW.get(i) =
        0.5 * (tilde_s_ext.get(i) + tilde_s_int.get(i)) -
        0.5 * (1.0 / max(get(abs_char_speed_int), get(abs_char_speed_ext))) *
            (-normal_dot_flux_tilde_s_int.get(i) -
             normal_dot_flux_tilde_s_ext.get(i));
    tilde_b_LW.get(i) =
        0.5 * (tilde_b_ext.get(i) + tilde_b_int.get(i)) -
        0.5 * (1.0 / max(get(abs_char_speed_int), get(abs_char_speed_ext))) *
            (-normal_dot_flux_tilde_b_int.get(i) -
             normal_dot_flux_tilde_b_ext.get(i));
  }

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  Scalar<DataVector> sqrt_det_spatial_metric =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 1.0);
  Scalar<DataVector> lapse =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 1.0);
  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);

  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0;
    inv_spatial_metric.get(i, i) = 1.0;
  }

  Scalar<DataVector> rest_mass_density_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> electron_fraction_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> specific_internal_energy_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  Scalar<DataVector> divergence_cleaning_field_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> lorentz_factor_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> pressure_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> temperature_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);

  ::grmhd::ValenciaDivClean::PrimitiveFromConservative<
      tmpl::list<
          ::grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
      true>::apply(make_not_null(&rest_mass_density_LW),
                   make_not_null(&electron_fraction_LW),
                   make_not_null(&specific_internal_energy_LW),
                   make_not_null(&spatial_velocity_LW),
                   make_not_null(&magnetic_field_LW),
                   make_not_null(&divergence_cleaning_field_LW),
                   make_not_null(&lorentz_factor_LW),
                   make_not_null(&pressure_LW), make_not_null(&temperature_LW),
                   tilde_d_LW, tilde_tau_LW, tilde_tau_LW, tilde_s_LW,
                   tilde_b_LW, tilde_phi_LW, spatial_metric, inv_spatial_metric,
                   sqrt_det_spatial_metric,
                   EquationsOfState::IdealFluid<true>(2.0),
                   ::grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions(
                       1.0e-12, 1.0e-12, 10.0));

  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_d_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_ye_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_tau_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::Ij<DataVector, 3, Frame::Inertial> flux_tilde_s_LW =
      make_with_value<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::IJ<DataVector, 3, Frame::Inertial> flux_tilde_b_LW =
      make_with_value<tnsr::IJ<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> flux_tilde_phi_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);

  ::grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&flux_tilde_d_LW), make_not_null(&flux_tilde_ye_LW),
      make_not_null(&flux_tilde_tau_LW), make_not_null(&flux_tilde_s_LW),
      make_not_null(&flux_tilde_b_LW), make_not_null(&flux_tilde_phi_LW),
      tilde_d_LW, tilde_ye_LW, tilde_tau_LW, tilde_s_LW, tilde_b_LW,
      tilde_phi_LW, lapse, shift, sqrt_det_spatial_metric, spatial_metric,
      inv_spatial_metric, pressure_LW, spatial_velocity_LW, lorentz_factor_LW,
      magnetic_field_LW);

  tnsr::i<DataVector, 3, Frame::Inertial> average_normal_covector =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    average_normal_covector.get(i) =
        0.5 * normal_covector_int.get(i) - 0.5 * normal_covector_ext.get(i);
  }

  Scalar<DataVector> normal_dot_flux_tilde_d_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> normal_dot_flux_tilde_ye_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  Scalar<DataVector> normal_dot_flux_tilde_tau_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> normal_dot_flux_tilde_s_LW =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> normal_dot_flux_tilde_b_LW =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
          abs_char_speed_int, 0.0);
  Scalar<DataVector> normal_dot_flux_tilde_phi_LW =
      make_with_value<Scalar<DataVector>>(abs_char_speed_int, 0.0);

  normal_dot_flux(make_not_null(&normal_dot_flux_tilde_d_LW),
                  average_normal_covector, flux_tilde_d_LW);
  normal_dot_flux(make_not_null(&normal_dot_flux_tilde_ye_LW),
                  average_normal_covector, flux_tilde_ye_LW);
  normal_dot_flux(make_not_null(&normal_dot_flux_tilde_tau_LW),
                  average_normal_covector, flux_tilde_tau_LW);
  normal_dot_flux(make_not_null(&normal_dot_flux_tilde_s_LW),
                  average_normal_covector, flux_tilde_s_LW);
  normal_dot_flux(make_not_null(&normal_dot_flux_tilde_b_LW),
                  average_normal_covector, flux_tilde_b_LW);
  normal_dot_flux(make_not_null(&normal_dot_flux_tilde_phi_LW),
                  average_normal_covector, flux_tilde_phi_LW);

  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_tilde_d) =
        (1.0 - omega_g) * (0.5 * (get(normal_dot_flux_tilde_d_int) -
                      get(normal_dot_flux_tilde_d_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_d_ext) - get(tilde_d_int))) +
        omega_g * get(normal_dot_flux_tilde_d_LW);
    get(*boundary_correction_tilde_ye) =
        (1.0 - omega_g) * (0.5 * (get(normal_dot_flux_tilde_ye_int) -
                      get(normal_dot_flux_tilde_ye_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_ye_ext) - get(tilde_ye_int))) +
        omega_g * get(normal_dot_flux_tilde_ye_LW);
    get(*boundary_correction_tilde_tau) =
        (1.0 - omega_g) * (0.5 * (get(normal_dot_flux_tilde_tau_int) -
                      get(normal_dot_flux_tilde_tau_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_tau_ext) - get(tilde_tau_int))) +
        omega_g * get(normal_dot_flux_tilde_tau_LW);
    get(*boundary_correction_tilde_phi) =
        (1.0 - omega_g) * (0.5 * (get(normal_dot_flux_tilde_phi_int) -
                      get(normal_dot_flux_tilde_phi_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_phi_ext) - get(tilde_phi_int))) +
        omega_g * get(normal_dot_flux_tilde_phi_LW);

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          (1.0 - omega_g) * (0.5 * (normal_dot_flux_tilde_s_int.get(i) -
                        normal_dot_flux_tilde_s_ext.get(i)) -
                 0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                     (tilde_s_ext.get(i) - tilde_s_int.get(i))) +
          omega_g * normal_dot_flux_tilde_s_LW.get(i);
      boundary_correction_tilde_b->get(i) =
          (1.0 - omega_g) * (0.5 * (normal_dot_flux_tilde_b_int.get(i) -
                        normal_dot_flux_tilde_b_ext.get(i)) -
                 0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                     (tilde_b_ext.get(i) - tilde_b_int.get(i))) +
          omega_g * normal_dot_flux_tilde_b_LW.get(i);
    }
  } else {
    get(*boundary_correction_tilde_d) =
        (1.0 - omega_g) * (-0.5 * (get(normal_dot_flux_tilde_d_int) +
                       get(normal_dot_flux_tilde_d_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_d_ext) - get(tilde_d_int))) +
        omega_g * get(normal_dot_flux_tilde_d_LW);
    get(*boundary_correction_tilde_ye) =
        (1.0 - omega_g) * (-0.5 * (get(normal_dot_flux_tilde_ye_int) +
                       get(normal_dot_flux_tilde_ye_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_ye_ext) - get(tilde_ye_int))) +
        omega_g * get(normal_dot_flux_tilde_ye_LW);
    get(*boundary_correction_tilde_tau) =
        (1.0 - omega_g) * (-0.5 * (get(normal_dot_flux_tilde_tau_int) +
                       get(normal_dot_flux_tilde_tau_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_tau_ext) - get(tilde_tau_int))) +
        omega_g * get(normal_dot_flux_tilde_tau_LW);
    get(*boundary_correction_tilde_phi) =
        (1.0 - omega_g) * (-0.5 * (get(normal_dot_flux_tilde_phi_int) +
                       get(normal_dot_flux_tilde_phi_ext)) -
               0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                   (get(tilde_phi_ext) - get(tilde_phi_int))) +
        omega_g * get(normal_dot_flux_tilde_phi_LW);

    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          (1.0 - omega_g) * (-0.5 * (normal_dot_flux_tilde_s_int.get(i) +
                         normal_dot_flux_tilde_s_ext.get(i)) -
                 0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                     (tilde_s_ext.get(i) - tilde_s_int.get(i))) +
          omega_g * normal_dot_flux_tilde_s_LW.get(i);
      boundary_correction_tilde_b->get(i) =
          (1.0 - omega_g) * (-0.5 * (normal_dot_flux_tilde_b_int.get(i) +
                         normal_dot_flux_tilde_b_ext.get(i)) -
                 0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
                     (tilde_b_ext.get(i) - tilde_b_int.get(i))) +
          omega_g * normal_dot_flux_tilde_b_LW.get(i);
    }
  }
}

bool operator==(const Gforce& /*lhs*/, const Gforce& /*rhs*/) { return true; }

bool operator!=(const Gforce& lhs, const Gforce& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Gforce::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
