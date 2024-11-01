// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace gr::Solutions {

/*!
 * \brief The Kerr-Schild radius corresponding to a Boyer-Lindquist radius.
 *
 * \details Computes the radius of a surface of constant Boyer-Lindquist radius
 * as a function of angles.  The input argument `theta_phi` is typically the
 * output of the `theta_phi_points()` method of a `ylm::Spherepack` object;
 * i.e., a std::array of two DataVectors containing the values of theta and
 * phi at each point on a Strahlkorper.
 *
 *
 * Derivation:
 *
 * Define spherical coordinates \f$(r,\theta,\phi)\f$ in the usual way
 * from the Cartesian Kerr-Schild coordinates \f$(x,y,z)\f$
 * (i.e. \f$x = r \sin\theta \cos\phi\f$ and so on).
 * Then the relationship between \f$r\f$ and the radial
 * Boyer-Lindquist coordinate \f$r_{BL}\f$ is
 * \f[
 * r_{BL}^2 = \frac{1}{2}(r^2 - a^2)
 *     + \left(\frac{1}{4}(r^2-a^2)^2 +
 *             r^2(\vec{a}\cdot \hat{x})^2\right)^{1/2},
 * \f]
 * where \f$\vec{a}\f$ is the Kerr spin vector (with units of mass),
 * \f$\hat{x}\f$ means \f$(x/r,y/r,z/r)\f$, and the dot product is
 * taken as in flat space.
 *
 * We solve the above equation for \f$r^2\f$ as a function of angles,
 * yielding
 * \f[
 *     r^2 = \frac{r_{BL}^2 (r_{BL}^2 + a^2)}
                  {r_{BL}^2+(\vec{a}\cdot \hat{x})^2},
 * \f]
 * where the angles are encoded in \f$\hat x\f$ and everything else on the
 * right-hand side is constant.
 *
 * The Kerr-Schild radius can be Lorentz boosted in an arbitrary direction
 * \f$\vec{\beta}\f$. We apply a Lorentz transformation on the Cartesian
 * Kerr-Schild coordinates and find that the boosted Kerr-Schild radius \f$r'\f$
 * as a function of angles satisfies
 * \f[
 * r'^2 = r^2 (1 + \gamma^2 \left( \beta_x\cos\phi\sin\theta +
 * \beta_y\sin\phi\sin\theta + \beta_z\cos\theta \right)^2)
 * \f]
 * where \f$\gamma = \frac{1}{\sqrt{1 - \beta^2}}\f$ is the Lorentz factor, and
 * the components of \f$\vec{\beta}\f$ are given in Cartesian Kerr-Schild
 * coordinates.
 */
template <typename DataType>
Scalar<DataType> kerr_schild_radius_from_boyer_lindquist(
    const double boyer_lindquist_radius,
    const std::array<DataType, 2>& theta_phi, double mass,
    const std::array<double, 3>& dimensionless_spin,
    const std::array<double, 3>& boost_velocity);
/*!
 * \brief The Kerr-Schild radius corresponding to a Kerr horizon.
 *
 * \details `kerr_horizon_radius` evaluates \f$r\f$ using the above equation in
 * the documentation for `kerr_schild_radius_from_boyer_lindquist`, and
 * using the standard expression for the Boyer-Lindquist radius of the
 * Kerr horizon:
 * \f[
 *   r_{BL} = r_+ = M + \sqrt{M^2-a^2}.
 * \f]
 *
 * \note If the spin is nearly extremal, this function has accuracy
 *       limited to roughly \f$10^{-8}\f$, because of roundoff amplification
 *       from computing \f$M + \sqrt{M^2-a^2}\f$.
 */
template <typename DataType>
Scalar<DataType> kerr_horizon_radius(
    const std::array<DataType, 2>& theta_phi, double mass,
    const std::array<double, 3>& dimensionless_spin);

}  // namespace gr::Solutions
