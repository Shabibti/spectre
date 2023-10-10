// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute transformation to local tetrad from lapse, shift, and spatial
 * metric
 *
 * \details
 *
 */

template <typename DataType>
std::pair<tnsr::Ab<DataType, 3, Frame::Inertial>,
          tnsr::Ab<DataType, 3, Frame::Inertial>>
local_tetrad(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inverse_spatial_metric);

template <typename DataType>
void local_tetrad(
    gsl::not_null<tnsr::Ab<DataType, 3, Frame::Inertial>*> local_tetrad_tensor,
    gsl::not_null<tnsr::Ab<DataType, 3, Frame::Inertial>*>
        inverse_local_tetrad_tensor,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inverse_spatial_metric);
/// @}
}  // namespace gr
