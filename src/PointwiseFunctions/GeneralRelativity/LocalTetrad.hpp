// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
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
