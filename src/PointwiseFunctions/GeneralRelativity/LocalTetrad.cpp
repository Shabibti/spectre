// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {

template <typename DataType, size_t SpatialDim>
void local_tetrad(
    gsl::not_null<tnsr::Ab<DataType, SpatialDim, Frame>*> local_tetrad_tensor,
    gsl::not_null<tnsr::Ab<DataType, SpatialDim, Frame>*>
        inverse_local_tetrad_tensor,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  // Define helper variables
  const auto Z = 1 / square(get(lapse));
  const auto B = square(get(lapse)) / sqrt(get<0, 0>(inverse_spatial_metric));
  const auto C = 1 / sqrt(get<2, 2>(spatial_metric));
  const auto D =
      C / sqrt(get<1, 1>(spatial_metric) * get<2, 2>(spatial_metric) -
               square(get<1, 2>(spatial_metric)));
  const auto E = Z * (get<0>(shift) * get<0, 1>(inverse_spatial_metric) -
                      get<1>(shift) * get<0, 0>(inverse_spatial_metric));
  const auto F = Z * get<0, 1>(inverse_spatial_metric);
  const auto G = Z * (get<0>(shift) * get<0, 2>(inverse_spatial_metric) -
                      get<2>(shift) * get<0, 0>(inverse_spatial_metric));
  const auto H = Z * get<0, 2>(inverse_spatial_metric);

  // Compute transformation components
  get<0, 0>(*local_tetrad_tensor) = 1 / get(lapse);
  get<1, 0>(*local_tetrad_tensor) = -get<0>(shift) / get(lapse);
  get<1, 1>(*local_tetrad_tensor) = sqrt(get<0, 0>(inverse_spatial_metric));
}
}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

// TODO: Fix this...
#define INSTANTIATE(_, data)                             \
  template void gr::local_tetrad(                        \
      const gsl::not_null<Scalar<DTYPE(data)>*> lapse,   \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& shift, \
      const tnsr::aa<DTYPE(data), 3, FRAME(data)>& spacetime_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
