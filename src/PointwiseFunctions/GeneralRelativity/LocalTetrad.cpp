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

template <typename DataType>
void local_tetrad(
    gsl::not_null<tnsr::Ab<DataType, 3>*> local_tetrad_tensor,
    gsl::not_null<tnsr::Ab<DataType, 3>*> inverse_local_tetrad_tensor,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inverse_spatial_metric) {
  // Define helper variables
  const auto Z = 1.0 / square(get(lapse));
  const auto B = square(get(lapse)) / sqrt(get<0, 0>(inverse_spatial_metric));
  const auto C = 1.0 / sqrt(get<2, 2>(spatial_metric));
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
  get<0, 0>(*local_tetrad_tensor) = 1.0 / get(lapse);
  get<1, 0>(*local_tetrad_tensor) = -get<0>(shift) / get(lapse);
  get<1, 1>(*local_tetrad_tensor) = sqrt(get<0, 0>(inverse_spatial_metric));
  get<2, 0>(*local_tetrad_tensor) = -get<1>(shift) / get(lapse);
  get<2, 1>(*local_tetrad_tensor) = get<0, 0>(inverse_spatial_metric) /
                                    sqrt(get<0, 1>(inverse_spatial_metric));
  get<2, 2>(*local_tetrad_tensor) = D * get<2, 2>(spatial_metric);
  get<3, 0>(*local_tetrad_tensor) = -get<2>(shift) / get(lapse);
  get<3, 1>(*local_tetrad_tensor) = get<0, 0>(inverse_spatial_metric) /
                                    sqrt(get<0, 2>(inverse_spatial_metric));
  get<3, 2>(*local_tetrad_tensor) = D * get<1, 2>(spatial_metric);
  get<3, 3>(*local_tetrad_tensor) = C;

  // Compute inverse components
  get<0, 0>(*inverse_local_tetrad_tensor) = get(lapse);
  get<1, 0>(*inverse_local_tetrad_tensor) =
      get<0>(shift) / sqrt(get<0, 0>(inverse_spatial_metric));
  get<1, 1>(*inverse_local_tetrad_tensor) =
      1.0 / sqrt(get<0, 0>(inverse_spatial_metric));
  get<2, 0>(*inverse_local_tetrad_tensor) =
      -square(B) * E / (D * get<2, 2>(spatial_metric) * square(get(lapse)));
  get<2, 1>(*inverse_local_tetrad_tensor) =
      -square(B) * F / (D * get<2, 2>(spatial_metric) * square(get(lapse)));
  get<2, 2>(*inverse_local_tetrad_tensor) = 1 / (D * get<2, 2>(spatial_metric));
  get<3, 0>(*inverse_local_tetrad_tensor) =
      -(square(B) / C) *
      (G + E * get<1, 2>(spatial_metric) / get<2, 2>(spatial_metric)) /
      square(get(lapse));
  get<3, 1>(*inverse_local_tetrad_tensor) =
      -(square(B) / C) *
      (H + F * get<1, 2>(spatial_metric) / get<2, 2>(spatial_metric)) /
      square(get(lapse));
  ;
  get<3, 2>(*inverse_local_tetrad_tensor) =
      get<1, 2>(spatial_metric) / (C * get<2, 2>(spatial_metric));
  get<3, 3>(*inverse_local_tetrad_tensor) = 1.0 / C;
}
}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

// TODO: Fix this...
#define INSTANTIATE(_, data)                                              \
  template void gr::local_tetrad(                                         \
      const gsl::not_null<tnsr::Ab<DTYPE(data), 3>*> local_tetrad_tensor, \
      const gsl::not_null<tnsr::Ab<DTYPE(data), 3>*>                      \
          inverse_local_tetrad_tensor,                                    \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& shift,                  \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spacetime_metric,      \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spacetime_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (Frame::Inertial));

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
