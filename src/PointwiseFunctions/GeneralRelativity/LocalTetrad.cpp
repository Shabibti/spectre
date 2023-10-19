// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/LocalTetrad.hpp"

#include <cmath>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename DataType>
std::pair<tnsr::Ab<DataType, 3, Frame::Inertial>,
          tnsr::Ab<DataType, 3, Frame::Inertial>>
local_tetrad(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inverse_spatial_metric) {
  auto local_tetrad_tensor =
      make_with_value<tnsr::Ab<DataType, 3, Frame::Inertial>>(lapse, 0.0);
  auto inverse_local_tetrad_tensor =
      make_with_value<tnsr::Ab<DataType, 3, Frame::Inertial>>(lapse, 0.0);
  local_tetrad(make_not_null(&local_tetrad_tensor),
               make_not_null(&inverse_local_tetrad_tensor), lapse, shift,
               spatial_metric, inverse_spatial_metric);
  return std::pair{local_tetrad_tensor, inverse_local_tetrad_tensor};
};

template <typename DataType>
void local_tetrad(
    gsl::not_null<tnsr::Ab<DataType, 3, Frame::Inertial>*> local_tetrad_tensor,
    gsl::not_null<tnsr::Ab<DataType, 3, Frame::Inertial>*>
        inverse_local_tetrad_tensor,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataType, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataType, 3, Frame::Inertial>& inverse_spatial_metric) {
  for (size_t i = 0; i < local_tetrad_tensor->size(); ++i) {
    (*local_tetrad_tensor)[i] = 0.0;
    (*inverse_local_tetrad_tensor)[i] = 0.0;
  }

  // Define helper variables
  const DataType inv_square_lapse = 1.0 / square(get(lapse));
  const DataType inv_sqrt_inv_gamma11 =
      1.0 / sqrt(get<0, 0>(inverse_spatial_metric));
  const DataType gamma12_div_gamma22 =
      get<1, 2>(spatial_metric) / get<2, 2>(spatial_metric);
  const DataType B = square(square(get(lapse)) * inv_sqrt_inv_gamma11);
  const DataType D =
      1.0 / sqrt(get<2, 2>(spatial_metric) *
                 (get<1, 1>(spatial_metric) * get<2, 2>(spatial_metric) -
                  square(get<1, 2>(spatial_metric))));
  const DataType E =
      inv_square_lapse * (get<0>(shift) * get<0, 1>(inverse_spatial_metric) -
                          get<1>(shift) * get<0, 0>(inverse_spatial_metric));
  const DataType F = inv_square_lapse * get<0, 1>(inverse_spatial_metric);
  const DataType G =
      inv_square_lapse * (get<0>(shift) * get<0, 2>(inverse_spatial_metric) -
                          get<2>(shift) * get<0, 0>(inverse_spatial_metric));
  const DataType H = inv_square_lapse * get<0, 2>(inverse_spatial_metric);

  // Compute transformation components
  get<0, 0>(*local_tetrad_tensor) = 1.0 / get(lapse);
  get<1, 0>(*local_tetrad_tensor) =
      -get<0>(shift) * get<0, 0>(*local_tetrad_tensor);
  get<1, 1>(*local_tetrad_tensor) = sqrt(get<0, 0>(inverse_spatial_metric));
  get<2, 0>(*local_tetrad_tensor) =
      -get<1>(shift) * get<0, 0>(*local_tetrad_tensor);
  get<2, 1>(*local_tetrad_tensor) =
      get<0, 1>(inverse_spatial_metric) * inv_sqrt_inv_gamma11;
  get<2, 2>(*local_tetrad_tensor) = D * get<2, 2>(spatial_metric);
  get<3, 0>(*local_tetrad_tensor) =
      -get<2>(shift) * get<0, 0>(*local_tetrad_tensor);
  get<3, 1>(*local_tetrad_tensor) =
      get<0, 2>(inverse_spatial_metric) * inv_sqrt_inv_gamma11;
  get<3, 2>(*local_tetrad_tensor) = -D * get<1, 2>(spatial_metric);
  get<3, 3>(*local_tetrad_tensor) = 1.0 / sqrt(get<2, 2>(spatial_metric));

  // Compute inverse components
  get<0, 0>(*inverse_local_tetrad_tensor) = get(lapse);
  get<1, 0>(*inverse_local_tetrad_tensor) =
      get<0>(shift) / sqrt(get<0, 0>(inverse_spatial_metric));
  get<1, 1>(*inverse_local_tetrad_tensor) = inv_sqrt_inv_gamma11;
  get<2, 2>(*inverse_local_tetrad_tensor) = 1 / (D * get<2, 2>(spatial_metric));
  get<2, 0>(*inverse_local_tetrad_tensor) =
      -B * E * inv_square_lapse * get<2, 2>(*inverse_local_tetrad_tensor);
  get<2, 1>(*inverse_local_tetrad_tensor) =
      -B * F * inv_square_lapse * get<2, 2>(*inverse_local_tetrad_tensor);
  get<3, 3>(*inverse_local_tetrad_tensor) = sqrt(get<2, 2>(spatial_metric));
  get<3, 0>(*inverse_local_tetrad_tensor) =
      -(B * get<3, 3>(*inverse_local_tetrad_tensor)) *
      (G + E * gamma12_div_gamma22) * inv_square_lapse;
  get<3, 1>(*inverse_local_tetrad_tensor) =
      -(B * get<3, 3>(*inverse_local_tetrad_tensor)) *
      (H + F * gamma12_div_gamma22) * inv_square_lapse;
  get<3, 2>(*inverse_local_tetrad_tensor) =
      get<1, 2>(spatial_metric) / (sqrt(get<2, 2>(spatial_metric)));
}
}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

// TODO: Fix this...
#define INSTANTIATE(_, data)                                                  \
  template std::pair<tnsr::Ab<DTYPE(data), 3, FRAME(data)>,                   \
                     tnsr::Ab<DTYPE(data), 3, FRAME(data)>>                   \
  gr::local_tetrad(                                                           \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& shift,                      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spacetime_metric,          \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spacetime_metric); \
  template void gr::local_tetrad(                                             \
      const gsl::not_null<tnsr::Ab<DTYPE(data), 3, FRAME(data)>*>             \
          local_tetrad_tensor,                                                \
      const gsl::not_null<tnsr::Ab<DTYPE(data), 3, FRAME(data)>*>             \
          inverse_local_tetrad_tensor,                                        \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& shift,                      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spacetime_metric,          \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spacetime_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (Frame::Inertial));

#undef DTYPE
#undef FRAME
#undef INSTANTIATE
