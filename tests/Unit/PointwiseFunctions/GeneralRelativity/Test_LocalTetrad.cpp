// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/LocalTetrad.hpp"

namespace {
template <typename DataType>
void test_local_tetrad_scalar(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size);
  const auto shift = TestHelpers::gr::random_shift<3>(make_not_null(&generator),
                                                      used_for_size);
  const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
      make_not_null(&generator), used_for_size);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto python_local_tetrad =
      pypp::call<tnsr::Ab<DataType, 3, Frame::Inertial>>(
          "LocalTetrad", "local_tetrad", lapse, shift, spatial_metric,
          inverse_spatial_metric);
  const auto python_inverse_local_tetrad =
      pypp::call<tnsr::Ab<DataType, 3, Frame::Inertial>>(
          "LocalTetrad", "inverse_local_tetrad", lapse, shift, spatial_metric,
          inverse_spatial_metric);
  const auto local_tetrad =
      gr::local_tetrad(lapse, shift, spatial_metric, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(local_tetrad.first, python_local_tetrad);
  CHECK_ITERABLE_APPROX(local_tetrad.second, python_inverse_local_tetrad);

  return;

  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::Ab<DataType, 3, Frame::Inertial>*>,
          gsl::not_null<tnsr::Ab<DataType, 3, Frame::Inertial>*>,
          const Scalar<DataType>&, const tnsr::I<DataType, 3, Frame::Inertial>&,
          const tnsr::ii<DataType, 3, Frame::Inertial>&,
          const tnsr::II<DataType, 3, Frame::Inertial>&)>(
          &gr::local_tetrad<DataType>),
      "LocalTetrad", {"local_tetrad", "inverse_local_tetrad"}, {{{0.01, 1.0}}},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.LocalTetrad",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/GeneralRelativity/"};

  test_local_tetrad_scalar(std::numeric_limits<double>::signaling_NaN());
}
