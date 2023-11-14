// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
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
      gr::local_tetrad(lapse, shift, spatial_metric, inverse_spatial_metric,
                       Direction<3>::upper_xi());
  CHECK_ITERABLE_APPROX(local_tetrad.first, python_local_tetrad);
  CHECK_ITERABLE_APPROX(local_tetrad.second, python_inverse_local_tetrad);

  return;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.LocalTetrad",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/GeneralRelativity/"};

  test_local_tetrad_scalar(std::numeric_limits<double>::signaling_NaN());
}
