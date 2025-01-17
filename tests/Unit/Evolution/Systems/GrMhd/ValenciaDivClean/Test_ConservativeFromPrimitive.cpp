// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.ConservativeFromPrimitive",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply,
      "TestFunctions",
      {"tilde_d", "tilde_ye", "tilde_tau", "tilde_s", "tilde_b", "tilde_phi"},
      {{{0.0, 1.0}}}, DataVector{5});
}
