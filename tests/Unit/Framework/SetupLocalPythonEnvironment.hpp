// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <patchlevel.h>
#include <string>

/// Contains all functions for pypp
namespace pypp {
/// Enable calling of python in the local scope, and add directory(ies) to the
/// front of the search path for modules. The directory which is appended to the
/// path is relative to the `tests/Unit` directory.
struct SetupLocalPythonEnvironment {
  explicit SetupLocalPythonEnvironment(
      const std::string& cur_dir_relative_to_unit_test_path);

  ~SetupLocalPythonEnvironment() = default;

  SetupLocalPythonEnvironment(const SetupLocalPythonEnvironment&) = delete;
  SetupLocalPythonEnvironment& operator=(const SetupLocalPythonEnvironment&) =
      delete;
  SetupLocalPythonEnvironment(const SetupLocalPythonEnvironment&&) = delete;
  SetupLocalPythonEnvironment& operator=(const SetupLocalPythonEnvironment&&) =
      delete;

  /// \cond
  // We have to clean up the Python environment only after all tests have
  // finished running, since there could be multiple tests run in a single
  // executable launch. This is done in TestMain(Charm).cpp.
  static void finalize_env();
  /// \endcond

 private:
// In order to use NumPy's API, import_array() must be called. However it is a
// macro which contains a return statement, returning NULL in python 3 and void
// in python 2. As such it needs to be factored into its own function which
// returns either nullptr or void depending on the version.
#if PY_MAJOR_VERSION == 3
  static std::nullptr_t init_numpy();
#else
  static void init_numpy();
#endif
  static bool initialized;
  static bool finalized;
};
}  // namespace pypp
