// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
/// Boundary corrections/numerical fluxes
namespace BoundaryCorrections {
/// \cond
class Gforce;
class Hll;
class LaxWendroff;
class Rusanov;
/// \endcond

/*!
 * \brief The base class used to make boundary corrections factory createable so
 * they can be specified in the input file.
 */
class BoundaryCorrection : public PUP::able {
 public:
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;
  ~BoundaryCorrection() override = default;

  /// \cond
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<Gforce, Hll, LaxWendroff, Rusanov>;

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace BoundaryCorrections
}  // namespace grmhd::ValenciaDivClean
