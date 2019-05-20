#ifndef DIFFOP_H_
#define DIFFOP_H_

#include <Eigen/Core>
#include "GaussQuadrature.h"
#include "Bases.h"
#include "Math.h"

namespace openfem{

  namespace diffop{

    template <typename Interpolator>
    struct Laplacian{
    private:
      static size_t const size = Interpolator::size;
      using Mat = Eigen::Matrix<double, size, size >;
      using Vec = Eigen::Matrix<double, size, 1 >;

    public:
      Mat operator()(Vec const& X, Vec const& Y) const{

        auto integrand = [&X, &Y](std::array<double,2> const& gp){
          Interpolator q1;
          Eigen::Matrix2d jac;
          std::array<Vec, 2> const dQdx = Bases::dQdx(q1, X,Y, gp, jac);
          double const dj = jac.determinant();

          return ((-1*openfem::outer_product(dQdx[0],dQdx[0]) -
                   openfem::outer_product(dQdx[1],dQdx[1]))*dj).eval();
        };

        return openfem::gq_integrate(integrand, GP::gp2x2);
      }
    };

  } // namespace diffop


} // namespace openfem


#endif /* DIFFOP_H_ */
