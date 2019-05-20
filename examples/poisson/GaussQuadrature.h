#ifndef GAUSSQUADRATURE_H_
#define GAUSSQUADRATURE_H_

#include "Traits.h"
#include <vector>

namespace GP{
  struct gp2x2_d{
    using gparray = std::array< std::array<double,2>,  4>;

    static constexpr gparray const gp = {{ {{-1.0/std::sqrt(3.0), -1.0/std::sqrt(3.0)}},
                                           {{-1.0/std::sqrt(3.0), +1.0/std::sqrt(3.0)}},
                                           {{+1.0/std::sqrt(3.0), -1.0/std::sqrt(3.0)}},
                                           {{+1.0/std::sqrt(3.0), +1.0/std::sqrt(3.0)}},
      }};

    std::array<double, 4> const w  = {{1.0, 1.0, 1.0, 1.0}};
  };

  gp2x2_d::gparray constexpr const gp2x2_d::gp;

  static const gp2x2_d gp2x2;
}


namespace gp_helper{
  template <typename T>
  struct is_container{
    static const bool value = false;
  };

  template <typename T, typename P>
  struct is_container< std::vector<T,P> > {
    static const bool value = true;
  };

  template <typename T, size_t N>
  struct is_container< std::array<T,N> > {
    static const bool value = true;
  };

  template <typename T>
  struct is_eigen{
    static const bool value = false;
  };

  template <typename T, int m, int n, int opts, int mr, int mc>
  struct is_eigen< Eigen::Matrix<T,m,n, opts, mr, mc> >{
    static const bool value = true;
  };
}

namespace openfem{

  // Numerical integration of function F using Gaussian quadrature
  template <typename F, typename G>
  auto gq_integrate(F && f, G const& g) ->
    typename std::enable_if< !gp_helper::is_container< decltype(f(g.gp[0])) >::value,
                             decltype(f(g.gp[0]))>::type
  {

    decltype(f(g.gp[0])) res;

    if ( gp_helper::is_eigen< decltype(f(g.gp[0])) >::value )
      res.setZero();

    for (size_t i=0; i<g.gp.size(); ++i)
      res += f(g.gp[i]) * g.w[i];

    return res;
  }


  // Numerical integration of function F using Gaussian quadrature
  // (Overload for containers)
  template <typename F, typename GaussianPoints>
  auto
  gq_integrate(F && f, GaussianPoints const& g) ->
    typename std::enable_if< gp_helper::is_container< decltype(f(g.gp[0])) >::value,
                             decltype(f(g.gp[0]))>::type
  {
    decltype(f(g.gp[0])) res;
    // assuming F container is a std::array and contains Eigen
    if ( gp_helper::is_eigen< typename decltype(f(g.gp[0]))::value_type >::value ){
      for (auto &x : res)
        x.setZero();
    }

    for (size_t i=0; i<g.gp.size(); ++i){
      auto eval = f(g.gp[i]);
      for (size_t j=0; j<eval.size(); ++j)
        res[j] +=  eval[j]*g.w[i];
    }
    return res;
  }

}


#endif /* GAUSSQUADRATURE_H_ */
