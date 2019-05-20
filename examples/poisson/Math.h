#ifndef MATH_H_
#define MATH_H_

#include <Eigen/Dense>

namespace openfem{

  template <typename T, int M>
  Eigen::Matrix<T,M,M> outer_product(Eigen::Matrix<T,M,1> const& v1,
                                     Eigen::Matrix<T,M,1> const& v2){
    return ( v1 * v2.transpose() ).eval();
  }


  inline Eigen::Matrix2d inverse(Eigen::Matrix2d const& A){

    Eigen::Matrix2d inv;
    double const det = A.determinant();

    inv(0,0) =  1.0/det * A(1,1);
    inv(1,1) =  1.0/det * A(0,0);
    inv(1,0) = -1.0/det * A(1,0);
    inv(0,1) = -1.0/det * A(0,1);

    return inv;
  }

}


#endif /* MATH_H_ */


