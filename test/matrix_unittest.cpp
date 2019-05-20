#include <limits>
#include <gtest/gtest.h>
#include <Petscpp/Matrix.h>
#include <Petscpp/Petscpp.h>
#include <Petscpp/Vector.h>
#include <Eigen/Dense>
#include <iostream>


TEST(Matrix, Ctor){
  using namespace Petscpp;
  // Default ctor
  Matrix m2;

  // Initiale size
  Matrix m1(10,10);
}


TEST(Matrix, GlobalSize){
  using namespace Petscpp;
  int const rowCount = 1300;
  int const colCount = 1200;

  // Specify global sizes of the matrix
  Matrix mat(rowCount, colCount);
  std::pair<int, int> const matsize = mat.globalSize();
  ASSERT_EQ(rowCount, matsize.first);
  ASSERT_EQ(colCount, matsize.second);

  ASSERT_EQ(rowCount, mat.globalRowCount());
  ASSERT_EQ(colCount, mat.globalColumnCount());
}


TEST(Matrix, localSize){
  using namespace Petscpp;
  int const rowCount = 300;
  int const colCount = 200;

  // Specify global sizes of the matrix
  Matrix mat(rowCount, colCount, false);
  std::pair<int, int> const matsize = mat.localSize();
  ASSERT_EQ(rowCount, matsize.first);
  ASSERT_EQ(colCount, matsize.second);

  ASSERT_EQ(rowCount, mat.localRowCount());
  ASSERT_EQ(colCount, mat.localColumnCount());
}


TEST(Matrix, SliceOperator_eigen_col_major){
  using namespace Petscpp;
  Eigen::Matrix3d emat;
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j){
      emat(i,j) = (j+1)+i*3;
    }

  double const abs_error = 1E-10;
  int const procCount  = Petscpp::procCount();
  int const mat_size   = procCount*100;

  Matrix pmat(mat_size, mat_size);
  int const starting_row = pmat.startingRow();

  pmat({1+starting_row, 2+starting_row, 3+starting_row},
       {1+starting_row, 2+starting_row, 3+starting_row}) = emat;
  pmat.assemble();
 
  for (int i=0; i<3; ++i){
    for (int j=0; j<3; ++j){
      ASSERT_NEAR(pmat(i+1+starting_row, j+1+starting_row).value(), emat(i,j), abs_error);
    }
  }
}


/*
 * TODO:
 *
 *  Matrix takes MatrixXd. When passing a Eigen::Matrix<double, m, n>,
 *  IsRowMajor defaults back to false even if it is set as
 *  RowMajor. For example:
 *
 *  Eigen<Matrix<double, 3,3, Eigen::ColMajor> mat_rowMajor;
 *
 *  (mat_rowMajor.IsRowMajor == true)  // true
 *
 *  Eigen::MatrixXd newMat = mat_rowMajor;
 *  (newMat.IsRowMajor == ture)        // false
 *
 *  In otherwords, this test is is no different does not do what is
 *  intended.
*/
TEST(Matrix, SliceOperator_eigen_row_major){
  using namespace Petscpp;
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> emat;
  Eigen::MatrixXd newMat = emat;

  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j){
      emat(i,j) = (j+1)+i*3;
    }

  double const abs_error = 1E-10;
  int const procCount  = Petscpp::procCount();
  int const mat_size   = procCount*100;

  Matrix pmat(mat_size, mat_size);
  int const starting_row = pmat.startingRow();

  pmat({1+starting_row, 2+starting_row, 3+starting_row},
       {1+starting_row, 2+starting_row, 3+starting_row}) = emat;
  pmat.assemble();


  for (int i=0; i<3; ++i){
    for (int j=0; j<3; ++j){
      ASSERT_NEAR(pmat(i+1+starting_row, j+1+starting_row).value(), emat(i,j), abs_error);
    }
  }
}


TEST(Matrix, toMatrix){
  // TODO:
}


TEST(Matrix, MatScale_multiplication){
  using namespace Petscpp;

  int const N = 16*Petscpp::procCount();
  double const abs_error = 1E-10;

  // Dense matrice
  Matrix m1(N, N, 0);
  std::pair<int,int> id = m1.startingIndex();
  int localRowCount = m1.localRowCount();
  int localColCount = m1.localColumnCount();
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      m1(i,j) = i+2*j;
    }
  }

  m1.assemble();
  double const scale = 2;

  // Test scaling from left and right
  Matrix m2 = m1*scale;
  Matrix m3 = scale*m1;
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(scale*(i+2*j), m2(i, j).value(), abs_error);
      ASSERT_NEAR(scale*(i+2*j), m3(i, j).value(), abs_error);
    }
  }

  // Test scaling already scale-op
  Matrix m4 = (m1*scale)*scale;
  Matrix m5 = scale*(scale*m1);
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(scale*scale*(i+2*j), m4(i, j).value(), abs_error);
      ASSERT_NEAR(scale*scale*(i+2*j), m5(i, j).value(), abs_error);
    }
  }
}


TEST(Matrix, MatScale_division){
  using namespace Petscpp;

  int const N = 16*Petscpp::procCount();
  double const abs_error = 1E-10;

  Matrix m1(N, N);
  std::pair<int,int> id = m1.startingIndex();
  int localRowCount = m1.localRowCount();
  int localColCount = m1.localColumnCount();

  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      m1(i,j) = i+2*j;
    }
  }

  m1.assemble();
  double const scale = 2;

  // Test scaling from left and right
  Matrix m2 = m1/scale;
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR((i+2*j)/scale, m2(i,j).value(), abs_error);
    }
  }

  // Mutliplication followed by division
  Matrix m3 = (m1*scale)/scale;
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR((i+2*j), m3(i,j).value(), abs_error);
    }
  }

  // Division followed by division
  Matrix m4 = (m1/scale)/scale;
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR((i+2*j)/(scale*scale), m4(i,j).value(), abs_error);
    }
  }
}


TEST(Matrix, MatAddOp){
  using namespace Petscpp;

  int const M = Petscpp::procCount()*16;
  int const N = Petscpp::procCount()*16;
  double const abs_err = 1E-10;

  // Dense matrices
  Matrix m1(M,N, 0);
  Matrix m2(M,N, 0);
  Eigen::MatrixXd m1_eigen = Eigen::MatrixXd::Zero(M,N);
  Eigen::MatrixXd m2_eigen = Eigen::MatrixXd::Zero(M,N);

  std::pair<int,int> id = m1.startingIndex();
  int localRowCount = m1.localRowCount();
  int localColCount = m1.localColumnCount();

  // Eigen matrix will be created fully on all processors
  for (int i=0; i<M; ++i){
    for (int j=0; j<N; ++j){
      m1_eigen(i,j) = i+2*j;
      m2_eigen(j,i) = i+2*j;
    }
  }

  // Create a non-symmetric matrix
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      m1(i,j) = i+2*j;
      m2(j,i) = i+2*j;
    }
  }

  // Simple addition without scaling
  Matrix m3 = m1 + m2;
  // Expected result
  Eigen::MatrixXd m3_eigen = m1_eigen + m2_eigen;

  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(m3_eigen(i,j), m3(i,j).value(), abs_err);
    }
  }
}


TEST(Matrix, ScaleMatAddOp){
  using namespace Petscpp;

  int const M = Petscpp::procCount()*16;
  int const N = Petscpp::procCount()*16;
  double const abs_err = 1E-10;

  // Dense matrices
  Matrix m1(M,N, 0);
  Matrix m2(M,N, 0);
  Eigen::MatrixXd m1_eigen(M,N);
  Eigen::MatrixXd m2_eigen(M,N);
  std::pair<int,int> id = m1.startingIndex();
  int localRowCount = m1.localRowCount();
  int localColCount = m1.localColumnCount();

  for (int i=0; i<M; ++i){
    for (int j=0; j<N; ++j){
      m1_eigen(i,j) = i+2*j;
      m2_eigen(j,i) = i+2*j;
    }
  }

  // Create a non-symmetric matrix
  for (int i=0; i<M; ++i){
    for (int j=0; j<N; ++j){
      m1(i,j) = i+2*j;
      m2(j,i) = i+2*j;
    }
  }

  // Simple addition without scaling
  double const scale = 4.3;
  Matrix m3 = (m1 + m2)*scale;
  Matrix m4 = scale*(m1 + m2);
  // Expected result
  Eigen::MatrixXd m3_eigen = (m1_eigen + m2_eigen)*scale;

  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(m3_eigen(i,j), m3(i,j).value(), abs_err);
      ASSERT_NEAR(m3_eigen(i,j), m4(i,j).value(), abs_err);
    }
  }

  // Simple addition without scaling
  Matrix m5 = m1 + m2*scale;
  Matrix m6 = scale*m1 + m2;
  Matrix m7 = (scale*m1 + m2)*scale;
  Matrix m8 = (scale*m1 + m2)/scale;
  // Expected result
  Eigen::MatrixXd m5_eigen = (m1_eigen + m2_eigen*scale);
  Eigen::MatrixXd m6_eigen = (scale*m1_eigen + m2_eigen);
  Eigen::MatrixXd m7_eigen = (scale*m1_eigen + m2_eigen)*scale;
  Eigen::MatrixXd m8_eigen = (scale*m1_eigen + m2_eigen)/scale;

  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(m5_eigen(i,j), m5(i,j).value(), abs_err);
      ASSERT_NEAR(m6_eigen(i,j), m6(i,j).value(), abs_err);
      ASSERT_NEAR(m7_eigen(i,j), m7(i,j).value(), abs_err);
      ASSERT_NEAR(m8_eigen(i,j), m8(i,j).value(), abs_err);
    }
  }
}


TEST(Matrix, MatMultOp){
  using namespace Petscpp;

  int const M = Petscpp::procCount()*16;
  int const N = Petscpp::procCount()*16;
  double const abs_err = 1E-10;

  // Create a dense matrix
  Matrix m1(M,N, 0);
  Matrix m2(M,N, 0);

  Eigen::MatrixXd m1_eigen = Eigen::MatrixXd::Zero(M,N);
  Eigen::MatrixXd m2_eigen = Eigen::MatrixXd::Zero(M,N);

  std::pair<int,int> id = m1.startingIndex();
  int localRowCount = m1.localRowCount();
  int localColCount = m1.localColumnCount();

  // Eigen matrix will be created fully on all processors
  for (int i=0; i<M; ++i){
    for (int j=0; j<N; ++j){
      m1_eigen(i,j) = (i+2*j)/(double)Petscpp::procCount();
      m2_eigen(j,i) = (i+2*j)/(double)Petscpp::procCount();
    }
  }

  // Create a non-symmetric matrix
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      m1(i,j) = (i+2*j)/(double)Petscpp::procCount();
      m2(j,i) = (i+2*j)/(double)Petscpp::procCount();
    }
  }

  // Simple addition without scaling
  Matrix m3 = m1 + m2;
  Eigen::MatrixXd m3_eigen = m1_eigen + m2_eigen;

  // Matrix multiplication with simple scaling
  Matrix m4 = 2*m1*m2*m3;
  Eigen::MatrixXd m4_eigen = m1_eigen*2*m2_eigen*m3_eigen;
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(m4_eigen(i,j), m4(i,j).value(), abs_err);
    }
  }
}


TEST(Matrix, ScaleMatMatMultOp){
  using namespace Petscpp;

  int const M = Petscpp::procCount()*16;
  int const N = Petscpp::procCount()*16;
  double const abs_err = 1E-10;

  // Create a dense matrix
  Matrix m1(M,N, 0);
  Matrix m2(M,N, 0);

  Eigen::MatrixXd m1_eigen = Eigen::MatrixXd::Zero(M,N);
  Eigen::MatrixXd m2_eigen = Eigen::MatrixXd::Zero(M,N);

  std::pair<int,int> id = m1.startingIndex();
  int localRowCount = m1.localRowCount();
  int localColCount = m1.localColumnCount();

  // Create a full eigen matrix on all processors
  for (int i=0; i<M; ++i){
    for (int j=0; j<N; ++j){
      m1_eigen(i,j) = (i+2*j)/(double)Petscpp::procCount();
      m2_eigen(j,i) = (i+2*j)/(double)Petscpp::procCount();
    }
  }

  // Create a non-symmetric distributed matrix
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      m1(i,j) = (i+2*j)/(double)Petscpp::procCount();
      m2(j,i) = (i+2*j)/(double)Petscpp::procCount();
    }
  }

  // Simple addition without scaling
  Matrix m3 = m1 + m2;
  Eigen::MatrixXd m3_eigen = m1_eigen + m2_eigen;

  Matrix m4 = 0.4*m1*(m2*m3)/2.0;
  Eigen::MatrixXd m4_eigen = 0.4*m1_eigen*(m2_eigen*m3_eigen)/2.0;
  for (int i=id.first; i<localRowCount; ++i){
    for (int j=id.second; j<localColCount; ++j){
      ASSERT_NEAR(m4_eigen(i,j), m4(i,j).value(), abs_err);
    }
  }
}


// TODO:
// TEST(Matrix, MatVecMult){
// }
