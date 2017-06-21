#include <limits>
#include <gtest/gtest.h>
#include <Petscpp/Matrix.h>
#include <Petscpp/Petscpp.h>
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
