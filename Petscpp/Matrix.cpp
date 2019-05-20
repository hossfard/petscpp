#include "Matrix.h"
#include "Petscpp.h"
#include "petscmat.h"
#include <iostream>

using namespace Petscpp;

constexpr bool DecorateCtors = false;

Matrix::
Matrix(){
  if (DecorateCtors) std::cout << "Matrix()" << std::endl;

  int m = 0;
  int n = 0;
  int nz = 1;

  MatCreate(PETSC_COMM_WORLD, &mat_);
  MatSetSizes(mat_, m, n, PETSC_DETERMINE, PETSC_DETERMINE);
  MatSetFromOptions(mat_);

  if (procCount() > 1)
    MatMPIAIJSetPreallocation(mat_, nz, PETSC_NULL, nz, PETSC_NULL);
  else
    MatSeqAIJSetPreallocation(mat_, nz, PETSC_NULL);

  MatSetOption(mat_, MAT_ROW_ORIENTED, PETSC_FALSE);
  MatZeroEntries(mat_);
}


Matrix::
Matrix(int m, int n, bool globalSizes /* = true */, int nzCount /*= -1*/){
  if (DecorateCtors) std::cout << "Matrix(m,n)" << std::endl;

  // If nz is specified as zero, default to 10% of the max(m,n) non-zero
  int nz = nzCount;
  if (nzCount < 0)
    nz = std::ceil(std::max(m,n)*0.1);

  MatCreate(PETSC_COMM_WORLD, &mat_);

  // Set sizes of the matrix
  if (globalSizes)
    MatSetSizes(mat_, PETSC_DECIDE, PETSC_DECIDE, m, n);
  else
    MatSetSizes(mat_, m, n, PETSC_DETERMINE, PETSC_DETERMINE);

  MatSetFromOptions(mat_);
  // MatSetSizes(mat_, PETSC_DECIDE, PETSC_DECIDE, m,n);
  // MatSetOption(mat_, MAT_ROW_ORIENTED, PETSC_FALSE);
  // MatSetOption(mat_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  // MatZeroEntries(mat_);

  if (procCount() > 1)
    MatMPIAIJSetPreallocation(mat_, nz, PETSC_NULL, nz,
                              PETSC_NULL); 
  else
    MatSeqAIJSetPreallocation(mat_, nz, PETSC_NULL);

  MatSetOption(mat_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  MatSetOption(mat_, MAT_ROW_ORIENTED, PETSC_FALSE);

  // allocate new space as needed
  MatSetOption(mat_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  assemble();
}


Matrix::
Matrix(Matrix const& other){
  if (DecorateCtors) std::cout << "Matrix(Matrix const&)" << std::endl;
  MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
}


Matrix& 
Matrix::
operator=(Matrix const& other) {
  if (DecorateCtors) std::cout << "operator=(Matrix const&)" << std::endl;
  // release existing matrix
  cleanup();
  MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
  return *this;
}


Matrix&
Matrix::
operator=(Matrix && other) {
  if (DecorateCtors) std::cout << "operator=(Matrix &&)" << std::endl;

  mat_ = other.mat_;
  other.mat_ = nullptr;
  // release existing matrix
  cleanup();
  // swap the internal mat ptr
  std::swap(mat_, other.mat_);
  return *this;
}


Matrix::
Matrix(MatrixScaleOp op)
  : mat_(nullptr)
{
  if (DecorateCtors) std::cout << "Matrix(MatrixScaleOp op) " << std::endl;

  *this = op.eval();
}


Matrix::
~Matrix(){
  if (DecorateCtors) std::cout << "~Matrix()" << std::endl;

  cleanup();
}


void
Matrix::
cleanup(){
  if (DecorateCtors) std::cout << "~Matrix()" << std::endl;

  if (mat_){
    MatDestroy(&mat_);
    mat_ = nullptr;
  }
}


void 
Matrix::
setSize(int m, int n){
  MatSetSizes(mat_, PETSC_DECIDE, PETSC_DECIDE, m,n);
}


void
Matrix::
assemble()
{
  MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY);
}


MatrixSlice
Matrix::
operator()(std::initializer_list<int> const& rowRange,
           std::initializer_list<int> const& colRange){
  return MatrixSlice(*this, rowRange, colRange);
}


Mat&
Matrix::
petscMat(){
  return mat_;
}


Mat const&
Matrix::
petscMat() const{
  return mat_;
}


MatrixElement
Matrix::
operator()(int i, int j){
  return MatrixElement(*this, i, j);
}


Matrix&
Matrix::
operator*=(double alpha){
  if (DecorateCtors) std::cout << "op *=(alpha = " << alpha << ") " << std::endl;

  MatScale(mat_, alpha);
  return *this;
}


std::pair<int,int>
Matrix::
globalSize() const{
  int rowCount = 0;
  int colCount = 0;
  MatGetSize(mat_, &rowCount, &colCount);
  return std::pair<int,int>(rowCount, colCount);
}


std::pair<int,int>
Matrix::
localSize() const{
  int rowCount = 0;
  int colCount = 0;
  MatGetLocalSize(mat_, &rowCount, &colCount);
  return std::pair<int,int>(rowCount, colCount);
}


int
Matrix::
localRowCount() const{
  std::pair<int,int> const size = localSize();
  return size.first;
}


int
Matrix::
localColumnCount() const{
  std::pair<int,int> const size = localSize();
  return size.second;
}


int
Matrix::
globalRowCount() const{
  std::pair<int,int> const size = globalSize();
  return size.first;
}


int
Matrix::
globalColumnCount() const{
  std::pair<int,int> const size = globalSize();
  return size.second;
}


int
Matrix::
startingRow() const{
  int start = 0;
  MatGetOwnershipRange(mat_, &start, nullptr);
  return start;
}


// -----------------------------------------------------------------


Matrix&
MatrixSlice::
operator=(Eigen::MatrixXd const& values){

  bool readjustOrientation = false;
  if (values.IsRowMajor){
    MatSetOption(matrix_.mat_, MAT_ROW_ORIENTED, PETSC_TRUE);
    readjustOrientation = true;
  }

  MatSetValues(matrix_.mat_,
               values.rows(), &rowIndices_[0],
               values.cols(), &colIndices_[0],
               &values.data()[0],
               INSERT_VALUES);

  if (readjustOrientation)
    MatSetOption(matrix_.mat_, MAT_ROW_ORIENTED, PETSC_FALSE);

  return matrix_;
}


Matrix&
MatrixSlice::
operator+=(Eigen::MatrixXd const& values){
  bool readjustOrientation = false;
  if (values.IsRowMajor){
    MatSetOption(matrix_.mat_, MAT_ROW_ORIENTED, PETSC_TRUE);
    readjustOrientation = true;
  }

  MatSetValues(matrix_.mat_,
               values.rows(), &rowIndices_[0],
               values.cols(), &colIndices_[0],
               &values.data()[0],
               ADD_VALUES);

  if (readjustOrientation)
    MatSetOption(matrix_.mat_, MAT_ROW_ORIENTED, PETSC_FALSE);

  return matrix_;
}


Matrix&
MatrixSlice::
operator=(double value){
  std::vector<double> values(rowIndices_.size()*colIndices_.size(), value);
  MatSetValues(matrix_.mat_,
               rowIndices_.size(), &rowIndices_[0],
               colIndices_.size(), &colIndices_[0],
               &values[0],
               INSERT_VALUES);
  return matrix_;
}


Mat
MatrixSlice::
petscMatrix() const{
  return matrix_.mat_;
}


Eigen::MatrixXd
MatrixSlice::
toMatrix() const{
  Eigen::MatrixXd matrix(rowIndices_.size(), colIndices_.size());

  MatGetValues(petscMatrix(),
               // rowIndices_.size(), rowIndices_.begin(),
               // colIndices_.size(), colIndices_.begin(),
               rowIndices_.size(), &rowIndices_[0],
               colIndices_.size(), &colIndices_[0],
               &matrix.data()[0]);

  return matrix;
}


std::ostream&
operator<<(std::ostream& stream, MatrixSlice const& slice){
  stream << slice.toMatrix() << "\n";
  return stream;
}


// -----------------------------------------------------------------

double
MatrixElement::
value() const{
  double val = 0;
  MatGetValues(mat_.petscMat(), 1, &m_, 1, &n_, &val);
  return val;
}


Matrix&
MatrixElement::
operator=(double val){
  MatSetValues(mat_.petscMat(), 1, &m_, 1, &n_, &val, INSERT_VALUES);
  return mat_;
}


// -----------------------------------------------------------------


// MatrixProxy&
// MatrixProxy::
// operator=(MatrixScaleOp op){
//   if (mat_.petscMat())
//     MatDestroy(&(mat_.petscMat()));

//   mat_ = op.eval();
//   return *this;
// }




MatrixScaleOp::
MatrixScaleOp(Matrix& vec, double alpha)
  : mat_(vec), alpha_(alpha)
{ }


MatrixScaleOp&
MatrixScaleOp::
operator*=(double alpha){
  alpha_ *= alpha;
  return *this;
}


MatrixScaleOp&
MatrixScaleOp::
operator/=(double alpha){
  alpha_ /= alpha;
  return *this;
}


Matrix
MatrixScaleOp::
eval() const{
  Matrix mat(mat_);
  mat *= alpha_;
  return mat;
}


// Mat&
// MatrixScaleOp::
// mat() const{
//   return mat_.petscMat();
// }

MatObj
MatrixScaleOp::
evalNoScale() const{
  return MatObj(mat_.petscMat(), false);
}


double
MatrixScaleOp::
scaleFactor() const{
  return alpha_;
}


void
MatrixScaleOp::
addTo(Matrix& mat) const{
  // TODO: what's the best way to specify the matrix structure?
  MatStructure str = DIFFERENT_NONZERO_PATTERN;
  // TODO: Potential excessive use of assemble()
  mat_.assemble();
  MatAXPY(mat.petscMat(), alpha_, mat_.petscMat(), str);
}


Mat
MatrixScaleOp::
duplicateMat(bool copyValues /*= true*/) const{
  Mat ret;
  MatDuplicateOption opt = MAT_COPY_VALUES;
  if (!copyValues){
    opt = MAT_DO_NOT_COPY_VALUES;
  }
  mat_.assemble();
  MatDuplicate(mat_.petscMat(), opt, &ret);
  return ret;
}


// Multiply with input vector, _WITHOUT_ scaling
VecObj
MatrixScaleOp::
vecMultiplyNoScale(VecObj const& v) const{
  Vec ret;
  VecDuplicate(v.get(), &ret);
  MatMult(mat_.petscMat(), v.get(), ret);
  return VecObj(ret,false);
}


// -----------------------------------------------------------------

MatrixScaleOp
Petscpp::
operator*(double alpha, Matrix &mat){
  return MatrixScaleOp(mat, alpha);
}


MatrixScaleOp
Petscpp::
operator/(Matrix &mat, double alpha){
  return MatrixScaleOp(mat, 1.0/alpha);
}


MatrixScaleOp
Petscpp::
operator*(Matrix &mat, double alpha){
  return MatrixScaleOp(mat, alpha);
}


MatrixScaleOp
Petscpp::
operator*(MatrixScaleOp s, double alpha){
  s *= alpha;
  return s;
}


MatrixScaleOp
Petscpp::
operator/(MatrixScaleOp s, double alpha){
  s /= alpha;
  return s;
}


MatrixScaleOp
Petscpp::
operator*(double alpha, MatrixScaleOp s){
  s *= alpha;
  return s;
}


MatrixScaleOp operator/(double alpha, MatrixScaleOp s){
  s /= alpha;
  return s;
}
