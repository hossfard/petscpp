#include "Vector.h"
#include "IndexSet.h"
#include <fstream>
#include <iostream>

using namespace Petscpp;

constexpr bool DecorateCtors = false;

void createVector(Vec &vec, size_t size, VectorType type, SizeType sizeType){
  switch (type){
  case VectorType::MPI:{
    VecCreate(PETSC_COMM_WORLD, &vec);
    if (sizeType == SizeType::Global)
      VecSetSizes(vec, PETSC_DECIDE, size);
    else
      VecSetSizes(vec, size, PETSC_DECIDE);
    VecSetType(vec, "mpi");
    break;
  }
  case VectorType::Sequential:{
    VecCreateSeq(PETSC_COMM_SELF, size, &vec);
    VecSetType(vec, "seq");
    break;
  }
  case VectorType::Standard:
    VecSetType(vec, "standard");
    break;
  default:
    break;
  }
}


Vector::
Vector()
  : vector_(nullptr)
{
  if (DecorateCtors) std::cout << "Vector()" << std::endl;

  VecCreate(PETSC_COMM_WORLD, &vector_);
  VecSetFromOptions(vector_);
  VecSetOption(vector_, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  VecSetSizes(vector_, PETSC_DECIDE, 0);
  iteratorCount_ = 0;
}


Vector::
Vector(size_t size, VectorType type /* = VectorType::MPI */,
       SizeType st /*= SizeType::Global*/){
  if (DecorateCtors) std::cout << "Vector(size)" << std::endl;

  createVector(vector_, size, type, st);
  VecSetOption(vector_, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  iteratorCount_ = 0;
}


Vector::
Vector(size_t size, std::vector<int> const& ghostNodes,
       SizeType st /*= SizeType::Global*/){
  createVector(vector_, size, VectorType::MPI, st);
  VecMPISetGhost(vector_, (int)ghostNodes.size(), &ghostNodes[0]);
  iteratorCount_ = 0;
}


Vector::
Vector(size_t size, double alpha){
  if (DecorateCtors)  std::cout << "Vector(size, double val)" << std::endl;

  VecCreate(PETSC_COMM_WORLD, &vector_);
  VecSetSizes(vector_, PETSC_DECIDE, size);
  VecSetFromOptions(vector_);
  VecSetOption(vector_, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

  VecSet(vector_, alpha);
  iteratorCount_ = 0;
}


Vector::
Vector(Vector const& other)
  : vector_(nullptr)
{
  if (DecorateCtors)  std::cout << "Vector(Vector const&)" << std::endl;
  *this = other;
  iteratorCount_ = 0;
}


Vector::
Vector(VectorScaleOp const& op)
  : vector_(nullptr)
{
  if (DecorateCtors)  std::cout << "Vector = (VectorScaleOp const&)" << std::endl;

  *this = op.eval();
  iteratorCount_ = 0;
}


Vector&
Vector::
operator=(Vector const& other){
  if (DecorateCtors)  std::cout << "Vector = (Vector const&)" << std::endl;

  if (vector_){
    VecDestroy(&vector_);
  }

  vector_ = other.duplicateVec();
  VecCopy(other.vector_, vector_);
  iteratorCount_ = 0;
  return *this;
}


Vector&
Vector::
operator=(Vector && other){
  if (DecorateCtors)  std::cout << "Vector = (Vector &&)" << std::endl;

  std::swap(vector_, other.vector_);
  other.vector_ = nullptr;
  iteratorCount_ = 0;
  return *this;
}



Vector::
~Vector(){
  if (DecorateCtors)  std::cout << "~Vector() " << std::endl;

  if (vector_)
    VecDestroy(&vector_);
}


VectorElement
Vector::
operator[](int index){
  return VectorElement(*this, index);
}


VectorElementConst
Vector::
operator[](int index) const{
  return VectorElementConst(*this, index);
}


VectorSlice
Vector::
operator()(std::initializer_list<int> const& indices){
  return VectorSlice(*this, indices);
}


VectorSliceConst
Vector::
operator()(std::initializer_list<int> const& indices) const{
  return VectorSliceConst(*this, indices);
}



void
Vector::
addTo(Vector &other) { 
  VecAXPY(other.vector_, 1, vector_);
}


Vec
Vector::
duplicateVec() const{
  Vec newVec;
  VecDuplicate(vector_, &newVec);
  return newVec;
}


void
Vector::
assemble(){
  VecAssemblyBegin(vector_);
  VecAssemblyEnd(vector_);
}


VectorProxy
Vector::
noAlias(){
  return VectorProxy(*this);
}


int
Vector::
globalSize() const{
  int size = 0;
  VecGetSize(vector_, &size);
  return size;
}


int
Vector::
localSize() const{
  int size = 0;
  VecGetLocalSize(vector_, &size);
  return size;
}


int
Vector::
startingIndex() const{
  int start = 0;
  VecGetOwnershipRange(vector_, &start, nullptr);
  return start;
}


Vec&
Vector::
petscVec(){
  return vector_;
}


Vec const&
Vector::
petscVec() const{
  return vector_;
}


bool
Vector::
isGhostPadded() const{
  Vec v;
  VecGhostGetLocalForm(vector_, &v);
  bool isGhostPadded = (v != nullptr);
  VecGhostRestoreLocalForm(vector_, &v);
  return isGhostPadded;
}


// -----------------------------------------------------------------
// VectorProxy

VectorProxy::
VectorProxy(Vector& vec) : vec_(vec) { }


Vector&
VectorProxy::
operator+=(Vector const& other){
  VecAXPY(vec_.vector_, 1, other.vector_);
  return vec_;
}


void
VectorProxy::
addTo(Vector& vec){
  vec_.addTo(vec);
}


// -----------------------------------------------------------------
// VectorSlice

VectorSlice::
VectorSlice(Vector& vec, std::vector<int> const& index)
  : vec_(vec), indices_(index)
{ }


Vector&
VectorSlice::
operator=(double val){
  std::vector<double> const vals(indices_.size(), val);
  VecSetValues(vec_.vector_, vals.size(), &indices_[0], &vals[0], INSERT_VALUES);
  return vec_;
}


Vector&
VectorSlice::
operator+=(Eigen::VectorXd const& vals){
  VecSetValues(vec_.vector_, vals.size(), &indices_[0], &vals[0], ADD_VALUES);
  return vec_;
}


OmMatrix_d<double>
VectorSlice::
values() const{
  OmMatrix_d<double> vals(indices_.size(), 1);
  VecGetValues(vec_.vector_, indices_.size(), &indices_[0], &vals.data()[0]);
  return vals;
}


// -----------------------------------------------------------------

VectorSliceConst::
VectorSliceConst(Vector const& vec, std::vector<int> const& index)
  : vec_(vec), indices_(index)
{ }


OmMatrix_d<double>
VectorSliceConst::
values() const{
  OmMatrix_d<double> vals(indices_.size(), 1);
  VecGetValues(vec_.vector_, indices_.size(), &indices_[0], &vals.data()[0]);
  return vals;
}


// -----------------------------------------------------------------
// VectorScaleOp

void
VectorScaleOp::
addTo(Vector& other){
  if (DecorateCtors) std::cout << "+ " << alpha_ << " * vector ";
  VecAXPY(other.vector_, alpha_, vec_.vector_);
}


Vec
VectorScaleOp::
duplicateVec() const{
  if (DecorateCtors) std::cout << "VectorScaleOp::duplicateVec()";
  Vec newVec;
  VecDuplicate(vec_.petscVec(), &newVec);
  return newVec;
}


VectorScaleOp&
VectorScaleOp::
operator*=(double alpha){
  alpha_ *= alpha; 
  return *this;
}


Vector
VectorScaleOp::
eval() const{
  Vector ret(vec_);
  VecScale(ret.petscVec(), alpha_);
  return ret;
}

// -----------------------------------------------------------------


VectorElement::
VectorElement(Vector& vec, int const& index)
  : vec_(vec), index_(index) { }


Vector&
VectorElement::
operator=(double val){
  VecSetValues(vec_.vector_, 1, &index_, &val, INSERT_VALUES);
  return vec_;
}


double
VectorElement::
value() const{
  double val = 0;
  VecGetValues(vec_.vector_, 1, &index_, &val);
  return val;
}


Vector&
VectorElement::
operator-=(double val){
  val *= -1;
  VecSetValues(vec_.vector_, 1, &index_, &val, ADD_VALUES);
  return vec_;
}



VectorElementConst::
VectorElementConst(Vector const& vec, int const& index)
  : vec_(vec), index_(index) { }


double
VectorElementConst::
value() const{
  double val = 0;
  VecGetValues(vec_.vector_, 1, &index_, &val);
  return val;
}


// -----------------------------------------------------------------
// Free functions

VectorAddOp<VectorScaleOp, VectorScaleOp> 
Petscpp::
operator+(Vector& lhs, Vector& rhs){
  return VectorAddOp<VectorScaleOp, VectorScaleOp>( VectorScaleOp(lhs,1), VectorScaleOp(rhs,1));
}


VectorAddOp<VectorScaleOp, VectorScaleOp>
Petscpp::
operator-(Vector& lhs, Vector& rhs){
  return VectorAddOp<VectorScaleOp, VectorScaleOp>(VectorScaleOp(lhs,1), VectorScaleOp(rhs,-1));
}


VectorScaleOp
Petscpp::
operator*(double alpha, Vector &vec){
  return VectorScaleOp(vec, alpha);
}


VectorScaleOp
Petscpp::
operator*(Vector &vec, double alpha){
  return VectorScaleOp(vec, alpha);
}


Vector
Petscpp::
duplicate(Vector const& vec1){
  Vector newVector;
  VecDuplicate(vec1.vector_, &newVector.vector_);
  return newVector;
}


VectorScaleOp
Petscpp::
operator-(Vector &vec){
  return VectorScaleOp(vec, -1);
}


bool
octaveDumpLocal(Vec v, std::string const& filename,
                std::string const& varName = "label"){

  std::ofstream stream(filename.c_str());
  if (!stream.is_open()){
    return false;
  }
  else{
    PetscReal *data;
    VecGetArray(v, &data);

    int localSize = 0;
    VecGetLocalSize(v, &localSize);

    if (stream.is_open()){
      stream << varName << " = [...\n";
      for (int i=0; i<localSize; ++i){
        stream << data[i] << "\n";
      }
      stream << "];\n";
    }

    VecRestoreArray(v, &data);
  }
  return true;
}


// Experimental
// bool octaveDumpLocal(Vector const& v, std::string const& filename,
//                 std::string const& varName = "label"){

//   std::ofstream stream(filename.c_str());
//   if (!stream.is_open()){
//     return false;
//   }
//   else{
//     if (stream.is_open()){
//       stream << varName << " = [...\n";
//       for (auto it = v.begin(); it != v.end(); ++it){
//         std::cout << "dump" << std::endl;
//         stream << *it << "\n";
//         ++it;
//       }
//       stream << "];\n";
//     }
//   }
//   return true;
// }


void
Petscpp::
octavePrintLocal(Vector const& v, std::string const& filename,
                 std::string const& varName /*= "data"*/){
  if (v.isGhostPadded()){
    Vec localVec;

    // Update ghost values
    VecGhostUpdateBegin(v.petscVec(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(v.petscVec(), INSERT_VALUES, SCATTER_FORWARD);

    // Write to file
    VecGhostGetLocalForm(v.petscVec(), &localVec);
    octaveDumpLocal(localVec, filename, varName);
    VecGhostRestoreLocalForm(v.petscVec(), &localVec);
  }
  else{
    octaveDumpLocal(v.petscVec(), filename, varName);
  }
}


// -----------------------------------------------------------------

Vector&
VectorScatterProxy::
operator=(VectorScatterProxy && other){
  VecScatter scatter;
  // create the scatter structure
  VecScatterCreate(other.vec_.petscVec(), other.indexSet_.petscIS(),
                   vec_.petscVec(), indexSet_.petscIS(), &scatter);

  // copy the values
  VecScatterBegin(scatter, other.vec_.petscVec(), vec_.petscVec(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter, other.vec_.petscVec(), vec_.petscVec(),
                INSERT_VALUES, SCATTER_FORWARD);

  // destroy the scatter structure
  VecScatterDestroy(&scatter);
  return vec_;
}


Vector&
VectorScatterProxy::
operator=(VectorScatterProxyConst && other){
  VecScatter scatter;
  // create the scatter structure
  VecScatterCreate(other.vec_.petscVec(), other.indexSet_.petscIS(),
                   vec_.petscVec(), indexSet_.petscIS(), &scatter);

  // copy the values
  VecScatterBegin(scatter, other.vec_.petscVec(), vec_.petscVec(),
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter, other.vec_.petscVec(), vec_.petscVec(),
                INSERT_VALUES, SCATTER_FORWARD);

  // destroy the scatter structure
  VecScatterDestroy(&scatter);
  return vec_;
}


Vector&
VectorScatterProxy::
petscVec(){
  return vec_;
}
