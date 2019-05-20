#include "IndexSet.h"
using namespace Petscpp;

IndexSet::
IndexSet()
  : is_(nullptr)
{ }


IndexSet::
IndexSet(std::vector<int> const& indices, int blockSize /* = 1 */)
  : is_(nullptr)
{
  if (indices.size() != 0){
    if (blockSize != 1)
      ISCreateBlock(PETSC_COMM_WORLD, blockSize, (int)indices.size(),
                    &indices[0], PETSC_COPY_VALUES, &is_);
    else
      ISCreateGeneral(PETSC_COMM_WORLD, (int)indices.size(), &indices[0],
                      PETSC_COPY_VALUES, &is_);
  }
}


IndexSet::
IndexSet(IndexSet const& other)
  : is_(nullptr)
{
  if (other.is_){
    // ISCopy(other.is_, is_);
    ISDuplicate(other.is_, &is_);
  }
}


IndexSet::
IndexSet(IndexSet const& other, int blockSize)
  : is_(nullptr)
{
  if (other.is_){
    ISCreateBlock(PETSC_COMM_WORLD, blockSize, other.localSize(), other.indexArray().data(),
                  PETSC_COPY_VALUES, &is_);
  }
}



IndexSet::
IndexSet(IS is)
  : is_(is)
{ }


IndexSet::
~IndexSet(){
  if (is_)
    ISDestroy(&is_);
}


IndexSet::
IndexSet(IndexSet&& other){
  std::swap(is_, other.is_);
  other.is_ = nullptr;
}


IndexSet&
IndexSet::
operator=(IndexSet&& other){
  std::swap(other.is_, is_);
  other.is_ = nullptr;
  return *this;
}


IndexSetArray
IndexSet::
indexArray() const{
  return IndexSetArray(*this);
}


IS const&
IndexSet::
petscIS() const{
  return is_;
}


int
IndexSet::
globalSize() const{
  int size = 0;
  ISGetSize(is_, &size);
  return size;
}


int
IndexSet::
localSize() const{
  int size = 0;
  ISGetLocalSize(is_, &size);
  return size;
}


IS&
IndexSet::
petscIS(){
  return is_;
}


int
IndexSet::
blockSize() const{
  int size = 0;
  ISGetBlockSize(is_, &size);
  return size;
}


std::vector<int>
IndexSet::
partitionCount() const{
  int const procCount = Petscpp::procCount();
  std::vector<int> ret(procCount);
  ISPartitioningCount(is_, procCount, &ret[0]);
  return ret;
}


// -----------------------------------------------------------------
// Free functions

IndexSet
Petscpp::
partitionToNumbering(IndexSet const& is){
  IS numbering;
  ISPartitioningToNumbering(is.is_, &numbering);
  return IndexSet(numbering);
}
