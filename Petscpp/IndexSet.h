#ifndef PINDEXSET_H_
#define PINDEXSET_H_

#include <petscis.h>
#include <vector>
#include "Petscpp.h"

namespace Petscpp{

  class IndexSetArray;

  /*! Wrapper for Petsc's IS */
  class IndexSet
  {
  public:
    IndexSet();
    IndexSet(std::vector<int> const&, int blockSize = 1);
    IndexSet(IndexSet&& other);
    IndexSet(IS is);
    IndexSet(IndexSet const&);
    IndexSet(IndexSet const& is, int blockSize);
    IndexSet& operator=(IndexSet const&) = delete;
    ~IndexSet();

    IS& petscIS();

    IS const& petscIS() const;

    // Return the global size of the index-set
    int globalSize() const;

    // Return the local size of the index-set
    int localSize() const;

    // Return the block-size of the index-set
    int blockSize() const;

    IndexSetArray indexArray() const;

    IndexSet& operator=(IndexSet&& other);

    std::vector<int> partitionCount()const;

    friend IndexSet partitionToNumbering(IndexSet const& is);

  private:
    IS is_;
  };


  IndexSet partitionToNumbering(IndexSet const& is);


  class IndexSetArray
  {
  public:
    IndexSetArray(IndexSet const& is) : indexSet_(is), indices_(nullptr) {
      ISGetIndices(indexSet_.petscIS(), &indices_);
      restore_ = true;
    }

    IndexSetArray(IndexSetArray && other)
      : indexSet_(other.indexSet_), indices_(other.indices_){
      other.restore_ = false;
    }

    IndexSetArray(IndexSetArray const&) = delete;
    IndexSetArray& operator=(IndexSetArray const&) = delete;

    ~IndexSetArray(){
      if (restore_)
        ISRestoreIndices(indexSet_.petscIS(), &indices_);
    }

    int operator[](int i) {
      return indices_[i];
    }

    int const* data() const{
      return indices_;
    }

    void restore(){
      ISRestoreIndices(indexSet_.petscIS(), &indices_);
      restore_ = false;
    }

  private:
    IndexSet const& indexSet_;
    int const* indices_;
    bool restore_;
  };


  inline void print(IndexSet const& is, bool all=true){
    if (is.petscIS() == nullptr)
      return;
    else{
      if (all)
        ISView(is.petscIS(), PETSC_VIEWER_STDOUT_WORLD);
      else
        ISView(is.petscIS(), PETSC_VIEWER_STDOUT_SELF);
    }
  }

} /* Petscpp */


#endif /* PINDEXSET_H_ */
