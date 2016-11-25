#include "MatrixPartitioning.h"
#include "Petscpp.h"
#include <iostream>
#include <petscmat.h>

using namespace Petscpp;

MatrixPartitioning::
MatrixPartitioning(int globalCellCount)
  : adj_(nullptr), globalCellCount_(globalCellCount)
{
  MatPartitioningCreate(PETSC_COMM_WORLD, &partitioning_);
  createAdjMat_ = true;
}


MatrixPartitioning::
MatrixPartitioning(Mat && adjacency)
  : adj_(adjacency)
{
  MatPartitioningCreate(PETSC_COMM_WORLD, &partitioning_);
  MatPartitioningSetAdjacency(partitioning_, adjacency);
  MatPartitioningSetFromOptions(partitioning_);
  globalCellCount_ = 0;
  createAdjMat_ = false;
}


MatrixPartitioning::
~MatrixPartitioning(){
  if (partitioning_)
    MatPartitioningDestroy(&partitioning_);
  if (adj_)
    MatDestroy(&adj_);
}


MatrixPartitioning::
MatrixPartitioning(MatrixPartitioning&& other){
  std::swap(partitioning_, other.partitioning_);
  std::swap(globalCellCount_, other.globalCellCount_);
  std::swap(createAdjMat_, other.createAdjMat_);
  other.partitioning_ = nullptr;
}


MatrixPartitioning&
MatrixPartitioning::
operator=(MatrixPartitioning&& other){
  std::swap(partitioning_, other.partitioning_);
  other.partitioning_ = nullptr;
  std::swap(globalCellCount_, other.globalCellCount_);
  std::swap(createAdjMat_, other.createAdjMat_);
  return *this;
}


IndexSet
MatrixPartitioning::
partitioning() const{

  std::lock_guard<std::mutex> guard(mutex_);

  if (createAdjMat_){
    adj_ = createAdjacencyMatrix();
    MatPartitioningSetAdjacency(partitioning_, adj_);
    MatPartitioningSetFromOptions(partitioning_);
    createAdjMat_ = false;
  }
  IS indexSet;
  MatPartitioningApply(partitioning_, &indexSet);
  return IndexSet(indexSet);
}


void
MatrixPartitioning::
setNeighbors(int row, std::vector<int> const& neighbors){
  adjacencyMap_[row] = std::set<int>(neighbors.begin(), neighbors.end());
  createAdjMat_ = true;
}


Mat
MatrixPartitioning::
createAdjacencyMatrix() const{

  /* Petscp takes the values in the following form:
   *   cols: vectorized list of neighbors for each local row
   *   indices: starting index of the vectorized column data
   *
   * Example:
   *  Suppose that proc0 has the following neighbors for each local
   *   cell:
   *      0: {1,3}
   *      1: {0,2,4}
   *      2: {1,5}
   *      3: {0,4,6}
   *      4: {1,3,5,7}
   *  Then the row and index data would be as follows:
   *    cols:    1,3 | 0,2,4, | 1,5, | 0,4,6, | 1,3,5,7 |
   *    indices: 0,    2,       5,     7,       10,      14
   */

  int colSize = 0;
  int indicesSize = 1;
  int localCellCount = adjacencyMap_.size();
  for (auto const& set_pair : adjacencyMap_){
    colSize += (int)set_pair.second.size();
    ++indicesSize;
  }

  // Put the data on the free-store since Petsc takes ownership of the
  // row/col data.
  int *sIndices, *cols;
  PetscMalloc(sizeof(int)*indicesSize, &sIndices);
  PetscMalloc(sizeof(int)*colSize, &cols);

  int jIndex = 0;
  int iIndex = 1;               // first index is always zero
  sIndices[0] = 0;
  for (auto const& set_pair : adjacencyMap_){
    for (auto x : set_pair.second){
      cols[jIndex] = x;
      ++jIndex;
    }
    sIndices[iIndex] = jIndex;
    ++iIndex;
  }

  Mat adj;
  MatCreateMPIAdj(PETSC_COMM_WORLD, localCellCount, globalCellCount_,
                  sIndices, cols, nullptr, &adj);
  return adj;
}
