#ifndef PMATRIXPARTITIONING_H_
#define PMATRIXPARTITIONING_H_

#include "IndexSet.h"
#include <map>
#include <set>
#include <mutex>

struct _p_Mat;
typedef struct _p_Mat* Mat;

namespace Petscpp{

  class MatrixPartitioning
  {
  public:
    MatrixPartitioning(int globCellCount);
    MatrixPartitioning(Mat && adjacency);
    MatrixPartitioning(MatrixPartitioning const&) = delete;
    MatrixPartitioning& operator=(MatrixPartitioning const&) = delete;
    MatrixPartitioning& operator=(MatrixPartitioning&& other);
    MatrixPartitioning(MatrixPartitioning&& other);
    ~MatrixPartitioning();

    IndexSet partitioning() const;

    /* Assign the neighbors for given local row
     *
     * - Duplicate entries in neighbor list will be ignored
     * - Neighbor list will be automatically sorted
     * - Do not set neighbor data if adjacency matrix has been set
     *
    */
    void setNeighbors(int row, std::vector<int> const& neighbors);

  private:
    Mat createAdjacencyMatrix() const;

    mutable bool createAdjMat_;
    mutable Mat adj_;
    mutable std::mutex mutex_;
    int globalCellCount_;
    std::map<int, std::set<int> > adjacencyMap_;
    MatPartitioning partitioning_;
  };

}


#endif /* PMATRIXPARTITIONING_H_ */
