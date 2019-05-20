#ifndef ADJACENCY_H_
#define ADJACENCY_H_

#include <vector>
#include <set>

namespace openfem{

  /* Return list of neighbors for input faces
   *
   * Input data is a vector containing indices of the conenctivity of
   * the indices of faces. For following grid
   *
   *    0 ---- 1 ---- 2
   *    |      |    /
   *    | [0]  |[1]/
   *    |      |  /
   *    |      | /
   *    4------3
   *
   * Input data can be defined as
   *
   *  links = {{0,4,3,1},
   *           {3,2,1}};
   *
   * Two faces are assumed to be neighbors if they share at least one edge.
   *
   * The expected output for the example is
   *  return = {{1},
   *            {0}};
   *
   */
  std::vector< std::set<int> >
  neighborList2d(std::vector< std::vector<int> > const& links);

}


#endif /* ADJACENCY_H_ */
