#ifndef PMAP_H_
#define PMAP_H_

#include <vector>

struct _p_AO;
typedef _p_AO* AO;


namespace Petscpp{

  // Experimental
  class Map
  {
  public:

    /*! Create mapping from 'from' to 'to vectors.
     * If to is not specified, it is mapped to natural ordering 0,1,2,3
     * Untested
     */
    Map(std::vector<int> const& from, std::vector<int> const& to = std::vector<int>());

    Map(Map const& other) = delete;
    Map operator=(Map const& other) = delete;
    ~Map();

    AO& petscAO();

    // Map 'data' to Petsc ordering
    void toPetsc(std::vector<int> &data) const;

    // Map 'data' to application-define ordering
    void toApplication(std::vector<int> &data) const;

  private:
    AO ao_;
  };

}


#endif /* PMAP_H_ */
