#ifndef PETSCPP_H_
#define PETSCPP_H_

#include "petscksp.h"
#include <string>


namespace Petscpp{

  class App
  {
  public:
    App(int argc, char *argv[], std::string const& settings = ""){
      if (settings != "")
        PetscInitialize(&argc, &argv, settings.c_str(), PETSC_NULL);
      else
        PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    }

    ~App(){
      PetscFinalize();
    }

    App(App const&) = delete;
    App& operator=(App const&) = delete;
  };

  static int procCount(){
    int worldSize;
    MPI_Comm_size(PETSC_COMM_WORLD, &worldSize);
    return worldSize;
  }

  static int procId(){
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank;
  }

} /* namespace Petscpp */





#endif /* PETSCPP_H_ */


