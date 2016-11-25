#include "Map.h"
#include "petscao.h"

using namespace Petscpp;

Map::
Map(std::vector<int> const& from,
     std::vector<int> const& to /*= std::vector<int>()*/){
  if (to.size() != 0)
    AOCreateBasic(PETSC_COMM_WORLD, (int)from.size(), &from[0], &to[0], &ao_);
  else
    AOCreateBasic(PETSC_COMM_WORLD, (int)from.size(), &from[0], nullptr, &ao_);
}


Map::
~Map(){
  AODestroy(&ao_);
}


AO&
Map::
petscAO(){
  return ao_;
}


void
Map::
toPetsc(std::vector<int> &data) const{
  AOApplicationToPetsc(ao_, data.size(), &data[0]);
}


void
Map::
toApplication(std::vector<int> &data) const{
  AOPetscToApplication(ao_, data.size(), &data[0]);
}
