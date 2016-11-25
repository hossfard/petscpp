#ifndef PBITARRAY_H_
#define PBITARRAY_H_

#include <petscbt.h>

namespace Petscpp{

  /*! Wrapper for Petsc's BT
   *
   * Distributed bit array. All processors have access to all entries
   */
  class BitArray
  {
  public:
    BitArray(int m)
      : bt_(nullptr)
    {
      PetscBTCreate(m, &bt_);
      PetscBTMemzero(m, bt_);
    }

    bool value(int i) const{
      return PetscBTLookup(bt_, i);
    }

    bool getAndSet(int i, bool tf = true){
      if (tf)
        return PetscBTLookupSet(bt_, i);
      else
        return PetscBTClear(bt_, i);
    }

    ~BitArray(){
      if (bt_)
        PetscBTDestroy(&bt_);
    }


    PetscBT& petscBT(){
      return bt_;
    }

  private:
    PetscBT bt_;
  };

} // namespace Petscpp


#endif /* PBITARRAY_H_ */
