#include <iostream>
#include <Petscpp/Vector.h>
#include <Petscpp/Petscpp.h>

int main(int argc, char *argv[])
{
  // Initialize Petsc
  Petscpp::App app(argc, argv);

  // Initialize a vector of size 16, with default value of 3.1
  Petscpp::Vector v1(16, 3.1);

  // Scale 'v1' and assign to a new vector
  Petscpp::Vector v2 = 2*v1;

  /* Iterate through local locally owned portion of 'v2' and multiply
   * each value component by 2
   *
   * 'v2.end()' is evaluated only once to overhead
   */
  for (auto i=v2.begin(), end=v2.end(); i!=end; ++i){
    *i *= 2.0;
  }

  /* Iterate through locally owned portion of 'v2' and print its
   * values to screen
   *
   * 'v2.end()' is evaluated only once to overhead
   */
  for (auto i=v2.begin(), end=v2.end(); i!=end; ++i){
    std::cout << "proc(" << Petscpp::procId() << "): "
              << *i << std::endl;
  }

  // Ok: valid assignment
  Petscpp::Vector v3 = v1 + v2;

  // Create a new vector of dimension 2
  Petscpp::Vector v4(2);

  /* Ok: (v1,v2,v3) have compatible parallel layouts. Original v4 is
   * destroyed, and a new vector with compatible layout to (v1,v2,v3)
   * is created. No temporaries are created.
   */
  v4 = v1 + v2 + v3;

  /* Compile-time error: current Petscpp implementation does not
   * consider aliasing
   */
  // v3 += v1 + v2;

  // OK. No aliasing
  v3.noAlias() += v1 + v2;

  /* Run-time error: 'v3' aliases itself. Current implementation does
   * not check for aliasing
   */
  // v3.noAlias() += v1 + v2 + v3;

  return 0;
}
