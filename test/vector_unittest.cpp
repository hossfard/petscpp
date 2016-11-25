#include <limits>
#include <gtest/gtest.h>
#include <Petscpp/Vector.h>
#include <Petscpp/Petscpp.h>
#include <iostream>


/*
 * PVector(size, alpha) contructor
 */
TEST(Vector, ctor){
  using namespace Petscpp;

  int    const vec_size   = 103;
  double const init_value = 1.2;
  double const abs_error  = 1E-8;

  Vector vec(vec_size, init_value);

  /* Test vector's reported size against assigned size */
  EXPECT_EQ(vec.globalSize(), vec_size);

  /* Local size of the vector must match its global size if running on
     single processor */
  if (procCount() == 1)
    EXPECT_EQ(vec.localSize(), vec_size);

  /* Test assigned values of the vector
   *  
   * Each processor can only access elements that are assigned to
   * it. For the time being, test for single processor
   */

  int const size = vec.localSize();
  int const i0 = vec.startingIndex();

  for (int i=i0; i<i0+size; ++i){
    ASSERT_NEAR(vec[i].value(), init_value, abs_error);
  }
}


/*
 * Assignment Operator
 */
TEST(Vector, AssignmentOperator){
  using namespace Petscpp;

  int    const vec_size   = procCount()*512;
  double const init_value = 1.2;
  double const abs_error  = 1E-8;

  Vector vec(vec_size, init_value);

  Vector copy;
  EXPECT_EQ(copy.globalSize(), 0);

  // Copy the vector
  copy = vec;

  /* Test vector's reported size against assigned size */
  EXPECT_EQ(copy.globalSize(), vec_size);
  EXPECT_EQ(vec.localSize(), copy.localSize());

  /* Local size of the vector must match its global size if running on
     single processor */
  if (procCount() == 1)
    EXPECT_EQ(copy.localSize(), vec_size);

  /* Test assigned values of the vector
   *  
   * Each processor can only access elements that are assigned to
   * it. For the time being, test for single processor
   */

  int const size = vec.localSize();
  int const i0 = vec.startingIndex();

  for (int i=i0; i<i0+size; ++i){
    ASSERT_NEAR(copy[i].value(), init_value, abs_error);
  }
}



TEST(Vector, SimpleAddition){
  using namespace Petscpp;

  int    const vec_size   = 103;
  double const abs_error  = 1E-8;
  double const init_value_1 = 1.5;
  double const init_value_2 = 3.5;
  double const expected_sum = init_value_1 + init_value_2;
  
  Vector vec_1(vec_size, init_value_1);
  Vector vec_2(vec_size, init_value_2);
  Vector vec_sum(vec_size, 0);
  
  vec_sum.noAlias() += vec_1 + vec_2;
  Vector vec_sum_copyctor = vec_1 + vec_2;

  // Sizes must match
  EXPECT_EQ(vec_1.localSize(), vec_sum.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum.globalSize());
  EXPECT_EQ(vec_1.localSize(),  vec_sum_copyctor.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum_copyctor.globalSize());

  int const size = vec_sum.localSize();
  int const i0 = vec_sum.startingIndex();

  for (int i=i0; i<size+i0; ++i){
    ASSERT_NEAR(vec_sum[i].value(), expected_sum, abs_error);
    ASSERT_NEAR(vec_sum_copyctor[i].value(), expected_sum, abs_error);
  }
}


TEST(Vector, alpha_v1_plus_alpha_v2){
  using namespace Petscpp;

  int    const vec_size   = 103;
  double const abs_error  = 1E-8;
  double const init_value_1 = 1.5;
  double const init_value_2 = 3.5;
  double const alpha_1 = 2.0;
  double const alpha_2 = 5.0;
  double const expected_sum = alpha_1*init_value_1 + alpha_2*init_value_2;
  
  Vector vec_1(vec_size, init_value_1);
  Vector vec_2(vec_size, init_value_2);
  Vector vec_sum(vec_size, 0);
  
  // Sum up the scaled vectors
  vec_sum.noAlias() += alpha_1*vec_1 + alpha_2*vec_2;
  Vector vec_sum_copyctor = alpha_1*vec_1 + alpha_2*vec_2;

  // Local and global sizes must match
  EXPECT_EQ(vec_1.localSize(), vec_sum.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum.globalSize());
  EXPECT_EQ(vec_1.localSize(),  vec_sum_copyctor.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum_copyctor.globalSize());

  int const size = vec_sum.localSize();
  int const i0 = vec_sum.startingIndex();

  for (int i=i0; i<i0+size; ++i){
    ASSERT_NEAR(vec_sum[i].value(), expected_sum, abs_error);
    ASSERT_NEAR(vec_sum_copyctor[i].value(), expected_sum, abs_error);
  }
}


TEST(Vector, alpha_v1_minus_alpha_v2){
  using namespace Petscpp;

  int    const vec_size   = 103;
  double const abs_error  = 1E-8;
  double const init_value_1 = 1.5;
  double const init_value_2 = 3.5;
  double const alpha_1 = 2.0;
  double const alpha_2 = 5.0;
  double const expected_sum = alpha_1*init_value_1 - alpha_2*init_value_2;
  
  Vector vec_1(vec_size, init_value_1);
  Vector vec_2(vec_size, init_value_2);
  Vector vec_sum(vec_size, 0);
  
  // Sum up the scaled vectors
  vec_sum.noAlias() += alpha_1*vec_1 - alpha_2*vec_2;
  Vector vec_sum_copyctor = alpha_1*vec_1 - alpha_2*vec_2;

  // Local and global sizes must match
  EXPECT_EQ(vec_1.localSize(), vec_sum.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum.globalSize());
  EXPECT_EQ(vec_1.localSize(),  vec_sum_copyctor.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum_copyctor.globalSize());

  int const size = vec_sum.localSize();
  int const i0 = vec_sum.startingIndex();

  for (int i=i0; i<i0+size; ++i){
    ASSERT_NEAR(vec_sum[i].value(), expected_sum, abs_error);
    ASSERT_NEAR(vec_sum_copyctor[i].value(), expected_sum, abs_error);
  }
}


TEST(Vector, minus_v1_plus_alpha_v2){
  using namespace Petscpp;

  int    const vec_size     = 103;
  double const abs_error    = 1E-8;
  double const init_value_1 = 1.5;
  double const init_value_2 = 3.5;
  double const alpha_2      = 5.0;
  double const expected_sum = -init_value_1 + alpha_2*init_value_2;
  
  Vector vec_1(vec_size, init_value_1);
  Vector vec_2(vec_size, init_value_2);
  Vector vec_sum(vec_size, 0);
  
  // Sum up the scaled vectors
  vec_sum.noAlias() += -vec_1 + alpha_2*vec_2;
  Vector vec_sum_copyctor = -vec_1 + alpha_2*vec_2;

  // Local and global sizes must match
  EXPECT_EQ(vec_1.localSize(), vec_sum.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum.globalSize());
  EXPECT_EQ(vec_1.localSize(),  vec_sum_copyctor.localSize());
  EXPECT_EQ(vec_1.globalSize(), vec_sum_copyctor.globalSize());

  int const size = vec_sum.localSize();
  int const i0   = vec_sum.startingIndex();

  for (int i=i0; i<size+i0; ++i){
    ASSERT_NEAR(vec_sum[i].value(), expected_sum, abs_error);
    ASSERT_NEAR(vec_sum_copyctor[i].value(), expected_sum, abs_error);
  }
}


TEST(Vector, ScaleAssignment){
  using namespace Petscpp;

  double const initValue = 3.1;
  double const expectedVal = 2*3.1;
  size_t const vecSize = 128*procCount();
  double const abs_error = 1E-8;

  Petscpp::Vector v1(vecSize, initValue);
  Petscpp::Vector v2 = 2*v1;

  // Sizes must match
  EXPECT_EQ(v1.localSize(), v2.localSize());
  EXPECT_EQ(v1.globalSize(), v2.globalSize());

  int const size = v2.localSize();
  int const i0   = v2.startingIndex();

  for (int i=i0; i<size+i0; ++i){
    ASSERT_NEAR(v2[i].value(), expectedVal, abs_error);
  }
}


TEST(Vector, Iterator){
  using namespace Petscpp;

  double const initValue = 3.1;
  double const expectedVal = 2*3.1;
  size_t const vecSize = 128*procCount();
  double const abs_error = 1E-8;

  Petscpp::Vector v1(vecSize, initValue);
  Petscpp::Vector v2 = 2*v1;

  // Sizes must match
  EXPECT_EQ(v1.localSize(), v2.localSize());
  EXPECT_EQ(v1.globalSize(), v2.globalSize());

  int const size = v2.localSize();
  int const i0   = v2.startingIndex();

  for (auto i=v2.begin(), end=v2.end(); i!=end; ++i){
    ASSERT_NEAR(*i, expectedVal, abs_error);
  }
}


// Test for checking operator()(...)
TEST(Vector, VectorSlice){
  using namespace Petscpp;

  // Create a vector of size 100*number_of_processors
  int    const procCount  = Petscpp::procCount();
  int    const vec_size   = procCount*100;
  double const abs_error  = 1E-8;
  double const init_value = 6.391;
  double const offset_val = 2.5;
  Vector vec(vec_size, init_value);

  // Set the even indices of the first processor equal to offset_value
  // All other vectors will be equal to initial value assigned
  if (procId() == 0){
    vec({0,20,40}) = offset_val;
  }

  int const size = vec.localSize();
  int const i0   = vec.startingIndex();

  if (procId() == 0){
    for (int i=i0; i<size+i0; ++i){
      if ( (i == 0) || (i == 20) || (i == 40) )
        ASSERT_NEAR(vec[i].value(), offset_val, abs_error);
      else
        ASSERT_NEAR(vec[i].value(), init_value, abs_error);
    }
  }
  else{
    for (int i=i0; i<size+i0; ++i)
      ASSERT_NEAR(vec[i].value(), init_value, abs_error);
  }
}


TEST(Vector, VectorSlice_2){
  using namespace Petscpp;

  // Create a vector of size 100*number_of_processors
  int    const procCount  = Petscpp::procCount();
  int    const vec_size   = procCount*100;
  double const abs_error  = 1E-8;
  double const init_value = 6.391;
  double const offset_val = 2.5;
  Vector vec(vec_size, init_value);

  int const size = vec.localSize();
  int const i0   = vec.startingIndex();

  // Set {0,20,40} from the starting index of the processor's share of
  // the vector to the offset_value
  vec({0+i0, i0+20, i0+40}) = offset_val;

  for (int i=i0; i<size+i0; ++i){
    if ( (i == i0) || (i == i0+20) || (i == i0+40) )
      ASSERT_NEAR(vec[i].value(), offset_val, abs_error);
    else
      ASSERT_NEAR(vec[i].value(), init_value, abs_error);
  }
}
