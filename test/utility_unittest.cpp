#include <limits>
#include <gtest/gtest.h>
#include "Petscpp/Vector.h"
#include "Petscpp/Petscpp.h"
#include "Petscpp/Utility.h"


/* MPI Send/Receive primitives helpers */
TEST(PUtility, SendReceive){

  // Run only if ran with MPI
  if (Petscpp::procCount() > 1){

    int const int_neg_value = -10;
    int const int_pos_value = 391;
    double const dbl_neg_value = -3.14159265;
    double const dbl_pos_value = +3.14159265;
    double const abs_err = 1E-10;

    if (Petscpp::procId() == 0){
      // 0-1s
      Petscpp::send<int>(int_neg_value, 1);
      // 0-2s
      Petscpp::send<int>(int_pos_value, 1);
      // 0-3s
      Petscpp::send<double>(dbl_neg_value, 1);
      // 0-4s
      Petscpp::send<double>(dbl_pos_value, 1);
    }
    if (Petscpp::procId() == 1){
      // 1-1r
      EXPECT_EQ(int_neg_value, Petscpp::receive<int>(0));
      // 1-2r
      EXPECT_EQ(int_pos_value, Petscpp::receive<int>(0));
      // 1-3r
      ASSERT_NEAR(dbl_neg_value, Petscpp::receive<double>(0), abs_err);
      // 1-4r
      ASSERT_NEAR(dbl_pos_value, Petscpp::receive<double>(0), abs_err);

      // Send back to proc0
      // 1-1s
      Petscpp::send<int>(int_pos_value, 0);
      // 1-2s
      Petscpp::send<int>(int_neg_value, 0);
      // 1-3s
      Petscpp::send<double>(dbl_pos_value, 0);
      // 1-4s
      Petscpp::send<double>(dbl_neg_value, 0);
    }
    if (Petscpp::procId() == 0){
      // 0-1r
      EXPECT_EQ(int_pos_value, Petscpp::receive<int>(1));
      // 0-2r
      EXPECT_EQ(int_neg_value, Petscpp::receive<int>(1));
      // 0-3r
      EXPECT_NEAR(dbl_pos_value, Petscpp::receive<double>(1), abs_err);
      // 0-4r
      EXPECT_NEAR(dbl_neg_value, Petscpp::receive<double>(1), abs_err);
    }

  }

}


TEST(PUtility, SendReceive_Vec){
  // TODO: 
}
