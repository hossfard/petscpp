#include <gtest/gtest.h>
#include <Petscpp/Petscpp.h>


int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  Petscpp::App app(argc, argv);

  return RUN_ALL_TESTS();
}
