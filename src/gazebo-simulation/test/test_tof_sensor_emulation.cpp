#include <common/macros.h>

THIRD_PARTY_HEADERS_BEGIN
#include <gtest/gtest.h>
THIRD_PARTY_HEADERS_END

class TofSensorEmulationTest : public ::testing::Test {
  // helper methods etc go here
};

/**!
* This is a placeholder for you REAL implementation unit tests.
*/
TEST_F(TofSensorEmulationTest, areTestsImplemented) {
  const bool test_is_implemented = false;
  ASSERT_TRUE(test_is_implemented);
}



// nothing to do here
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
