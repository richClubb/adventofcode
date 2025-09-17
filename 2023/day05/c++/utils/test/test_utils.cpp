#include <gtest/gtest.h>

#include <stdint.h>

#include <vector>

#include "utils.h"


TEST(UtilsTests, ExtractNumbers) {
    
    std::vector<uint64_t> numbers = extract_numbers("1 2 5 10 100\n");

    EXPECT_EQ(numbers.size(), 5);

    EXPECT_EQ(numbers[0], 1);
    EXPECT_EQ(numbers[1], 2);
    EXPECT_EQ(numbers[2], 5);
    EXPECT_EQ(numbers[3], 10);
    EXPECT_EQ(numbers[4], 100);
}

TEST(UtilsTests, GetSeeds_1) {
    
    std::vector<uint64_t> numbers = get_seeds("seeds: 1 2 5 10 100\n");

    EXPECT_EQ(numbers.size(), 5);

    EXPECT_EQ(numbers[0], 1);
    EXPECT_EQ(numbers[1], 2);
    EXPECT_EQ(numbers[2], 5);
    EXPECT_EQ(numbers[3], 10);
    EXPECT_EQ(numbers[4], 100);
}

TEST(UtilsTests, GetSeeds_sample) {
    
    std::vector<uint64_t> numbers = get_seeds("seeds: 79 14 55 13\n");

    EXPECT_EQ(numbers.size(), 4);

    EXPECT_EQ(numbers[0], 79);
    EXPECT_EQ(numbers[1], 14);
    EXPECT_EQ(numbers[2], 55);
    EXPECT_EQ(numbers[3], 13);
}

TEST(UtilsTests, GetSeeds_input) {
    
    std::vector<uint64_t> numbers = get_seeds("seeds: 28965817 302170009 1752849261 48290258 804904201 243492043 2150339939 385349830 1267802202 350474859 2566296746 17565716 3543571814 291402104 447111316 279196488 3227221259 47952959 1828835733 9607836\n");

    EXPECT_EQ(numbers.size(), 20);

    EXPECT_EQ(numbers[0], 28965817);
    EXPECT_EQ(numbers[1], 302170009);
    EXPECT_EQ(numbers[2], 1752849261);
    EXPECT_EQ(numbers[3], 48290258);
    EXPECT_EQ(numbers[4], 804904201);
    EXPECT_EQ(numbers[5], 243492043);
    EXPECT_EQ(numbers[6], 2150339939);
    EXPECT_EQ(numbers[7], 385349830);
    EXPECT_EQ(numbers[8], 1267802202);
    EXPECT_EQ(numbers[9], 350474859);
    EXPECT_EQ(numbers[10], 2566296746);
    EXPECT_EQ(numbers[11], 17565716);
    EXPECT_EQ(numbers[12], 3543571814);
    EXPECT_EQ(numbers[13], 291402104);
    EXPECT_EQ(numbers[14], 447111316);
    EXPECT_EQ(numbers[15], 279196488);
    EXPECT_EQ(numbers[16], 3227221259);
    EXPECT_EQ(numbers[17], 47952959);
    EXPECT_EQ(numbers[18], 1828835733);
    EXPECT_EQ(numbers[19], 9607836);
}