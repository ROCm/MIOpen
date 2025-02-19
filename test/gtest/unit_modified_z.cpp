/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/utility/modified_z.hpp>
#include <gtest/gtest.h>

TEST(CPU_UnitTestModifiedZ_NONE, TestMean)
{
    std::vector<double> testSorted     = {1, 2, 3, 4, 5, 6, 7};
    std::vector<double> testDuplicates = {1, 1, 1};
    std::vector<double> testUnsorted   = {5, 9, 11, 7, 3};
    std::vector<double> testOdd        = {13.5, 10.5, 12.5};
    std::vector<double> testEven       = {13.5, 10.5, 12.5, 15.5};
    std::vector<double> testSingle     = {1};
    std::vector<double> testEmpty      = {};

    EXPECT_DOUBLE_EQ(miopen::mean(testSorted), 4);
    EXPECT_DOUBLE_EQ(miopen::mean(testDuplicates), 1);
    EXPECT_DOUBLE_EQ(miopen::mean(testUnsorted), 7);
    EXPECT_DOUBLE_EQ(miopen::mean(testOdd), 12.16666666666667);
    EXPECT_DOUBLE_EQ(miopen::mean(testEven), 13.0);
    EXPECT_DOUBLE_EQ(miopen::mean(testSingle), 1);
    EXPECT_THROW(miopen::mean(testEmpty), miopen::Exception);
}

TEST(CPU_UnitTestModifiedZ_NONE, TestMedian)
{
    std::vector<double> testSorted     = {1, 2, 3, 4, 5, 6, 7};
    std::vector<double> testDuplicates = {1, 1, 1};
    std::vector<double> testUnsorted   = {5, 9, 11, 7, 3};
    std::vector<double> testOdd        = {13.5, 10.5, 12.5};
    std::vector<double> testEven       = {13.5, 10.5, 12.5, 15.5};
    std::vector<double> testSingle     = {1};
    std::vector<double> testEmpty      = {};

    EXPECT_DOUBLE_EQ(miopen::median(testSorted), 4);
    EXPECT_DOUBLE_EQ(miopen::median(testDuplicates), 1);
    EXPECT_DOUBLE_EQ(miopen::median(testUnsorted), 7);
    EXPECT_DOUBLE_EQ(miopen::median(testOdd), 12.5);
    EXPECT_DOUBLE_EQ(miopen::median(testEven), 13.0);
    EXPECT_DOUBLE_EQ(miopen::median(testSingle), 1);
    EXPECT_THROW(miopen::median(testEmpty), miopen::Exception);
}

TEST(CPU_UnitTestModifiedZ_NONE, TestMedianOfSortedData)
{
    std::vector<double> testSorted     = {1, 2, 3, 4, 5, 6, 7};
    std::vector<double> testDuplicates = {1, 1, 1};
    std::vector<double> testUnsorted   = {5, 9, 11, 7, 3};
    std::vector<double> testSingle     = {1};
    std::vector<double> testEmpty      = {};

    EXPECT_DOUBLE_EQ(miopen::medianOfSortedData(testSorted), 4);
    EXPECT_DOUBLE_EQ(miopen::medianOfSortedData(testDuplicates), 1);
    EXPECT_NE(miopen::medianOfSortedData(testUnsorted), 7);
    EXPECT_DOUBLE_EQ(miopen::medianOfSortedData(testSingle), 1);
    EXPECT_THROW(miopen::medianOfSortedData(testEmpty), miopen::Exception);
}

TEST(CPU_UnitTestModifiedZ_NONE, TestMedianAbsoluteDeviation)
{
    std::vector<double> testNoDeviation      = {1, 1, 1};
    std::vector<double> testZeroMAD          = {7, 7, 7, 1000};
    std::vector<double> testDeviationOdd     = {1, 2, 3, 4, 5};
    std::vector<double> testDeviationEven    = {1, 2, 3, 4};
    std::vector<double> testDeviationRepeats = {1, 2, 2, 3, 4};
    std::vector<double> testSingle           = {1};
    std::vector<double> testEmpty            = {};

    std::vector<double> mad1 = miopen::medianAbsoluteDeviation(testNoDeviation);
    std::vector<double> mad2 = miopen::medianAbsoluteDeviation(testZeroMAD);
    std::vector<double> mad3 = miopen::medianAbsoluteDeviation(testDeviationOdd);
    std::vector<double> mad4 = miopen::medianAbsoluteDeviation(testDeviationEven);
    std::vector<double> mad5 = miopen::medianAbsoluteDeviation(testDeviationRepeats);
    std::vector<double> mad6 = miopen::medianAbsoluteDeviation(testSingle);

    std::vector<double> expected1 = {0, 0, 0};
    std::vector<double> expected2 = {0, 0, 0, 993};
    std::vector<double> expected3 = {2, 1, 0, 1, 2};
    std::vector<double> expected4 = {1.5, 0.5, 0.5, 1.5};
    std::vector<double> expected5 = {1, 0, 0, 1, 2};
    std::vector<double> expected6 = {0};

    EXPECT_EQ(mad1, expected1);
    EXPECT_EQ(mad2, expected2);
    EXPECT_EQ(mad3, expected3);
    EXPECT_EQ(mad4, expected4);
    EXPECT_EQ(mad5, expected5);
    EXPECT_EQ(mad6, expected6);
    EXPECT_THROW(miopen::medianAbsoluteDeviation(testEmpty), miopen::Exception);
}

TEST(CPU_UnitTestModifiedZ_NONE, TestModifiedZScores)
{
    std::vector<double> testNoDeviation      = {1, 1, 1};
    std::vector<double> testZeroMAD          = {7, 7, 7, 1000};
    std::vector<double> testDeviationOdd     = {1, 2, 3, 4, 5};
    std::vector<double> testDeviationEven    = {1, 2, 3, 4};
    std::vector<double> testDeviationRepeats = {1, 2, 2, 3, 4};
    std::vector<double> testSingle           = {1};
    std::vector<double> testEmpty            = {};

    std::vector<double> modZScores1 = miopen::modifiedZScores(testNoDeviation);
    std::vector<double> modZScores2 = miopen::modifiedZScores(testZeroMAD);
    std::vector<double> modZScores3 = miopen::modifiedZScores(testDeviationOdd);
    std::vector<double> modZScores4 = miopen::modifiedZScores(testDeviationEven);
    std::vector<double> modZScores5 = miopen::modifiedZScores(testDeviationRepeats);
    std::vector<double> modZScores6 = miopen::modifiedZScores(testSingle);

    std::vector<double> expected1 = {0, 0, 0};
    std::vector<double> expected2 = {0, 0, 0, 0};
    std::vector<double> expected3 = {0.6745 * -2, 0.6745 * -1, 0, 0.6745, 0.6745 * 2};
    std::vector<double> expected4 = {0.6745 * -1.5, 0.6745 * -0.5, 0.6745 * 0.5, 0.6745 * 1.5};
    std::vector<double> expected5 = {0.6745 * -1, 0, 0, 0.6745 * 1, 0.6745 * 2};
    std::vector<double> expected6 = {0};

    EXPECT_EQ(modZScores1, expected1);
    EXPECT_EQ(modZScores2, expected2);
    EXPECT_EQ(modZScores3, expected3);
    EXPECT_EQ(modZScores4, expected4);
    EXPECT_EQ(modZScores5, expected5);
    EXPECT_EQ(modZScores6, expected6);
    EXPECT_THROW(miopen::modifiedZScores(testEmpty), miopen::Exception);
}

TEST(CPU_UnitTestModifiedZ_NONE, TestRemoveHighOutliersAndGetMean)
{
    std::vector<double> testNoDeviation      = {1, 1, 1};
    std::vector<double> testZeroMAD          = {7, 7, 7, 1000};
    std::vector<double> testDeviationOdd     = {1, 2, 3, 4, 5};
    std::vector<double> testDeviationEven    = {1, 2, 3, 4};
    std::vector<double> testDeviationRepeats = {1, 2, 2, 3, 10};
    std::vector<double> testWithOutliers1    = {1, 2, 3, 4, 5, 900, 1000};
    std::vector<double> testWithOutliers2    = {1, 2, 2, 3, 900, 1000};
    std::vector<double> testWithOutliers3    = {
        1, 2, 2, 3, 900, 1000, 1000, 1105, 1106, 1107, 1108, 100000};
    std::vector<double> testSingle = {1};
    std::vector<double> testEmpty  = {};

    double mean1 = miopen::removeHighOutliersAndGetMean(testNoDeviation, 1.0);
    double mean2 = miopen::removeHighOutliersAndGetMean(testZeroMAD, 1.0);
    double mean3 = miopen::removeHighOutliersAndGetMean(testDeviationOdd, 1.0);
    double mean4 = miopen::removeHighOutliersAndGetMean(testDeviationEven, 1.0);
    double mean5 = miopen::removeHighOutliersAndGetMean(testDeviationRepeats, 1.0);
    double mean6 = miopen::removeHighOutliersAndGetMean(testWithOutliers1, 1.0);
    double mean7 = miopen::removeHighOutliersAndGetMean(testWithOutliers2, 1.0);
    double mean8 = miopen::removeHighOutliersAndGetMean(testWithOutliers3, 1.0);
    double mean9 = miopen::removeHighOutliersAndGetMean(testSingle, 1.0);

    EXPECT_DOUBLE_EQ(mean1, 1);
    EXPECT_DOUBLE_EQ(mean2, 255.25);
    EXPECT_DOUBLE_EQ(mean3, 2.5);
    EXPECT_DOUBLE_EQ(mean4, 2);
    EXPECT_DOUBLE_EQ(mean5, 2);
    EXPECT_DOUBLE_EQ(mean6, 3);
    EXPECT_DOUBLE_EQ(mean7, 2);
    EXPECT_DOUBLE_EQ(mean8, 666.72727272727275);
    EXPECT_DOUBLE_EQ(mean9, 1);
    EXPECT_THROW(miopen::removeHighOutliersAndGetMean(testEmpty, 1.0), miopen::Exception);
    EXPECT_THROW(miopen::removeHighOutliersAndGetMean(testDeviationRepeats, -1.0),
                 miopen::Exception);
}
