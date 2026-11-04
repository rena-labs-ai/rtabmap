#include <gtest/gtest.h>
#include <rtabmap/core/LabelRaster.h>
#include <rtabmap/utilite/UException.h>

using namespace rtabmap;

TEST(LabelRasterTest, CellsAreTheVotedOnesAtTheirCentres)
{
	LabelRaster raster(10, -2, 0.1f, 2, 3);
	raster.set(0, 0, 0);
	raster.set(1, 2, 3);
	EXPECT_EQ(raster.label(0, 1), -1) << "no vote";
	EXPECT_THROW(raster.set(0, 1, 128), UException) << "128-255 is not a label";

	std::vector<cv::Vec2f> nothing;
	std::vector<cv::Vec3f> labelled;
	raster.cells(nothing, labelled);
	ASSERT_EQ(nothing.size(), 1u);
	EXPECT_NEAR(nothing[0][0], 1.05f, 1e-5);
	EXPECT_NEAR(nothing[0][1], -0.15f, 1e-5);
	ASSERT_EQ(labelled.size(), 1u);
	EXPECT_NEAR(labelled[0][0], 1.25f, 1e-5);
	EXPECT_NEAR(labelled[0][1], -0.05f, 1e-5);
	EXPECT_EQ(labelled[0][2], 3.0f);
}

TEST(LabelRasterTest, EncodesAndDecodes)
{
	LabelRaster raster(-5, 7, 0.025f, 3, 2);
	raster.set(2, 1, 50);
	raster.set(0, 0, 0);

	const cv::Mat bytes = raster.encode();
	ASSERT_EQ(bytes.type(), CV_8UC1);
	EXPECT_EQ(bytes.total(), 17u + 3 * 2) << "header, then a byte per cell";
	const LabelRaster decoded = LabelRaster::decode(bytes);
	ASSERT_EQ(decoded.rows(), 3);
	ASSERT_EQ(decoded.cols(), 2);
	EXPECT_EQ(decoded.label(2, 1), 50);
	EXPECT_EQ(decoded.label(0, 0), 0);
	EXPECT_EQ(decoded.label(1, 0), -1);
	EXPECT_EQ(cv::norm(decoded.encode(), bytes, cv::NORM_INF), 0.0);

	EXPECT_TRUE(LabelRaster::decode(cv::Mat()).empty());
	cv::Mat other = bytes.clone();
	other.at<unsigned char>(0, 0) = 2;
	EXPECT_THROW(LabelRaster::decode(other), UException) << "version 2, packed labelers";
}
