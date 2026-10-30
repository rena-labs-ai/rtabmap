#include <gtest/gtest.h>
#include <rtabmap/core/LabelRaster.h>
#include <rtabmap/utilite/UException.h>

using namespace rtabmap;

TEST(LabelLayoutTest, EachLabelerHasItsOwnBits)
{
	const LabelLayout layout = LabelLayout::parse("floorseg:2 floorplan:4 objseg:10");
	ASSERT_EQ(layout.size(), 3);
	EXPECT_EQ(layout.name(1), "floorplan");
	EXPECT_EQ(layout.maxLabel(0), 2);
	EXPECT_EQ(layout.maxLabel(1), 14);
	EXPECT_EQ(layout.maxLabel(2), 1022);

	uint16_t word = 0;
	EXPECT_EQ(layout.label(word, 0), -1) << "0: no vote";
	word = layout.pack(word, 0, 1);
	word = layout.pack(word, 1, 0);
	word = layout.pack(word, 2, 1022);
	EXPECT_EQ(layout.label(word, 0), 1);
	EXPECT_EQ(layout.label(word, 1), 0) << "seen with nothing there";
	EXPECT_EQ(layout.label(word, 2), 1022);
	EXPECT_EQ(word, uint16_t(2 | (1 << 2) | (1023 << 6)));

	word = layout.pack(word, 1, 14);
	EXPECT_EQ(layout.label(word, 0), 1) << "the other labelers keep theirs";
	EXPECT_EQ(layout.label(word, 1), 14);
	EXPECT_EQ(layout.label(word, 2), 1022);
}

TEST(LabelLayoutTest, RefusesWhatDoesNotFit)
{
	EXPECT_EQ(LabelLayout::parse("").size(), 0);
	EXPECT_THROW(LabelLayout::parse("a:8 b:9"), UException) << "17 bits";
	EXPECT_THROW(LabelLayout::parse("floorseg"), UException) << "no bits";
	EXPECT_THROW(LabelLayout::parse("a:0"), UException);
	EXPECT_THROW(LabelLayout::parse("a:2").pack(0, 0, 3), UException) << "3 needs more than 2 bits";
}

TEST(LabelRasterTest, CellsAreTheVotedOnesAtTheirCentres)
{
	const LabelLayout layout = LabelLayout::parse("floorseg:2 objseg:10");
	LabelRaster raster(10, -2, 0.1f, 2, 3);
	raster.set(layout, 0, 0, 0, 0);
	raster.set(layout, 0, 1, 2, 1);
	raster.set(layout, 1, 1, 2, 3);

	std::vector<cv::Vec2f> nothing;
	std::vector<cv::Vec3f> labelled;
	raster.cells(layout, 0, nothing, labelled);
	ASSERT_EQ(nothing.size(), 1u);
	EXPECT_NEAR(nothing[0][0], 1.05f, 1e-5);
	EXPECT_NEAR(nothing[0][1], -0.15f, 1e-5);
	ASSERT_EQ(labelled.size(), 1u);
	EXPECT_NEAR(labelled[0][0], 1.25f, 1e-5);
	EXPECT_NEAR(labelled[0][1], -0.05f, 1e-5);
	EXPECT_EQ(labelled[0][2], 1.0f);

	nothing.clear();
	labelled.clear();
	raster.cells(layout, 1, nothing, labelled);
	EXPECT_TRUE(nothing.empty());
	ASSERT_EQ(labelled.size(), 1u);
	EXPECT_EQ(labelled[0][2], 3.0f);
}

TEST(LabelRasterTest, EncodesAndDecodes)
{
	const LabelLayout layout = LabelLayout::parse("floorseg:2 floorplan:4 objseg:10");
	LabelRaster raster(-5, 7, 0.025f, 3, 2);
	raster.set(layout, 2, 2, 1, 500);
	raster.set(layout, 1, 0, 0, 6);

	const cv::Mat bytes = raster.encode();
	ASSERT_EQ(bytes.type(), CV_8UC1);
	EXPECT_EQ(bytes.total(), 17u + 3 * 2 * 2) << "header, then 16 bits per cell";
	const LabelRaster decoded = LabelRaster::decode(bytes);
	ASSERT_EQ(decoded.rows(), 3);
	ASSERT_EQ(decoded.cols(), 2);
	EXPECT_EQ(decoded.label(layout, 2, 2, 1), 500);
	EXPECT_EQ(decoded.label(layout, 1, 0, 0), 6);
	EXPECT_EQ(decoded.label(layout, 0, 0, 0), -1);
	EXPECT_EQ(cv::norm(decoded.encode(), bytes, cv::NORM_INF), 0.0);

	EXPECT_TRUE(LabelRaster::decode(cv::Mat()).empty());
	cv::Mat other = bytes.clone();
	other.at<unsigned char>(0, 0) = 2;
	EXPECT_THROW(LabelRaster::decode(other), UException) << "another version";
}
