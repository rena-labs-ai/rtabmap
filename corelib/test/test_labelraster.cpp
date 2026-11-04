#include <gtest/gtest.h>
#include <rtabmap/core/LabelRaster.h>
#include <rtabmap/utilite/UException.h>

using namespace rtabmap;

TEST(LabelLayoutTest, EachLabelerHasItsOwnBits)
{
	const LabelLayout layout = LabelLayout::parse("floorseg:2 floorplan:6");
	ASSERT_EQ(layout.size(), 2);
	EXPECT_EQ(layout.name(1), "floorplan");
	EXPECT_EQ(layout.maxLabel(0), 2);
	EXPECT_EQ(layout.maxLabel(1), 62);

	LabelWord word = 0;
	EXPECT_EQ(layout.label(word, 0), -1) << "0: no vote";
	word = layout.pack(word, 0, 1);
	word = layout.pack(word, 1, 0);
	EXPECT_EQ(layout.label(word, 0), 1);
	EXPECT_EQ(layout.label(word, 1), 0) << "seen with nothing there";
	EXPECT_EQ(word, LabelWord(2 | (1 << 2)));

	word = layout.pack(word, 1, 62);
	EXPECT_EQ(layout.label(word, 0), 1) << "the other labeler keeps its";
	EXPECT_EQ(layout.label(word, 1), 62);
	EXPECT_EQ(word, LabelWord(2 | (63 << 2)));
}

TEST(LabelLayoutTest, RefusesWhatDoesNotFit)
{
	EXPECT_EQ(LabelLayout::parse("").size(), 0);
	EXPECT_THROW(LabelLayout::parse("a:4 b:5"), UException) << "9 bits";
	EXPECT_THROW(LabelLayout::parse("floorseg"), UException) << "no bits";
	EXPECT_THROW(LabelLayout::parse("a:-1"), UException) << "0 bits is an input only, fewer is nothing";
	EXPECT_THROW(LabelLayout::parse("a:2").pack(0, 0, 3), UException) << "3 needs more than 2 bits";
}

TEST(LabelRasterTest, CellsAreTheVotedOnesAtTheirCentres)
{
	const LabelLayout layout = LabelLayout::parse("floorseg:2 floorplan:6");
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
	const LabelLayout layout = LabelLayout::parse("floorseg:2 floorplan:6");
	LabelRaster raster(-5, 7, 0.025f, 3, 2);
	raster.set(layout, 1, 2, 1, 50);
	raster.set(layout, 0, 0, 0, 1);

	const cv::Mat bytes = raster.encode();
	ASSERT_EQ(bytes.type(), CV_8UC1);
	EXPECT_EQ(bytes.total(), 17u + 3 * 2) << "header, then 8 bits per cell";
	const LabelRaster decoded = LabelRaster::decode(bytes);
	ASSERT_EQ(decoded.rows(), 3);
	ASSERT_EQ(decoded.cols(), 2);
	EXPECT_EQ(decoded.label(layout, 1, 2, 1), 50);
	EXPECT_EQ(decoded.label(layout, 0, 0, 0), 1);
	EXPECT_EQ(decoded.label(layout, 1, 0, 0), -1);
	EXPECT_EQ(cv::norm(decoded.encode(), bytes, cv::NORM_INF), 0.0);

	EXPECT_TRUE(LabelRaster::decode(cv::Mat()).empty());
	cv::Mat other = bytes.clone();
	other.at<unsigned char>(0, 0) = 1;
	EXPECT_THROW(LabelRaster::decode(other), UException) << "version 1, 16-bit words";
}

TEST(LabelLayoutTest, ALabelerOfZeroBitsIsAnInputOnly)
{
	const LabelLayout layout = LabelLayout::parse("floorseg:0 floorplan:8");
	ASSERT_EQ(layout.size(), 2);
	EXPECT_EQ(layout.index("floorseg"), 0);
	EXPECT_EQ(layout.index("floorplan"), 1);
	EXPECT_EQ(layout.index("objseg"), -1);
	EXPECT_EQ(layout.maxLabel(0), -1) << "stores no label";
	EXPECT_EQ(layout.maxLabel(1), 254);
	const LabelWord word = layout.pack(0, 1, 254);
	EXPECT_EQ(layout.label(word, 1), 254);
	EXPECT_EQ(layout.label(word, 0), -1);
}
