/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "rtabmap/core/LabelRaster.h"

#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UConversion.h>

#include <cstring>

namespace rtabmap {

namespace {

// 3: one label per cell (2 packed labelers in 8 bits, 1 in 16)
const unsigned char kVersion = 3;
const unsigned char kNoVote = 255;

// version, cell size, x0, y0, rows, cols: packed one after the other.
const size_t kHeaderBytes = sizeof(unsigned char) + sizeof(float) + 2*sizeof(int32_t) + 2*sizeof(uint16_t);

template<typename T>
void put(unsigned char *& out, const T & value)
{
	std::memcpy(out, &value, sizeof(T));
	out += sizeof(T);
}

template<typename T>
T take(const unsigned char *& in)
{
	T value;
	std::memcpy(&value, in, sizeof(T));
	in += sizeof(T);
	return value;
}

} // namespace

LabelRaster::LabelRaster(int x0, int y0, float cellSize, int rows, int cols) :
		x0_(x0),
		y0_(y0),
		cellSize_(cellSize),
		labels_(rows, cols, CV_8UC1, cv::Scalar(kNoVote))
{
	UASSERT(cellSize > 0.0f && rows > 0 && cols > 0 && rows <= UINT16_MAX && cols <= UINT16_MAX);
}

void LabelRaster::set(int row, int col, int label)
{
	UASSERT(label >= 0 && label <= 127);
	labels_.at<unsigned char>(row, col) = (unsigned char)label;
}

int LabelRaster::label(int row, int col) const
{
	const unsigned char label = labels_.at<unsigned char>(row, col);
	return label == kNoVote ? -1 : label;
}

void LabelRaster::cells(std::vector<cv::Vec2f> & nothing, std::vector<cv::Vec3f> & labelled) const
{
	for(int row=0; row<labels_.rows; ++row)
	{
		for(int col=0; col<labels_.cols; ++col)
		{
			const int l = label(row, col);
			if(l < 0)
			{
				continue;
			}
			const float x = (x0_ + col + 0.5f) * cellSize_;
			const float y = (y0_ + row + 0.5f) * cellSize_;
			if(l == 0)
			{
				nothing.push_back(cv::Vec2f(x, y));
			}
			else
			{
				labelled.push_back(cv::Vec3f(x, y, float(l)));
			}
		}
	}
}

cv::Mat LabelRaster::encode() const
{
	if(empty())
	{
		return cv::Mat();
	}
	const size_t cells = labels_.total();
	cv::Mat bytes(1, int(kHeaderBytes + cells), CV_8UC1);
	unsigned char * out = bytes.data;
	put(out, kVersion);
	put(out, cellSize_);
	put(out, int32_t(x0_));
	put(out, int32_t(y0_));
	put(out, uint16_t(labels_.rows));
	put(out, uint16_t(labels_.cols));
	std::memcpy(out, labels_.data, cells);
	return bytes;
}

LabelRaster LabelRaster::decode(const cv::Mat & bytes)
{
	if(bytes.empty())
	{
		return LabelRaster();
	}
	UASSERT(bytes.type() == CV_8UC1 && bytes.rows == 1 && bytes.total() >= kHeaderBytes);
	const unsigned char * in = bytes.data;
	const unsigned char version = take<unsigned char>(in);
	UASSERT_MSG(version == kVersion, uFormat("Label raster version %d, this rtabmap reads %d", (int)version, (int)kVersion).c_str());
	const float cellSize = take<float>(in);
	const int32_t x0 = take<int32_t>(in);
	const int32_t y0 = take<int32_t>(in);
	const uint16_t rows = take<uint16_t>(in);
	const uint16_t cols = take<uint16_t>(in);
	UASSERT(bytes.total() == kHeaderBytes + size_t(rows) * cols);
	LabelRaster raster(x0, y0, cellSize, rows, cols);
	std::memcpy(raster.labels_.data, in, raster.labels_.total());
	return raster;
}

} // namespace rtabmap
