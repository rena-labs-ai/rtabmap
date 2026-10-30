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
#include <rtabmap/utilite/UStl.h>

#include <cstring>

namespace rtabmap {

namespace {

const unsigned char kVersion = 1;

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

LabelLayout LabelLayout::parse(const std::string & labelers)
{
	LabelLayout layout;
	int shift = 0;
	for(const std::string & labeler : uSplit(labelers, ' '))
	{
		if(labeler.empty())
		{
			continue;
		}
		const std::list<std::string> nameBits = uSplit(labeler, ':');
		UASSERT_MSG(nameBits.size() == 2, uFormat("Labeler \"%s\" is not name:bits (labelers \"%s\")", labeler.c_str(), labelers.c_str()).c_str());
		const int bits = uStr2Int(nameBits.back());
		UASSERT_MSG(bits >= 1, uFormat("Labeler \"%s\" needs at least 1 bit (labelers \"%s\")", labeler.c_str(), labelers.c_str()).c_str());
		layout.names_.push_back(nameBits.front());
		layout.bits_.push_back(bits);
		layout.shifts_.push_back(shift);
		shift += bits;
	}
	UASSERT_MSG(shift <= 16, uFormat("Labelers \"%s\" need %d bits, a cell holds 16", labelers.c_str(), shift).c_str());
	return layout;
}

uint16_t LabelLayout::pack(uint16_t word, int labeler, int label) const
{
	UASSERT(label >= 0 && label <= maxLabel(labeler));
	const int mask = (1 << bits_[labeler]) - 1;
	return uint16_t((word & ~(mask << shifts_[labeler])) | ((label + 1) << shifts_[labeler]));
}

int LabelLayout::label(uint16_t word, int labeler) const
{
	return ((word >> shifts_.at(labeler)) & ((1 << bits_[labeler]) - 1)) - 1;
}

LabelRaster::LabelRaster(int x0, int y0, float cellSize, int rows, int cols) :
		x0_(x0),
		y0_(y0),
		cellSize_(cellSize),
		words_(cv::Mat::zeros(rows, cols, CV_16UC1))
{
	UASSERT(cellSize > 0.0f && rows > 0 && cols > 0 && rows <= UINT16_MAX && cols <= UINT16_MAX);
}

void LabelRaster::set(const LabelLayout & layout, int labeler, int row, int col, int label)
{
	uint16_t & word = words_.at<uint16_t>(row, col);
	word = layout.pack(word, labeler, label);
}

int LabelRaster::label(const LabelLayout & layout, int labeler, int row, int col) const
{
	return layout.label(words_.at<uint16_t>(row, col), labeler);
}

void LabelRaster::cells(const LabelLayout & layout, int labeler, std::vector<cv::Vec2f> & nothing, std::vector<cv::Vec3f> & labelled) const
{
	for(int row=0; row<words_.rows; ++row)
	{
		for(int col=0; col<words_.cols; ++col)
		{
			const int l = label(layout, labeler, row, col);
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
	const size_t words = words_.total() * sizeof(uint16_t);
	cv::Mat bytes(1, int(kHeaderBytes + words), CV_8UC1);
	unsigned char * out = bytes.data;
	put(out, kVersion);
	put(out, cellSize_);
	put(out, int32_t(x0_));
	put(out, int32_t(y0_));
	put(out, uint16_t(words_.rows));
	put(out, uint16_t(words_.cols));
	std::memcpy(out, words_.data, words);
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
	UASSERT(bytes.total() == kHeaderBytes + size_t(rows) * cols * sizeof(uint16_t));
	LabelRaster raster(x0, y0, cellSize, rows, cols);
	std::memcpy(raster.words_.data, in, raster.words_.total() * sizeof(uint16_t));
	return raster;
}

} // namespace rtabmap
