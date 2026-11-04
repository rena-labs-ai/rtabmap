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

#ifndef CORELIB_INCLUDE_RTABMAP_CORE_LABELRASTER_H_
#define CORELIB_INCLUDE_RTABMAP_CORE_LABELRASTER_H_

#include "rtabmap/core/rtabmap_core_export.h"

#include <opencv2/core/core.hpp>
#include <cstdint>
#include <vector>

namespace rtabmap {

/**
 * @class LabelRaster
 * @brief One node's labels per local grid cell (see LocalGridMaker::createLabelRaster()):
 * a raster over the cells its label frames' points fall in, robot/base frame, each cell
 * one label (1 byte): 0 seen with nothing there, 1-127 a label, 255 no vote. Saved with the node's occupancy grid (encode()).
 */
class RTABMAP_CORE_EXPORT LabelRaster
{
public:
	LabelRaster() {}
	/// Every cell no vote: row r, column c is the cell (x0 + c, y0 + r) of cellSize meters.
	LabelRaster(int x0, int y0, float cellSize, int rows, int cols);

	bool empty() const {return labels_.empty();}
	int rows() const {return labels_.rows;}
	int cols() const {return labels_.cols;}

	/// label: 0-127.
	void set(int row, int col, int label);
	/// The label at cell (row, col), -1 for no vote.
	int label(int row, int col) const;
	/// The cells voted on, centres in the robot/base frame: seen with nothing there
	/// (label 0) in @p nothing, (x, y, label) in @p labelled.
	void cells(std::vector<cv::Vec2f> & nothing, std::vector<cv::Vec3f> & labelled) const;

	/// 1×N `CV_8UC1`: version, cell size, x0, y0, rows, cols, then the labels row by row.
	cv::Mat encode() const;
	/// encode()'s bytes back; empty for no bytes, an error for another version.
	static LabelRaster decode(const cv::Mat & bytes);

private:
	int x0_ = 0;
	int y0_ = 0;
	float cellSize_ = 0.0f;
	cv::Mat labels_; ///< CV_8UC1, kNoVote for no vote
};

} // namespace rtabmap

#endif /* CORELIB_INCLUDE_RTABMAP_CORE_LABELRASTER_H_ */
