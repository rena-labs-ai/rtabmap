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

#include <rtabmap/core/global_map/ObjectGrid.h>
#include <rtabmap/core/util3d_transforms.h>

namespace rtabmap {

namespace {

cv::Point cellIndex(const float * xy, const Transform & pose, float cellSize)
{
	const cv::Point3f point = util3d::transformPoint(cv::Point3f(xy[0], xy[1], 0), pose);
	return cv::Point((int)std::floor(point.x / cellSize), (int)std::floor(point.y / cellSize));
}

// Calls visit(index, label) for each cell of grid at pose, label 0 for floor. Objects
// first: where a node's object and floor fall in one cell, the object gets its vote.
template<typename Visit>
void forEachCell(const LocalGrid & grid, const Transform & pose, float cellSize, Visit visit)
{
	for(int i=0; i<grid.obstacleCells.cols; ++i)
	{
		const float * cell = grid.obstacleCells.ptr<float>(0, i);
		visit(cellIndex(cell, pose, cellSize), (int)cell[2]);
	}
	for(int i=0; i<grid.groundCells.cols; ++i)
	{
		visit(cellIndex(grid.groundCells.ptr<float>(0, i), pose, cellSize), 0);
	}
}

} // namespace

ObjectGrid::ObjectGrid(const LocalGridCache * cache, const ParametersMap & parameters) :
	GlobalMap(cache, parameters)
{
}

void ObjectGrid::clear()
{
	cells_ = cv::Mat();
	origin_ = cv::Point();
	GlobalMap::clear();
}

cv::Mat ObjectGrid::getMap(float & xMin, float & yMin) const
{
	xMin = minValues_[0];
	yMin = minValues_[1];
	const float occupied = logodds(occupancyThr_);
	cv::Mat map(cells_.size(), CV_8SC1, cv::Scalar(-1));
	for(int i=0; i<cells_.rows; ++i)
	{
		for(int j=0; j<cells_.cols; ++j)
		{
			const cv::Vec3f & cell = cells_.at<cv::Vec3f>(i, j);
			if(cell[2] == 0.0f)
			{
				continue;
			}
			map.at<signed char>(i, j) = cell[0] >= occupied ? (signed char)cell[1] : 0;
		}
	}
	return map;
}

cv::Mat ObjectGrid::getLastNodeMap(float & xMin, float & yMin) const
{
	xMin = minValues_[0];
	yMin = minValues_[1];
	cv::Mat nodes;
	cv::extractChannel(cells_, nodes, 2);
	nodes.convertTo(nodes, CV_32SC1);
	return nodes;
}

void ObjectGrid::assemble(const std::list<std::pair<int, Transform> > & newPoses)
{
	cv::Rect bounds;
	for(const std::pair<int, Transform> & pose : newPoses)
	{
		std::map<int, LocalGrid>::const_iterator grid = cache().find(pose.first);
		if(grid == cache().end())
		{
			continue;
		}
		forEachCell(grid->second, pose.second, cellSize_, [&](const cv::Point & index, int)
		{
			bounds |= cv::Rect(index, cv::Size(1, 1));
		});
	}
	grow(bounds);

	for(const std::pair<int, Transform> & pose : newPoses)
	{
		std::map<int, LocalGrid>::const_iterator grid = cache().find(pose.first);
		if(grid == cache().end())
		{
			continue;
		}
		addAssembledNode(pose.first, pose.second);
		forEachCell(grid->second, pose.second, cellSize_, [&](const cv::Point & index, int label)
		{
			cv::Vec3f & cell = cells_.at<cv::Vec3f>(index - origin_);
			if(pose.first > 0 && cell[2] == pose.first)
			{
				return;
			}
			if(pose.first > 0)
			{
				// The live frame (id <= 0) votes but is no node: it neither marks a cell
				// observed nor replaces the node that last did, as OccupancyGrid.
				cell[2] = pose.first;
			}
			vote(cell, label);
		});
	}
}

void ObjectGrid::grow(const cv::Rect & cells)
{
	const cv::Rect current(origin_, cells_.size());
	const cv::Rect grown = cells_.empty() ? cells : (cells | current);
	if(grown == current)
	{
		return;
	}
	cv::Mat grownCells(grown.size(), CV_32FC3, cv::Scalar::all(0));
	if(!cells_.empty())
	{
		cells_.copyTo(grownCells(current - grown.tl()));
	}
	cells_ = grownCells;
	origin_ = grown.tl();
	minValues_[0] = origin_.x * cellSize_;
	minValues_[1] = origin_.y * cellSize_;
	maxValues_[0] = (origin_.x + cells_.cols) * cellSize_;
	maxValues_[1] = (origin_.y + cells_.rows) * cellSize_;
}

void ObjectGrid::vote(cv::Vec3f & cell, int label) const
{
	if(label == 0)
	{
		cell[0] = std::max(cell[0] + logOddsMiss_, logOddsClampingMin_);
	}
	else if(label == (int)cell[1] || cell[1] == 0.0f)
	{
		cell[0] = std::min(cell[0] + logOddsHit_, logOddsClampingMax_);
		cell[1] = label;
	}
	else
	{
		cell[0] = std::max(cell[0] - logOddsHit_, logOddsClampingMin_);
		cell[1] = cell[0] < logodds(occupancyThr_) ? label : cell[1];
	}
}

}
