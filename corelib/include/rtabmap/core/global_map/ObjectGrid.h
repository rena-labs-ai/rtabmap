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

#ifndef CORELIB_SRC_OBJECTGRID_H_
#define CORELIB_SRC_OBJECTGRID_H_

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <rtabmap/core/GlobalMap.h>

namespace rtabmap {

/**
 * @class ObjectGrid
 * @brief 2D map of one labeler's labels, assembled from the nodes' label rasters
 * (@ref LocalGridMaker::createLabelRaster(), @ref LabelRaster::cells()) at their poses.
 *
 * In the cache, a node's ground cells are cells seen with nothing there (label 0, misses)
 * and its obstacle cells its labels, `CV_32FC3` (x, y, label). A cell holds one label and the
 * log-odds that it is there, `GridGlobal/Prob*`: a hit of that label adds `ProbHit`, a
 * miss adds `ProbMiss`, a hit of another label takes `ProbHit` away and, once the odds
 * are under `GridGlobal/OccupancyThr`, gives the cell to that label. So the label with
 * the most evidence holds a cell. A node votes once per cell.
 */
class RTABMAP_CORE_EXPORT ObjectGrid : public GlobalMap
{
public:
	ObjectGrid(const LocalGridCache * cache, const ParametersMap & parameters = ParametersMap());

	virtual void clear();

	/**
	 * @param[out] xMin world x of the map's origin, in meters
	 * @param[out] yMin world y of the map's origin, in meters
	 * @return `CV_8SC1`: -1 never seen, 0 seen with no object, otherwise the label of the
	 * object there. Empty if nothing is assembled.
	 */
	cv::Mat getMap(float & xMin, float & yMin) const;
	/**
	 * @return `CV_32SC1` on getMap()'s cells: the newest node that voted on each cell (0: none).
	 */
	cv::Mat getLastNodeMap(float & xMin, float & yMin) const;

protected:
	virtual void assemble(const std::list<std::pair<int, Transform> > & newPoses);

private:
	void grow(const cv::Rect & cells);
	void vote(cv::Vec3f & cell, int label) const;

	cv::Mat cells_; ///< `CV_32FC3` per cell: log-odds, label (0 none), last node that voted
	cv::Point origin_; ///< Index of cells_'s first cell, in cells from the world origin
};

}

#endif /* CORELIB_SRC_OBJECTGRID_H_ */
