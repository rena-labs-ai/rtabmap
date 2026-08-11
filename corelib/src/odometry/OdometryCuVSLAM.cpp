/*
Copyright (c) 2025 Felix Toft
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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "rtabmap/core/odometry/OdometryCuVSLAM.h"

#include "rtabmap/core/OdometryInfo.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UTimer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#ifdef RTABMAP_CUVSLAM
#include "rtabmap/core/CameraModel.h"
#include "rtabmap/core/IMU.h"
#include "rtabmap/core/SensorData.h"
#include "rtabmap/core/StereoCameraModel.h"
#include "rtabmap/core/Transform.h"

#include <Eigen/Eigenvalues>
#include <cuvslam/cuvslam2.h>
#include <opencv2/imgproc.hpp>
#endif

namespace rtabmap {

class OdometryCuVSLAM::Impl
{
public:
#ifdef RTABMAP_CUVSLAM
	std::unique_ptr<cuvslam::Odometry> odometry;
	cuvslam::Rig rig;
	Transform previousPose = Transform::getIdentity();
	int multicamMode = 0;
	int consecutiveTrackingFailures = 0;
	int64_t lastTimestampNs = std::numeric_limits<int64_t>::min();
	bool trackingStarted = false;
	bool continuityLost = false;

	bool imuFusion = Parameters::defaultOdomCuVSLAMImuFusion();
	float imuGyroNoiseDensity = Parameters::defaultOdomCuVSLAMImuGyroNoiseDensity();
	float imuGyroRandomWalk = Parameters::defaultOdomCuVSLAMImuGyroRandomWalk();
	float imuAccelNoiseDensity = Parameters::defaultOdomCuVSLAMImuAccelNoiseDensity();
	float imuAccelRandomWalk = Parameters::defaultOdomCuVSLAMImuAccelRandomWalk();
	float imuFrequency = Parameters::defaultOdomCuVSLAMImuFrequency();
	bool imuCalibrationKnown = false;
	cuvslam::ImuCalibration imuCalibration{};
	std::vector<cuvslam::ImuMeasurement> pendingImu;
	int64_t lastImuTimestampNs = std::numeric_limits<int64_t>::min();
	bool imuIgnoredWarned = false;
#endif
};

#ifdef RTABMAP_CUVSLAM

namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

constexpr int kSupportedCuVSLAMMajorVersion = 17;
constexpr int kMaxConsecutiveTrackingFailures = 10;

cv::Mat highCovariance()
{
	return cv::Mat::eye(6, 6, CV_64FC1) * 9999.0;
}

void setFailureInfo(OdometryInfo * info, UTimer & timer)
{
	if(info)
	{
		info->type = Odometry::kTypeF2F;
		info->reg.covariance = highCovariance();
		info->timeEstimation = timer.ticks();
	}
}

cuvslam::Pose toCuVSLAMPose(const Transform & transform)
{
	const Eigen::Quaternionf quaternion = transform.getQuaternionf();
	cuvslam::Pose pose;
	pose.rotation = {quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
	pose.translation = {transform.x(), transform.y(), transform.z()};
	return pose;
}

Transform fromCuVSLAMPose(const cuvslam::Pose & pose)
{
	return Transform(pose.translation[0], pose.translation[1], pose.translation[2], pose.rotation[0], pose.rotation[1],
	                 pose.rotation[2], pose.rotation[3]);
}

bool toCuVSLAMImuMeasurement(const IMU & imu, double stamp, cuvslam::ImuMeasurement * measurement)
{
	const long double nanoseconds = static_cast<long double>(stamp) * 1000000000.0L;
	if(!std::isfinite(stamp) || nanoseconds < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
	   nanoseconds > static_cast<long double>(std::numeric_limits<int64_t>::max()))
	{
		return false;
	}
	measurement->timestamp_ns = static_cast<int64_t>(std::llround(nanoseconds));
	for(int i = 0; i < 3; ++i)
	{
		measurement->linear_accelerations[i] = static_cast<float>(imu.linearAcceleration()[i]);
		measurement->angular_velocities[i] = static_cast<float>(imu.angularVelocity()[i]);
	}
	return true;
}

cuvslam::Odometry::MulticameraMode toCuVSLAMMulticameraMode(int mode)
{
	// Preserve the values of OdomCuVSLAM/MulticamMode used by the legacy API.
	switch(mode)
	{
	case 0:
		return cuvslam::Odometry::MulticameraMode::Moderate;
	case 1:
		return cuvslam::Odometry::MulticameraMode::Performance;
	case 2:
		return cuvslam::Odometry::MulticameraMode::Precision;
	default:
		return cuvslam::Odometry::MulticameraMode::Moderate;
	}
}

bool configureCamera(const CameraModel & model, const Transform & rigFromCamera, unsigned int cameraIndex,
                     bool rawImages, cuvslam::Camera * camera)
{
	if(!model.isValidForProjection() || model.imageWidth() <= 0 || model.imageHeight() <= 0)
	{
		UERROR("Invalid camera model %u for cuVSLAM initialization", cameraIndex);
		return false;
	}
	if(rigFromCamera.isNull() || !rigFromCamera.isInvertible())
	{
		UERROR("Invalid camera transform %u for cuVSLAM initialization", cameraIndex);
		return false;
	}

	camera->size = {model.imageWidth(), model.imageHeight()};
	camera->principal = {static_cast<float>(model.cx()), static_cast<float>(model.cy())};
	camera->focal = {static_cast<float>(model.fx()), static_cast<float>(model.fy())};
	camera->rig_from_camera = toCuVSLAMPose(rigFromCamera);
	if(rawImages)
	{
		// OpenCV/ROS D order (k1,k2,p1,p2,k3,k4,k5,k6) is the tail of the
		// cuVSLAM Polynomial model; absent trailing coefficients are zero.
		const cv::Mat distortion = model.D();
		camera->distortion.model = cuvslam::Distortion::Model::Polynomial;
		camera->distortion.parameters.assign(8, 0.0f);
		for(int i = 0; i < 8 && i < static_cast<int>(distortion.total()); ++i)
		{
			camera->distortion.parameters[i] = static_cast<float>(distortion.at<double>(i));
		}
	}
	else
	{
		camera->distortion.model = cuvslam::Distortion::Model::Pinhole;
		camera->distortion.parameters.clear();
	}
	return true;
}

bool approximatelyEqual(float first, float second, float absoluteTolerance, float relativeTolerance = 1e-6f)
{
	const float scale = std::max(std::abs(first), std::abs(second));
	return std::abs(first - second) <= std::max(absoluteTolerance, scale * relativeTolerance);
}

bool isRectifiedStereoPair(const StereoCameraModel & stereo, size_t stereoIndex)
{
	const CameraModel & left = stereo.left();
	const CameraModel & right = stereo.right();
	if(left.imageSize() != right.imageSize() ||
	   !approximatelyEqual(static_cast<float>(left.fx()), static_cast<float>(right.fx()), 1e-3f, 1e-5f) ||
	   !approximatelyEqual(static_cast<float>(left.fy()), static_cast<float>(right.fy()), 1e-3f, 1e-5f) ||
	   !approximatelyEqual(static_cast<float>(left.cy()), static_cast<float>(right.cy()), 1e-3f, 1e-5f))
	{
		UERROR("Stereo camera model %zu is not a horizontal rectified pair", stereoIndex);
		return false;
	}
	const cv::Mat leftDistortion = left.D();
	const cv::Mat rightDistortion = right.D();
	if((!leftDistortion.empty() && cv::norm(leftDistortion, cv::NORM_INF) > 1e-8) ||
	   (!rightDistortion.empty() && cv::norm(rightDistortion, cv::NORM_INF) > 1e-8))
	{
		UERROR("Stereo camera model %zu has distortion but cuVSLAM expects rectified images", stereoIndex);
		return false;
	}
	return true;
}

// Pose of the right camera in the left camera frame. Rectified pairs share the
// rectified frame, so the right camera is a pure baseline translation. A raw
// pair does not: both rectification rotations map into that shared frame, so
// undo the left one and re-apply the right one to get back to raw left.
Transform leftFromRight(const StereoCameraModel & stereo, bool rawImages, size_t stereoIndex)
{
	const double baseline = stereo.baseline();
	if(!rawImages)
	{
		return Transform(1.0f, 0.0f, 0.0f, static_cast<float>(baseline), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                 0.0f);
	}

	const cv::Mat leftRectification = stereo.left().R();
	const cv::Mat rightRectification = stereo.right().R();
	if(leftRectification.rows != 3 || leftRectification.cols != 3 || rightRectification.rows != 3 ||
	   rightRectification.cols != 3)
	{
		UERROR("Stereo camera model %zu carries raw images but no rectification rotations; cannot place the right "
		       "camera in the rig",
		       stereoIndex);
		return Transform();
	}

	const cv::Mat leftRectificationInverse = leftRectification.t();
	const cv::Mat rotation = leftRectificationInverse * rightRectification;
	const cv::Mat translation = leftRectificationInverse * (cv::Mat_<double>(3, 1) << baseline, 0.0, 0.0);
	return Transform(rotation.at<double>(0, 0), rotation.at<double>(0, 1), rotation.at<double>(0, 2),
	                 translation.at<double>(0), rotation.at<double>(1, 0), rotation.at<double>(1, 1),
	                 rotation.at<double>(1, 2), translation.at<double>(1), rotation.at<double>(2, 0),
	                 rotation.at<double>(2, 1), rotation.at<double>(2, 2), translation.at<double>(2));
}

bool createRig(const SensorData & data, const cuvslam::ImuCalibration * imu, bool rawImages, cuvslam::Rig * rig)
{
	const std::vector<StereoCameraModel> & models = data.stereoCameraModels();
	if(models.empty())
	{
		UERROR("cuVSLAM odometry requires at least one stereo camera model");
		return false;
	}
	if(models.size() * 2 > 32)
	{
		UERROR("cuVSLAM supports at most 32 cameras, got %zu", models.size() * 2);
		return false;
	}

	rig->cameras.clear();
	rig->cameras.reserve(models.size() * 2);
	rig->imus.clear();
	if(imu)
	{
		rig->imus.push_back(*imu);
	}
	for(size_t i = 0; i < models.size(); ++i)
	{
		const StereoCameraModel & stereo = models[i];
		if(!stereo.isValidForProjection() || !std::isfinite(stereo.baseline()))
		{
			UERROR("Invalid stereo camera model %zu for cuVSLAM initialization", i);
			return false;
		}
		if(!rawImages && !isRectifiedStereoPair(stereo, i))
		{
			return false;
		}

		const Transform & rigFromLeft = stereo.localTransform();
		const Transform rightInLeft = leftFromRight(stereo, rawImages, i);
		if(rightInLeft.isNull())
		{
			return false;
		}

		cuvslam::Camera leftCamera;
		if(!configureCamera(stereo.left(), rigFromLeft, static_cast<unsigned int>(i * 2), rawImages, &leftCamera))
		{
			return false;
		}
		rig->cameras.push_back(std::move(leftCamera));

		cuvslam::Camera rightCamera;
		if(!configureCamera(stereo.right(), rigFromLeft * rightInLeft, static_cast<unsigned int>(i * 2 + 1),
		                    rawImages, &rightCamera))
		{
			return false;
		}
		rig->cameras.push_back(std::move(rightCamera));
	}

	for(size_t i = 1; i < rig->cameras.size(); ++i)
	{
		if(rig->cameras[i].size != rig->cameras[0].size)
		{
			UERROR("cuVSLAM requires all camera image sizes to match");
			return false;
		}
	}
	return true;
}

bool posesMatch(const cuvslam::Pose & first, const cuvslam::Pose & second)
{
	float quaternionDot = 0.0f;
	for(size_t i = 0; i < first.rotation.size(); ++i)
	{
		quaternionDot += first.rotation[i] * second.rotation[i];
	}
	const float quaternionSign = quaternionDot < 0.0f ? -1.0f : 1.0f;
	for(size_t i = 0; i < first.rotation.size(); ++i)
	{
		if(!approximatelyEqual(first.rotation[i], quaternionSign * second.rotation[i], 1e-6f))
		{
			return false;
		}
	}
	for(size_t i = 0; i < first.translation.size(); ++i)
	{
		if(!approximatelyEqual(first.translation[i], second.translation[i], 1e-6f))
		{
			return false;
		}
	}
	return true;
}

bool camerasMatch(const cuvslam::Camera & first, const cuvslam::Camera & second)
{
	if(first.size != second.size || first.distortion.model != second.distortion.model ||
	   first.distortion.parameters.size() != second.distortion.parameters.size() ||
	   first.border_top != second.border_top || first.border_bottom != second.border_bottom ||
	   first.border_left != second.border_left || first.border_right != second.border_right ||
	   !posesMatch(first.rig_from_camera, second.rig_from_camera))
	{
		return false;
	}
	for(size_t i = 0; i < first.principal.size(); ++i)
	{
		if(!approximatelyEqual(first.principal[i], second.principal[i], 1e-4f) ||
		   !approximatelyEqual(first.focal[i], second.focal[i], 1e-4f))
		{
			return false;
		}
	}
	for(size_t i = 0; i < first.distortion.parameters.size(); ++i)
	{
		if(!approximatelyEqual(first.distortion.parameters[i], second.distortion.parameters[i], 1e-8f))
		{
			return false;
		}
	}
	return true;
}

bool rigsMatch(const cuvslam::Rig & first, const cuvslam::Rig & second)
{
	if(first.cameras.size() != second.cameras.size() || first.imus.size() != second.imus.size())
	{
		return false;
	}
	for(size_t i = 0; i < first.cameras.size(); ++i)
	{
		if(!camerasMatch(first.cameras[i], second.cameras[i]))
		{
			return false;
		}
	}
	return true;
}

bool createTracker(const SensorData & data, int multicamMode, const cuvslam::ImuCalibration * imu, bool rawImages,
                   cuvslam::Rig * rig, std::unique_ptr<cuvslam::Odometry> * odometry)
{
	cuvslam::Rig candidateRig;
	if(!createRig(data, imu, rawImages, &candidateRig))
	{
		return false;
	}

	cuvslam::Odometry::Config configuration = cuvslam::Odometry::GetDefaultConfig();
	if(imu)
	{
		// Inertial = single stereo + single IMU (fusion is not supported in
		// Multicamera mode). The IMU drives pose prediction, so the internal
		// motion model is off per the cuVSLAM guidance.
		configuration.odometry_mode = cuvslam::Odometry::OdometryMode::Inertial;
		configuration.use_motion_model = false;
	}
	else
	{
		configuration.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
		configuration.use_motion_model = true;
	}
	configuration.multicam_mode = toCuVSLAMMulticameraMode(multicamMode);
	configuration.rectified_stereo_camera = !rawImages;
	// Observation export feeds the per-frame quality (tracked feature count)
	// reported through OdometryInfo.
	configuration.enable_observations_export = true;
	configuration.enable_landmarks_export = false;
	configuration.enable_final_landmarks_export = false;
	// cuVSLAM observations do not include descriptors. Let the RTAB-Map backend
	// extract its own loop-closure features instead of paying the export cost.

	try
	{
		int major = 0;
		int minor = 0;
		int patch = 0;
		cuvslam::GetVersion(&major, &minor, &patch);
		if(major != kSupportedCuVSLAMMajorVersion)
		{
			UERROR("Unsupported cuVSLAM library version %d.%d.%d (expected %d.x)", major, minor, patch,
			       kSupportedCuVSLAMMajorVersion);
			return false;
		}
		cuvslam::SetVerbosity(0);
		std::unique_ptr<cuvslam::Odometry> candidateOdometry(new cuvslam::Odometry(candidateRig, configuration));
		*rig = std::move(candidateRig);
		*odometry = std::move(candidateOdometry);
		UINFO("Using cuVSLAM %d.%d.%d", major, minor, patch);
	} catch(const std::exception & exception)
	{
		UERROR("Failed to initialize cuVSLAM: %s", exception.what());
		return false;
	}
	return true;
}

bool timestampToNanoseconds(double stamp, int64_t * timestampNs)
{
	if(!std::isfinite(stamp))
	{
		UERROR("Invalid non-finite timestamp for cuVSLAM");
		return false;
	}

	const long double nanoseconds = static_cast<long double>(stamp) * 1000000000.0L;
	if(nanoseconds < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
	   nanoseconds > static_cast<long double>(std::numeric_limits<int64_t>::max()))
	{
		UERROR("Timestamp %.9f is outside the cuVSLAM nanosecond range", stamp);
		return false;
	}
	*timestampNs = static_cast<int64_t>(std::llround(nanoseconds));
	return true;
}

bool addImage(const cv::Mat & stitchedImage, const cv::Rect & region, uint32_t cameraIndex, int64_t timestampNs,
              std::vector<cv::Mat> * buffers, cuvslam::Odometry::ImageSet * images)
{
	if(region.x < 0 || region.y < 0 || region.width <= 0 || region.height <= 0 ||
	   region.x + region.width > stitchedImage.cols || region.y + region.height > stitchedImage.rows)
	{
		UERROR(
		    "Camera %u image region (%d,%d %dx%d) is outside the stitched image "
		    "(%dx%d)",
		    cameraIndex, region.x, region.y, region.width, region.height, stitchedImage.cols, stitchedImage.rows);
		return false;
	}

	const cv::Mat imageRegion = stitchedImage(region);
	cuvslam::ImageData::Encoding encoding;
	cv::Mat buffer;
	if(imageRegion.channels() == 1)
	{
		buffer = imageRegion.clone();
		encoding = cuvslam::ImageData::Encoding::MONO;
	}
	else if(imageRegion.channels() == 3)
	{
		cv::cvtColor(imageRegion, buffer, cv::COLOR_BGR2RGB);
		encoding = cuvslam::ImageData::Encoding::RGB;
	}
	else
	{
		UERROR("Camera %u has unsupported image channel count %d", cameraIndex, imageRegion.channels());
		return false;
	}

	if(buffer.step > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
	{
		UERROR("Camera %u image pitch is too large for cuVSLAM", cameraIndex);
		return false;
	}

	buffers->push_back(std::move(buffer));
	const cv::Mat & storedImage = buffers->back();
	cuvslam::Image image{};
	image.pixels = storedImage.data;
	image.width = storedImage.cols;
	image.height = storedImage.rows;
	image.pitch = static_cast<int32_t>(storedImage.step);
	image.encoding = encoding;
	image.data_type = cuvslam::ImageData::DataType::UINT8;
	image.is_gpu_mem = false;
	image.timestamp_ns = timestampNs;
	image.camera_index = cameraIndex;
	images->push_back(image);
	return true;
}

bool prepareImages(const SensorData & data, int64_t timestampNs, std::vector<cv::Mat> * buffers,
                   cuvslam::Odometry::ImageSet * images)
{
	const cv::Mat & leftImage = data.imageRaw();
	const cv::Mat & rightImage = data.rightRaw();
	if(leftImage.empty() || rightImage.empty())
	{
		UERROR("cuVSLAM odometry requires both left and right images");
		return false;
	}
	if(leftImage.depth() != CV_8U || rightImage.depth() != CV_8U)
	{
		UERROR("cuVSLAM requires 8-bit images, got left depth %d and right depth %d", leftImage.depth(),
		       rightImage.depth());
		return false;
	}
	if((leftImage.channels() != 1 && leftImage.channels() != 3) ||
	   (rightImage.channels() != 1 && rightImage.channels() != 3))
	{
		UERROR(
		    "cuVSLAM supports mono or RGB images, got %d left channels and %d "
		    "right channels",
		    leftImage.channels(), rightImage.channels());
		return false;
	}

	const std::vector<StereoCameraModel> & models = data.stereoCameraModels();
	buffers->clear();
	images->clear();
	buffers->reserve(models.size() * 2);
	images->reserve(models.size() * 2);

	int leftOffset = 0;
	int rightOffset = 0;
	for(size_t i = 0; i < models.size(); ++i)
	{
		const CameraModel & leftModel = models[i].left();
		const CameraModel & rightModel = models[i].right();
		if(leftModel.imageHeight() != leftImage.rows || rightModel.imageHeight() != rightImage.rows)
		{
			UERROR("Stereo camera model %zu image heights do not match the input", i);
			return false;
		}
		if(!addImage(leftImage, cv::Rect(leftOffset, 0, leftModel.imageWidth(), leftModel.imageHeight()),
		             static_cast<uint32_t>(i * 2), timestampNs, buffers, images))
		{
			return false;
		}
		if(!addImage(rightImage, cv::Rect(rightOffset, 0, rightModel.imageWidth(), rightModel.imageHeight()),
		             static_cast<uint32_t>(i * 2 + 1), timestampNs, buffers, images))
		{
			return false;
		}
		leftOffset += leftModel.imageWidth();
		rightOffset += rightModel.imageWidth();
	}
	if(leftOffset != leftImage.cols || rightOffset != rightImage.cols)
	{
		UERROR(
		    "Stereo camera model widths (%d left, %d right) do not match the "
		    "stitched images (%d left, %d right)",
		    leftOffset, rightOffset, leftImage.cols, rightImage.cols);
		return false;
	}
	return true;
}

double normalizeAngle(double angle)
{
	return std::atan2(std::sin(angle), std::cos(angle));
}

Vector6d transformToVector(const Transform & transform)
{
	float x;
	float y;
	float z;
	float roll;
	float pitch;
	float yaw;
	transform.getTranslationAndEulerAngles(x, y, z, roll, pitch, yaw);
	Vector6d vector;
	vector << x, y, z, roll, pitch, yaw;
	return vector;
}

Transform vectorToTransform(const Vector6d & vector)
{
	return Transform(static_cast<float>(vector[0]), static_cast<float>(vector[1]), static_cast<float>(vector[2]),
	                 static_cast<float>(vector[3]), static_cast<float>(vector[4]), static_cast<float>(vector[5]));
}

Vector6d relativePoseVector(const Transform & previousPose, const Vector6d & currentPoseVector)
{
	return transformToVector(previousPose.inverse() * vectorToTransform(currentPoseVector));
}

cv::Mat relativeCovariance(const Transform & previousPose, const Transform & currentPose,
                           const cuvslam::PoseCovariance & absoluteCovariance)
{
	Matrix6d covariance;
	for(int row = 0; row < 6; ++row)
	{
		for(int column = 0; column < 6; ++column)
		{
			covariance(row, column) = absoluteCovariance[row * 6 + column];
		}
	}
	if(!covariance.allFinite())
	{
		UWARN("cuVSLAM returned a non-finite covariance");
		return highCovariance();
	}
	for(int i = 0; i < 6; ++i)
	{
		if(covariance(i, i) < 0.0)
		{
			UWARN("cuVSLAM returned a negative covariance diagonal");
			return highCovariance();
		}
	}
	covariance = 0.5 * (covariance + covariance.transpose());

	Eigen::SelfAdjointEigenSolver<Matrix6d> absoluteSolver(covariance);
	if(absoluteSolver.info() != Eigen::Success)
	{
		UWARN("Could not validate the cuVSLAM covariance");
		return highCovariance();
	}
	const double absoluteTolerance = std::max(1e-12, 1e-6 * absoluteSolver.eigenvalues().cwiseAbs().maxCoeff());
	if(absoluteSolver.eigenvalues().minCoeff() < -absoluteTolerance)
	{
		UWARN("cuVSLAM returned an indefinite covariance");
		return highCovariance();
	}
	covariance = absoluteSolver.eigenvectors() * absoluteSolver.eigenvalues().cwiseMax(0.0).asDiagonal() *
	             absoluteSolver.eigenvectors().transpose();

	// cuVSLAM reports covariance for its absolute xyz/RPY pose. RTAB-Map
	// consumes covariance for the incremental pose. Cross-covariance between
	// frames is not exposed, so propagate the current covariance while treating
	// the previous pose as fixed.
	const Vector6d currentPoseVector = transformToVector(currentPose);
	Matrix6d jacobian;
	for(int column = 0; column < 6; ++column)
	{
		const double epsilon = column < 3 ? 1e-4 : 1e-5;
		Vector6d plus = currentPoseVector;
		Vector6d minus = currentPoseVector;
		plus[column] += epsilon;
		minus[column] -= epsilon;
		const Vector6d relativePlus = relativePoseVector(previousPose, plus);
		const Vector6d relativeMinus = relativePoseVector(previousPose, minus);
		for(int row = 0; row < 6; ++row)
		{
			double difference = relativePlus[row] - relativeMinus[row];
			if(row >= 3)
			{
				difference = normalizeAngle(difference);
			}
			jacobian(row, column) = difference / (2.0 * epsilon);
		}
	}

	Matrix6d propagated = jacobian * covariance * jacobian.transpose();
	propagated = 0.5 * (propagated + propagated.transpose());
	if(!propagated.allFinite())
	{
		UWARN("Could not propagate the cuVSLAM covariance");
		return highCovariance();
	}

	Eigen::SelfAdjointEigenSolver<Matrix6d> solver(propagated);
	if(solver.info() != Eigen::Success)
	{
		UWARN(
		    "Could not make the propagated cuVSLAM covariance positive "
		    "semidefinite");
		return highCovariance();
	}
	const double propagatedTolerance = std::max(1e-12, 1e-9 * solver.eigenvalues().cwiseAbs().maxCoeff());
	if(solver.eigenvalues().minCoeff() < -propagatedTolerance)
	{
		UWARN("Propagated cuVSLAM covariance is indefinite");
		return highCovariance();
	}
	const Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues().cwiseMax(1e-9);
	propagated = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();

	cv::Mat output(6, 6, CV_64FC1);
	for(int row = 0; row < 6; ++row)
	{
		for(int column = 0; column < 6; ++column)
		{
			output.at<double>(row, column) = propagated(row, column);
		}
	}
	return output;
}

}  // namespace

#endif  // RTABMAP_CUVSLAM

OdometryCuVSLAM::OdometryCuVSLAM(const ParametersMap & parameters) : Odometry(parameters), impl_(new Impl)
{
#ifdef RTABMAP_CUVSLAM
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMMulticamMode(), impl_->multicamMode);
	if(impl_->multicamMode < 0 || impl_->multicamMode > 2)
	{
		UWARN("Invalid %s=%d, using 0 (moderate)", Parameters::kOdomCuVSLAMMulticamMode().c_str(), impl_->multicamMode);
		impl_->multicamMode = 0;
	}
	UINFO("%s=%d", Parameters::kOdomCuVSLAMMulticamMode().c_str(), impl_->multicamMode);
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMImuFusion(), impl_->imuFusion);
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMImuGyroNoiseDensity(), impl_->imuGyroNoiseDensity);
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMImuGyroRandomWalk(), impl_->imuGyroRandomWalk);
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMImuAccelNoiseDensity(), impl_->imuAccelNoiseDensity);
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMImuAccelRandomWalk(), impl_->imuAccelRandomWalk);
	Parameters::parse(parameters, Parameters::kOdomCuVSLAMImuFrequency(), impl_->imuFrequency);
	UINFO("%s=%s", Parameters::kOdomCuVSLAMImuFusion().c_str(), impl_->imuFusion ? "true" : "false");
#endif
}

bool OdometryCuVSLAM::canProcessAsyncIMU() const
{
#ifdef RTABMAP_CUVSLAM
	return impl_->imuFusion;
#else
	return false;
#endif
}

OdometryCuVSLAM::~OdometryCuVSLAM() = default;

void OdometryCuVSLAM::reset(const Transform & initialPose)
{
#ifdef RTABMAP_CUVSLAM
	const bool continuityLost = impl_->continuityLost || impl_->trackingStarted;
#endif
	Odometry::reset(initialPose);
	cleanupCuVSLAMResources();
#ifdef RTABMAP_CUVSLAM
	impl_->continuityLost = continuityLost;
#endif
}

void OdometryCuVSLAM::cleanupCuVSLAMResources()
{
#ifdef RTABMAP_CUVSLAM
	impl_->odometry.reset();
	impl_->rig = cuvslam::Rig{};
	impl_->previousPose = Transform::getIdentity();
	impl_->consecutiveTrackingFailures = 0;
	impl_->lastTimestampNs = std::numeric_limits<int64_t>::min();
	impl_->trackingStarted = false;
#endif
}

Transform OdometryCuVSLAM::computeTransform(SensorData & data, const Transform & guess, OdometryInfo * info)
{
#ifdef RTABMAP_CUVSLAM
	UTimer timer;
	(void)guess;  // cuVSLAM 17 has no external pose-prediction input.

	if(impl_->imuFusion && !data.imu().empty())
	{
		if(!impl_->imuCalibrationKnown)
		{
			const Transform & rigFromImu = data.imu().localTransform();
			if(rigFromImu.isNull())
			{
				UERROR("IMU sample has no local transform; cannot calibrate cuVSLAM IMU");
			}
			else
			{
				impl_->imuCalibration.rig_from_imu = toCuVSLAMPose(rigFromImu);
				impl_->imuCalibration.gyroscope_noise_density = impl_->imuGyroNoiseDensity;
				impl_->imuCalibration.gyroscope_random_walk = impl_->imuGyroRandomWalk;
				impl_->imuCalibration.accelerometer_noise_density = impl_->imuAccelNoiseDensity;
				impl_->imuCalibration.accelerometer_random_walk = impl_->imuAccelRandomWalk;
				impl_->imuCalibration.frequency = impl_->imuFrequency;
				impl_->imuCalibrationKnown = true;
			}
		}
		// A live tracker without an IMU in its rig will never consume samples
		// (multi-camera mode) — don't accumulate them.
		const bool consumable = !impl_->odometry || !impl_->rig.imus.empty();
		cuvslam::ImuMeasurement measurement{};
		if(consumable && impl_->imuCalibrationKnown &&
		   toCuVSLAMImuMeasurement(data.imu(), data.stamp(), &measurement) &&
		   measurement.timestamp_ns > impl_->lastImuTimestampNs)
		{
			impl_->lastImuTimestampNs = measurement.timestamp_ns;
			impl_->pendingImu.push_back(measurement);
		}
		if(data.imageRaw().empty())
		{
			// Async IMU-only event; nothing to track.
			return Transform();
		}
	}

	if(data.imageRaw().empty() || data.rightRaw().empty())
	{
		UERROR("cuVSLAM odometry requires stereo images");
		setFailureInfo(info, timer);
		return Transform();
	}
	if(data.stereoCameraModels().empty())
	{
		UERROR("cuVSLAM odometry requires stereo camera models");
		setFailureInfo(info, timer);
		return Transform();
	}

	// IMU fusion is only supported for a single stereo camera (cuVSLAM
	// Inertial mode); multi-camera rigs keep the visual-only Multicamera mode.
	const bool wantImu =
	    impl_->imuFusion && impl_->imuCalibrationKnown && data.stereoCameraModels().size() == 1;
	if(impl_->imuFusion && impl_->imuCalibrationKnown && data.stereoCameraModels().size() > 1 &&
	   !impl_->imuIgnoredWarned)
	{
		UWARN("cuVSLAM IMU fusion supports a single stereo camera only; ignoring the IMU (multicamera mode)");
		impl_->imuIgnoredWarned = true;
	}

	// The rig is fixed at tracker creation. If the IMU showed up after a
	// visual-only tracker was created but tracking never engaged, recreate the
	// tracker with the IMU instead of silently staying visual-only.
	if(impl_->odometry && wantImu && impl_->rig.imus.empty() && !impl_->trackingStarted)
	{
		cleanupCuVSLAMResources();
	}

	// Odometry::process() leaves raw pairs untouched (canProcessRawImages), so
	// the models still carry lens distortion and per-camera rectification
	// rotations: cuVSLAM undistorts instead.
	const bool rawImages = !this->imagesAlreadyRectified();

	const cuvslam::ImuCalibration * imu = nullptr;
	if(!impl_->odometry)
	{
		imu = wantImu ? &impl_->imuCalibration : nullptr;
		if(!createTracker(data, impl_->multicamMode, imu, rawImages, &impl_->rig, &impl_->odometry))
		{
			setFailureInfo(info, timer);
			return Transform();
		}
	}
	else
	{
		imu = impl_->rig.imus.empty() ? nullptr : &impl_->imuCalibration;
		cuvslam::Rig currentRig;
		if(!createRig(data, imu, rawImages, &currentRig) || !rigsMatch(impl_->rig, currentRig))
		{
			UERROR("Stereo camera calibration changed; restarting cuVSLAM");
			impl_->continuityLost = impl_->continuityLost || impl_->trackingStarted;
			cleanupCuVSLAMResources();
			setFailureInfo(info, timer);
			return Transform();
		}
	}

	int64_t timestampNs = 0;
	if(!timestampToNanoseconds(data.stamp(), &timestampNs))
	{
		setFailureInfo(info, timer);
		return Transform();
	}
	if(timestampNs <= impl_->lastTimestampNs)
	{
		UERROR(
		    "cuVSLAM image timestamps must increase: current=%lld ns "
		    "previous=%lld ns",
		    static_cast<long long>(timestampNs), static_cast<long long>(impl_->lastTimestampNs));
		setFailureInfo(info, timer);
		return Transform();
	}

	std::vector<cv::Mat> imageBuffers;
	cuvslam::Odometry::ImageSet images;
	if(!prepareImages(data, timestampNs, &imageBuffers, &images))
	{
		setFailureInfo(info, timer);
		return Transform();
	}

	if(!impl_->rig.imus.empty() && !impl_->pendingImu.empty())
	{
		// Register buffered samples up to the frame stamp; Track() and
		// RegisterImuMeasurement() must be called in timestamp order.
		size_t registered = 0;
		try
		{
			while(registered < impl_->pendingImu.size() &&
			      impl_->pendingImu[registered].timestamp_ns <= timestampNs)
			{
				impl_->odometry->RegisterImuMeasurement(0, impl_->pendingImu[registered]);
				++registered;
			}
		} catch(const std::exception & exception)
		{
			UWARN("cuVSLAM rejected an IMU measurement: %s", exception.what());
			++registered;  // drop the offending sample
		}
		impl_->pendingImu.erase(impl_->pendingImu.begin(), impl_->pendingImu.begin() + registered);
	}

	cuvslam::PoseEstimate estimate{};
	try
	{
		estimate = impl_->odometry->Track(images);
	} catch(const std::exception & exception)
	{
		UERROR("cuVSLAM tracking failed: %s", exception.what());
		impl_->continuityLost = impl_->continuityLost || impl_->trackingStarted;
		cleanupCuVSLAMResources();
		setFailureInfo(info, timer);
		return Transform();
	}
	const double frameGapMs = impl_->lastTimestampNs > 0
		? double(timestampNs - impl_->lastTimestampNs) / 1e6
		: 0.0;
	impl_->lastTimestampNs = timestampNs;

	if(!estimate.world_from_rig)
	{
		if(impl_->trackingStarted)
		{
			impl_->continuityLost = true;
			++impl_->consecutiveTrackingFailures;
			if(impl_->consecutiveTrackingFailures >= kMaxConsecutiveTrackingFailures)
			{
				UWARN("cuVSLAM failed to recover after %d frames; restarting the visual frontend",
				      impl_->consecutiveTrackingFailures);
				cleanupCuVSLAMResources();
			}
			else
			{
				UWARN("cuVSLAM tracking update failed (%d/%d); keeping the frontend alive for recovery",
				      impl_->consecutiveTrackingFailures, kMaxConsecutiveTrackingFailures);
			}
		}
		else
		{
			UWARN("cuVSLAM has not initialized a pose yet");
		}
		setFailureInfo(info, timer);
		return Transform();
	}

	const cuvslam::PoseWithCovariance & poseWithCovariance = *estimate.world_from_rig;
	const Transform currentPose = fromCuVSLAMPose(poseWithCovariance.pose);
	if(currentPose.isNull() || !currentPose.isInvertible())
	{
		UERROR("cuVSLAM returned an invalid pose");
		impl_->continuityLost = impl_->continuityLost || impl_->trackingStarted;
		cleanupCuVSLAMResources();
		setFailureInfo(info, timer);
		return Transform();
	}

	const int failuresBridged = impl_->consecutiveTrackingFailures;
	const bool bridgedGap = impl_->continuityLost;
	uint32_t observationCount = 0;
	{
		cuvslam::Odometry::State state;
		impl_->odometry->GetState(state);
		observationCount = static_cast<uint32_t>(state.observations.size());
	}
	const Transform transform = impl_->previousPose.inverse() * currentPose;
	cv::Mat covariance;
	if(impl_->continuityLost)
	{
		UWARN("cuVSLAM recovered after an unobserved interval; reporting an unconstrained bridge");
		covariance = highCovariance();
		impl_->continuityLost = false;
	}
	else
	{
		covariance = relativeCovariance(impl_->previousPose, currentPose, poseWithCovariance.covariance_xyz_rpy);
	}
	impl_->previousPose = currentPose;
	impl_->consecutiveTrackingFailures = 0;
	impl_->trackingStarted = true;

	const Eigen::Quaternionf q = transform.getQuaternionf();
	const float rotDeg = 2.0f * std::acos(std::min(1.0f, std::fabs(q.w()))) * 180.0f / float(M_PI);
	UINFO("cuVSLAM dt=%.0fms |t|=%.3fm |r|=%.1fdeg obs=%u fails=%d bridge=%d",
	      frameGapMs, transform.getNorm(), rotDeg, observationCount,
	      failuresBridged, bridgedGap ? 1 : 0);

	if(info)
	{
		info->type = kTypeF2F;
		info->reg.covariance = covariance;
		info->reg.inliers = static_cast<int>(observationCount);
		info->features = static_cast<int>(observationCount);
		info->timeEstimation = timer.ticks();
	}
	return transform;
#else
	(void)data;
	(void)guess;
	(void)info;
	UERROR("cuVSLAM support not compiled in RTAB-Map");
	return Transform();
#endif
}

}  // namespace rtabmap
