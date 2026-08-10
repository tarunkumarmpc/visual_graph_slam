#ifndef VISUAL_GRAPH_SLAM_SENSOR_CAMERA_MODEL_HPP
#define VISUAL_GRAPH_SLAM_SENSOR_CAMERA_MODEL_HPP

#include <memory>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

namespace slam::sensor {

struct CameraIntrinsics {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
};

class ICameraModel {
public:
    virtual ~ICameraModel() = default;

    virtual std::string name() const = 0;
    virtual Eigen::Vector2d project(const Eigen::Vector3d& point_c) const = 0;
    virtual Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const = 0;
};

class PinholeCameraModel final : public ICameraModel {
public:
    explicit PinholeCameraModel(CameraIntrinsics intrinsics)
        : intrinsics_(intrinsics) {}

    std::string name() const override {
        return "pinhole";
    }

    Eigen::Vector2d project(const Eigen::Vector3d& point_c) const override {
        const double z = point_c.z();
        if (z <= 1e-9) {
            return Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
        }
        return {
            (intrinsics_.fx * point_c.x() / z) + intrinsics_.cx,
            (intrinsics_.fy * point_c.y() / z) + intrinsics_.cy
        };
    }

    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const override {
        return {
            ((pixel.x() - intrinsics_.cx) * depth) / intrinsics_.fx,
            ((pixel.y() - intrinsics_.cy) * depth) / intrinsics_.fy,
            depth
        };
    }

private:
    CameraIntrinsics intrinsics_;
};

class FisheyeCameraModel final : public ICameraModel {
public:
    FisheyeCameraModel(CameraIntrinsics intrinsics, Eigen::Vector4d distortion)
        : intrinsics_(intrinsics), distortion_(distortion) {}

    std::string name() const override {
        return "fisheye";
    }

    Eigen::Vector2d project(const Eigen::Vector3d& point_c) const override {
        // Kannala-Brandt-style radial mapping (lightweight approximation).
        const double x = point_c.x() / point_c.z();
        const double y = point_c.y() / point_c.z();
        const double r = std::sqrt(x * x + y * y);
        if (r < 1e-12) {
            return {intrinsics_.cx, intrinsics_.cy};
        }

        const double theta = std::atan(r);
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;
        const double theta_d = theta * (1.0 + distortion_[0] * theta2 + distortion_[1] * theta4 +
                                        distortion_[2] * theta6 + distortion_[3] * theta8);

        const double scale = theta_d / r;
        const double xd = x * scale;
        const double yd = y * scale;

        return {
            intrinsics_.fx * xd + intrinsics_.cx,
            intrinsics_.fy * yd + intrinsics_.cy
        };
    }

    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const override {
        // Undo focal length and principal point to get distorted normalised coords
        const double xd = (pixel.x() - intrinsics_.cx) / intrinsics_.fx;
        const double yd = (pixel.y() - intrinsics_.cy) / intrinsics_.fy;
        const double r_d = std::sqrt(xd * xd + yd * yd);

        if (r_d < 1e-10) {
            // Ray along optical axis
            return {0.0, 0.0, depth};
        }

        // Newton-Raphson inversion of the Kannala-Brandt polynomial:
        //   theta_d = theta * (1 + k1*theta² + k2*theta⁴ + k3*theta⁶ + k4*theta⁸)
        // We solve for theta given theta_d = r_d.
        // Initial guess: theta = atan(r_d)  (valid for wide angles)
        double theta = std::atan(r_d);
        for (int iter = 0; iter < 10; ++iter) {
            const double t2 = theta * theta;
            const double t4 = t2 * t2;
            const double t6 = t4 * t2;
            const double t8 = t4 * t4;
            const double f  = theta * (1.0 + distortion_[0] * t2 +
                                             distortion_[1] * t4 +
                                             distortion_[2] * t6 +
                                             distortion_[3] * t8) - r_d;
            const double df = 1.0 + 3.0 * distortion_[0] * t2 +
                                    5.0 * distortion_[1] * t4 +
                                    7.0 * distortion_[2] * t6 +
                                    9.0 * distortion_[3] * t8;
            const double step = f / std::max(df, 1e-12);
            theta -= step;
            if (std::abs(step) < 1e-9) { break; }
        }

        // Recover undistorted ray direction (normalised)
        const double r_u = std::tan(theta);       // = r in pinhole equivalent
        const double scale = (r_d < 1e-9) ? 1.0 : r_u / r_d;
        const double xu = xd * scale;
        const double yu = yd * scale;

        // Scale by depth: z = depth / ||[xu,yu,1]||₂  (for metric point)
        const double ray_norm = std::sqrt(xu * xu + yu * yu + 1.0);
        const double z = depth / ray_norm;
        return {xu * z, yu * z, z};
    }

private:
    CameraIntrinsics intrinsics_;
    Eigen::Vector4d distortion_;
};

inline std::unique_ptr<ICameraModel> createCameraModel(const std::string& model,
                                                        const CameraIntrinsics& intrinsics,
                                                        const Eigen::Vector4d& distortion = Eigen::Vector4d::Zero()) {
    if (model == "pinhole") {
        return std::make_unique<PinholeCameraModel>(intrinsics);
    }
    if (model == "fisheye") {
        return std::make_unique<FisheyeCameraModel>(intrinsics, distortion);
    }
    throw std::invalid_argument("Unsupported camera model: " + model);
}

}  // namespace slam::sensor

#endif  // VISUAL_GRAPH_SLAM_SENSOR_CAMERA_MODEL_HPP
