#include "visual_graph_slam/plugins/frontend.hpp"
#include "visual_graph_slam/frontend/visual_odometry_frontend.hpp"
#include "visual_graph_slam/frontend/feature_pipeline.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <stdexcept>

namespace slam::plugins {

class FrontendSift : public slam::core::Frontend {
public:
    FrontendSift() = default;
    ~FrontendSift() override = default;

    void initialize(rclcpp::Node::SharedPtr node, const std::string& plugin_name) override {
        node_ = node;
        logger_ = node_->get_logger();

        std::string ns = plugin_name + ".";
        int max_features = node_->declare_parameter<int>(ns + "max_features", 0); // 0 means keep all
        int min_tracked_pts = node_->declare_parameter<int>(ns + "min_tracked_pts", 80);
        int reinit_threshold = node_->declare_parameter<int>(ns + "reinit_threshold", 40);
        int max_corners = node_->declare_parameter<int>(ns + "max_corners", 500);
        double min_flow_px = node_->declare_parameter<double>(ns + "min_flow_px", 1.5);
        
        std::vector<double> cam_mat_vec = node_->declare_parameter<std::vector<double>>(
            "camera_matrix", {1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0});
            
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        if (cam_mat_vec.size() == 9) {
            for (int i=0; i<3; ++i)
                for (int j=0; j<3; ++j)
                    camera_matrix.at<double>(i, j) = cam_mat_vec[i*3 + j];
        }

        RCLCPP_INFO(logger_, "Initializing FrontendSift with max_features=%d", max_features);

        auto pipeline = std::make_unique<slam::frontend::SiftFeaturePipeline>(max_features);
        
        vo_frontend_ = std::make_unique<slam::frontend::VisualOdometryFrontend>(
            camera_matrix,
            std::move(pipeline),
            min_tracked_pts,
            reinit_threshold,
            max_corners,
            min_flow_px
        );
    }

    slam::core::FrontendResult processFrame(const cv::Mat& image,
                                            std::optional<double> external_displacement_prior,
                                            const std::optional<Eigen::Matrix3d>& imu_rotation) override {
        if (!vo_frontend_) {
            throw std::runtime_error("FrontendSift not initialized");
        }

        auto vo = vo_frontend_->track(image, external_displacement_prior, imu_rotation);
        
        slam::core::FrontendResult result;
        result.valid = vo.valid;
        if (vo.valid) {
            result.relative_transform = vo.relative_transform;
            result.confidence = vo.confidence;
            result.inliers = vo.inliers;
            result.inlier_ratio = vo.inlier_ratio;
            result.reference_age = vo.reference_age;
            result.unit_sphere_parallax = vo.unit_sphere_parallax;
        }
        return result;
    }

    void addKeyframe() override {
    }

    void reset() override {
        if (vo_frontend_) {
            vo_frontend_->reset();
        }
    }

    std::vector<cv::KeyPoint> getCurrentKeypoints() const override {
        return vo_frontend_ ? vo_frontend_->currentKeypoints() : std::vector<cv::KeyPoint>{};
    }

    cv::Mat getCurrentDescriptors() const override {
        return vo_frontend_ ? vo_frontend_->currentDescriptors() : cv::Mat{};
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_{rclcpp::get_logger("FrontendSift")};
    std::unique_ptr<slam::frontend::VisualOdometryFrontend> vo_frontend_;
};

} // namespace slam::plugins

PLUGINLIB_EXPORT_CLASS(slam::plugins::FrontendSift, slam::core::Frontend)
