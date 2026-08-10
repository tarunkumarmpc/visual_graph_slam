#include "visual_graph_slam/frontend/feature_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace {

std::string toLower(std::string input) {
    std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return input;
}

}  // namespace

namespace slam::frontend {

std::unique_ptr<FeaturePipeline> createFeaturePipeline(const std::string& frontend_name) {
    const std::string normalized = toLower(frontend_name);

    if (normalized == "orb") {
        return std::make_unique<OrbFeaturePipeline>();
    }

    if (normalized == "superpoint") {
        return std::make_unique<SuperPointFeaturePipeline>();
    }

    if (normalized == "sift") {
        return std::make_unique<SiftFeaturePipeline>();
    }

    if (normalized == "custom" || normalized == "custom_orb" || normalized == "custom-clahe-orb") {
        return std::make_unique<CustomFeaturePipeline>();
    }

    if (normalized == "surf") {
        throw std::invalid_argument(
            "Unsupported frontend.detector: surf. SURF is not available in this build. "
            "Use orb, sift, superpoint, or custom.");
    }

    throw std::invalid_argument("Unsupported frontend.detector: " + frontend_name +
                                ". Supported: orb, sift, superpoint, custom");
}

}  // namespace slam::frontend
