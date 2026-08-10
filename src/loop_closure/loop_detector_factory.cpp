#include "visual_graph_slam/loop_closure/loop_detector.hpp"

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

namespace slam::loop_closure {

std::unique_ptr<LoopDetector> createLoopDetector(const std::string& module_name) {
    const std::string normalized = toLower(module_name);

    if (normalized == "orb_bow" || normalized == "orb-bow" || normalized == "bow") {
        return std::make_unique<OrbBowDetector>();
    }

    if (normalized == "dbow2") {
#if VISUAL_GRAPH_SLAM_WITH_DBOW2
        return std::make_unique<DBoW2Detector>();
#else
        throw std::invalid_argument(
            "Unsupported loop.vpr_backend: dbow2. DBoW2 is not available in this build. "
            "Use loop.vpr_backend=orb_bow or install DBoW2.");
#endif
    }

    if (normalized == "netvlad") {
        throw std::invalid_argument(
            "Unsupported loop.vpr_backend: netvlad. Supported backends: orb_bow"
#if VISUAL_GRAPH_SLAM_WITH_DBOW2
            ", dbow2"
#endif
            "."
        );
    }

    throw std::invalid_argument("Unsupported loop.vpr_backend: " + module_name +
                                ". Supported: orb_bow"
#if VISUAL_GRAPH_SLAM_WITH_DBOW2
                                ", dbow2"
#endif
                                );
}

}  // namespace slam::loop_closure
