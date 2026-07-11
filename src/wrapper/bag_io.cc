//
// Created by xiang on 23-12-14.
//

#include "bag_io.h"

#include <glog/logging.h>
#include <filesystem>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <yaml-cpp/yaml.h>

namespace lightning {

void RosbagIO::Go(int sleep_usec) {
    std::filesystem::path p(bag_file_);
    rosbag2_cpp::Reader reader(std::make_unique<rosbag2_cpp::readers::SequentialReader>());
    rosbag2_cpp::ConverterOptions cv_options{"cdr", "cdr"};
    reader.open({bag_file_, "sqlite3"}, cv_options);

    while (reader.has_next()) {
        auto msg = reader.read_next();
        auto iter = process_func_.find(msg->topic_name);
        if (iter != process_func_.end()) {
            const bool keep_running = iter->second(msg);
            if (!keep_running) {
                LOG(INFO) << "bag processing stopped by callback.";
                return;
            }
        }

        if (sleep_usec > 0) {
            usleep(sleep_usec);
        }

    }

    LOG(INFO) << "bag " << bag_file_ << " finished.";
}

uint64_t RosbagIO::CountMessagesFromMetadata(const std::set<std::string>& topics) const {
    namespace fs = std::filesystem;

    fs::path bag_path(bag_file_);
    fs::path metadata_path;

    if (fs::is_directory(bag_path)) {
        metadata_path = bag_path / "metadata.yaml";
    } else {
        metadata_path = bag_path.parent_path() / "metadata.yaml";
    }

    if (!fs::exists(metadata_path)) {
        LOG(WARNING) << "metadata.yaml not found: " << metadata_path.string();
        return 0;
    }

    YAML::Node root = YAML::LoadFile(metadata_path.string());
    YAML::Node info = root["rosbag2_bagfile_information"];
    if (!info) {
        LOG(WARNING) << "invalid rosbag metadata: missing rosbag2_bagfile_information";
        return 0;
    }

    YAML::Node topic_infos = info["topics_with_message_count"];
    if (!topic_infos || !topic_infos.IsSequence()) {
        LOG(WARNING) << "invalid rosbag metadata: missing topics_with_message_count";
        return 0;
    }

    uint64_t count = 0;
    for (const auto& item : topic_infos) {
        const auto topic_metadata = item["topic_metadata"];
        if (!topic_metadata || !topic_metadata["name"]) {
            continue;
        }

        const std::string topic_name = topic_metadata["name"].as<std::string>();
        if (!topics.empty() && topics.count(topic_name) == 0) {
            continue;
        }

        if (item["message_count"]) {
            count += item["message_count"].as<uint64_t>();
        }
    }

    return count;
}

}  // namespace lightning
