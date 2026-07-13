#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "lightning_interfaces/srv/cancel_task.hpp"
#include "lightning_interfaces/srv/finish_localization.hpp"
#include "lightning_interfaces/srv/finish_mapping.hpp"
#include "lightning_interfaces/srv/get_localization_quality.hpp"
#include "lightning_interfaces/srv/get_map_path.hpp"
#include "lightning_interfaces/srv/get_offline_mapping_progress.hpp"
#include "lightning_interfaces/srv/get_status.hpp"
#include "lightning_interfaces/srv/load_bag.hpp"
#include "lightning_interfaces/srv/set_location.hpp"
#include "lightning_interfaces/srv/set_map_path.hpp"
#include "lightning_interfaces/srv/set_mode.hpp"
#include "lightning_interfaces/srv/start_mapping.hpp"

#include "runtime/lightning.h"

namespace lightning::runtime {

class Service {
   public:
    bool Init(rclcpp::Node::SharedPtr node, std::shared_ptr<Lightning> lightning);

   private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<Lightning> lightning_;

    rclcpp::Service<lightning_interfaces::srv::SetMode>::SharedPtr set_mode_srv_;
    rclcpp::Service<lightning_interfaces::srv::GetStatus>::SharedPtr get_status_srv_;
    rclcpp::Service<lightning_interfaces::srv::CancelTask>::SharedPtr cancel_task_srv_;
    rclcpp::Service<lightning_interfaces::srv::GetOfflineMappingProgress>::SharedPtr get_offline_progress_srv_;
    rclcpp::Service<lightning_interfaces::srv::StartMapping>::SharedPtr start_mapping_srv_;
    rclcpp::Service<lightning_interfaces::srv::LoadBag>::SharedPtr load_mapping_bag_srv_;
    rclcpp::Service<lightning_interfaces::srv::LoadBag>::SharedPtr load_localization_bag_srv_;
    rclcpp::Service<lightning_interfaces::srv::SetMapPath>::SharedPtr set_localization_map_path_srv_;
    rclcpp::Service<lightning_interfaces::srv::FinishMapping>::SharedPtr finish_mapping_srv_;
    rclcpp::Service<lightning_interfaces::srv::FinishLocalization>::SharedPtr finish_localization_srv_;
    rclcpp::Service<lightning_interfaces::srv::GetMapPath>::SharedPtr get_mapping_map_path_srv_;
    rclcpp::Service<lightning_interfaces::srv::GetMapPath>::SharedPtr get_localization_map_path_srv_;
    rclcpp::Service<lightning_interfaces::srv::SetLocation>::SharedPtr set_location_srv_;
    rclcpp::Service<lightning_interfaces::srv::GetLocalizationQuality>::SharedPtr get_localization_quality_srv_;
};

}  // namespace lightning::runtime
