# lightning service 架构

The runtime supports four modes:

```text
offline_mapping
online_mapping
offline_localization
online_localization
```

Global services:

```text
/lightning/set_mode
/lightning/get_status
/lightning/cancel_task
```

Mapping services:

```text
/lightning/mapping/start_mapping
/lightning/mapping/load_bag
/lightning/mapping/finish_mapping
/lightning/mapping/get_map_path
/lightning/mapping/get_offline_mapping_progress
```

Localization services:

```text
/lightning/localization/load_bag
/lightning/localization/set_map_path
/lightning/localization/set_location
/lightning/localization/finish_localization
/lightning/localization/get_map_path
/lightning/localization/get_localization_quality
```

`save_path` 是地图的输出保存路径. `map_path` 是定位流程种用于NDT 匹配的地图路径.
## 启动程序

```bash
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/my_mapping.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/cx16.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/s3e_campus_road_1_bob.yaml
```

## CX16 标准传感器输入

Lightning 不依赖厂家 `msg_out`。使用 `cx16.yaml` 时，在线订阅和离线 bag
读取使用同一组标准 ROS 2 消息：

```text
/cx/cloud             sensor_msgs/msg/PointCloud2
/imu_data             sensor_msgs/msg/Imu
/fdilink/gnss_fix     sensor_msgs/msg/NavSatFix
/ins/orientation      sensor_msgs/msg/Imu
/ins/velocity         geometry_msgs/msg/TwistWithCovarianceStamped
```

`common` 下的 topic 名称就是对应输入链路的唯一开关：topic 非空时订阅，
topic 为空或缺失时不订阅，不再设置额外的传感器启用布尔参数。例如：

```yaml
common:
  rtk_velocity_topic: "/ins/velocity"  # 非空：作为二维速度观测
  wheel_odometry_topic: ""             # 为空：不订阅、不更新轮速观测
```

`localization.mode` 保留用于兼容服务和日志，不参与传感器开关判断；是否使用
某一种观测只由对应 topic 是否为空决定。当前定位滤波器是状态为
`[x, y, yaw, velocity, yaw_rate]` 的标准二维 EKF，预测模型为 CTRV；过程噪声、
初始化不确定度、各观测的 fallback 标准差及卡方门限统一放在
`localization.ekf` 中。

RTK 位置、INS 姿态和 INS 速度是三个相互独立的观测，不再组装成复合 RTK 消息。
每一条观测到来后，EKF 都先用 CTRV 预测到对应 `Header.stamp`，再执行该观测
自己的更新。`Pose2D` 没有时间戳和协方差，不再作为 INS 航向输入。
`/ins/velocity` 非空时作为 map 系二维速度观测，轮速 topic 非空时更新车体前向
速度和 yaw rate。`imu_topic` 不会被隐式当作 yaw-rate 观测，只继续服务建图、
点云去畸变和未来扩展。

在线模式的缓存规则只有两条：

- 在线建图：LiDAR 只保留最新帧，IMU 保留所有帧；
- 在线定位：LiDAR、IMU、GNSS 位置、INS 姿态、INS 速度和轮速各自只保留
  一个最新值，新消息覆盖尚未消费的同类旧消息。

`cx16.yaml` 中 `localization.map_from_enu` 的 `lat/lon/alt/pitch/roll/yaw`
描述建图起点；其中角度单位为度，`yaw` 沿用厂家航向角定义（北为 0°、
顺时针为正）。定位启动时用这六个地图参考值和现有
`lever_arm_tracking` 一次性计算固定的 `map <- UTM/ENU` 关系，不再运行时
积累轨迹求对齐。


## 在线建图

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_mapping'}"
ros2 service call /lightning/mapping/start_mapping lightning_interfaces/srv/StartMapping "{save_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/mapping/finish_mapping lightning_interfaces/srv/FinishMapping "{save_map: false}"
ros2 service call /lightning/mapping/get_map_path lightning_interfaces/srv/GetMapPath "{}"
```

## 离线建图

当bag完成后自动保存地图. `finish_mapping` 是为了提前终止; 设置 `save_map` 为 `false` 意味着不保存地图.

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_mapping'}"
ros2 service call /lightning/mapping/start_mapping lightning_interfaces/srv/StartMapping "{save_path: '/home/mt/maps/S3E_Campus_Road_1_map'}"
ros2 service call /lightning/mapping/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/S3E_Campus_Road_1'}"
ros2 service call /lightning/mapping/get_offline_mapping_progress lightning_interfaces/srv/GetOfflineMappingProgress "{}"
ros2 service call /lightning/mapping/finish_mapping lightning_interfaces/srv/FinishMapping "{save_map: true}"
```


## 在线定位

使用 `cx16.yaml` 的融合模式时，`set_map_path` 成功后会立即开始接收定位
观测。第一组时间接近的 GNSS 位置和 INS 姿态先初始化 EKF，并自动作为
第一帧 NDT 的地图内初值；不再要求先调用 `set_location`。`set_location`
仍可用于人工重定位。纯 `ndt_only` 配置仍然必须先调用 `set_location`。

当 `lidar_topic` 和 `livox_lidar_topic` 都为空时，不创建点云订阅，也不产生
NDT 观测；但 `set_map_path` 仍会加载地图并创建定位 UI。RTK 位置、INS yaw
或 NDT 每次更新后，ROS 定位话题和 UI 红色轨迹都使用同一个最终 EKF 状态。
原始 NDT 位姿在所有模式下都只作为 EKF 的一项观测，不再直接写入 UI 轨迹。
调试时可同时查看 `/lightning/localization/debug/raw_rtk_path`、
`/lightning/localization/debug/raw_ndt_path` 和最终
`/lightning/localization/path`，用来区分坐标预处理与滤波本身的问题。

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_localization'}"

ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/localization/get_map_path lightning_interfaces/srv/GetMapPath "{}"
ros2 service call /lightning/localization/get_localization_quality lightning_interfaces/srv/GetLocalizationQuality "{}"


ros2 service call /lightning/localization/finish_localization lightning_interfaces/srv/FinishLocalization "{}"
```

## 离线定位

融合模式下，先设置 bag 或先设置地图均可；bag 和地图都准备好后会自动
开始逐条读取，所有消息按 bag 内原始顺序处理且不跳帧。纯 `ndt_only`
模式仍在 `set_location` 之后开始读取。

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_localization'}"
ros2 service call /lightning/localization/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260905-02_standard'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/chaosuan_map'}"
```

## 取消任务

```bash
ros2 service call /lightning/cancel_task lightning_interfaces/srv/CancelTask "{}"
```

```bash
Localizer::RegisterFrame()
    ↓
ndt_ptr_->align(output, initial_guess)
    ↓
pcl::Registration::align()
    ↓
MultiGridNormalDistributionsTransform::computeTransformation()
    ↓
computeDerivatives()
    ↓
updateDerivatives()
```
