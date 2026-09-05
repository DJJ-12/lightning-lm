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

`common` 下的 topic 名称就是对应传感器的唯一开关：topic 非空时订阅并使用，
topic 为空或缺失时不订阅、也不进入定位融合，不再设置额外的传感器启用
布尔参数。例如：

```yaml
common:
  rtk_velocity_topic: "/ins/velocity"  # 启用 INS 速度观测
  wheel_odometry_topic: ""             # 关闭轮速计观测
```

`localization.mode` 只选择 `ndt_only`、融合定位或纯 RTK/INS 定位算法；它不再
参与 topic 订阅开关判断。轴选择、噪声和卡方门限仍放在 `localization.eskf`
中，用于描述已启用观测的融合方式。

RTK 位置、INS 姿态和 ENU 速度是三个相互独立的观测，不再同步或组装成
复合 RTK 消息。每个观测到来后，由 ESKF 在对应 `Header.stamp` 执行一次
`PredictTo(stamp) + Update`。`Pose2D` 没有时间戳和协方差，不再作为 INS
航向输入；`/ins/velocity` 是组合导航速度，不是轮速计。

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
观测。第一组时间接近的 GNSS 位置和 INS 姿态先初始化 ESKF，并自动作为
第一帧 NDT 的地图内初值；不再要求先调用 `set_location`。`set_location`
仍可用于人工重定位。纯 `ndt_only` 配置仍然必须先调用 `set_location`。

当 `lidar_topic` 和 `livox_lidar_topic` 都为空时，不创建点云订阅，也不产生
NDT 观测；但 `set_map_path` 仍会加载地图并创建定位 UI。RTK、INS、IMU 或
轮速观测每次使 ESKF 前进后，ROS 定位话题和 UI 红色轨迹都使用同一个最终
ESKF 状态。融合模式下原始 NDT 位姿只作为 ESKF 的一项观测，不再直接写入
UI 轨迹。

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
