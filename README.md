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
`[x_map, y_map, yaw_map, vx_map, vy_map, yaw_rate]` 的标准二维 EKF。预测使用
map 系恒速度、恒 yaw-rate 模型；过程噪声、初始化不确定度、各观测的
fallback 标准差及卡方门限统一放在 `localization.ekf` 中。

RTK 位置、INS 姿态和 INS 速度是三个相互独立的观测，不再组装成复合 RTK 消息。
每一条观测到来后，EKF 都先预测到对应 `Header.stamp`，再执行该观测
自己的更新。`Pose2D` 没有时间戳和协方差，不再作为 INS 航向输入。
`/ins/velocity` 的 ENU 东、北速度先旋转到 map，然后直接更新 `vx_map、vy_map`，
不再套用车体前向速度模型，也不再使用位置杆臂修正速度。轮速 topic 非空时只把
`linear.x` 作为 body 系前向速度，通过
`cos(yaw) * vx_map + sin(yaw) * vy_map` 更新；不会把 `angular.z` 隐式当作陀螺仪。
`imu_topic` 继续只服务建图、点云去畸变和未来扩展。

在线模式的缓存规则只有两条：

- 在线建图：LiDAR 只保留最新帧，IMU 保留所有帧；
- 在线定位：LiDAR、IMU、GNSS 位置、INS 姿态、INS 速度和轮速各自只保留
  一个最新值，新消息覆盖尚未消费的同类旧消息。

`cx16.yaml` 中 `localization.map_from_enu` 的 `lat/lon/alt/pitch/roll/yaw`
描述建图起点；其中角度单位为度，`yaw` 是从真北轴到 map 的 +X 轴逆时针
旋转的角度。定位启动时用这六个地图参考值和现有
`lever_arm_tracking` 一次性计算固定的 `map <- UTM/ENU` 关系，不再运行时
积累轨迹求对齐。这里的 `yaw` 不是设备安装角，而是从真北轴到 map 的 +X 轴
逆时针旋转的角度。map 的 +X 轴在标准 true ENU 中的 yaw 是 `yaw+90°`；把
true ENU 坐标值转换到 map 时使用逆变换
`R_map_true_enu=Rz(-(yaw+90°))`。UTM 位置另行消除参考点的网格收敛角。
定位模块不再读取或乘入 `lio_sam.extrinsicRPY`。`pitch/roll` 保留为建图起点
的参考元数据，但不参与当前二维 EKF 的平面坐标旋转。


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
Pangolin UI 中红线是最终 EKF 状态轨迹，绿线是进入 EKF 更新前、已经完成
WGS84/UTM/ENU 到 map 转换的原始 RTK 天线位置轨迹，黄线是激光扫描位姿轨迹。

只配置 `rtk_fix_topic` 也可以直接启动在线或离线定位。此时第一帧有效
`NavSatFix` 直接初始化位置 EKF，后续每帧继续更新位置并输出红色轨迹；绿色
轨迹始终是未经滤波的 GNSS 天线位置。由于没有航向就无法把天线杆臂旋转到
map，位置单传感器模式会主动忽略 `lever_arm_tracking`，同时把输出 yaw 保持为
未观测的 0，并给它较大的初始协方差。只有配置了 `rtk_orientation_topic` 时才
等待同步的初始航向并启用杆臂补偿。

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
