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
/ins/velocity         geometry_msgs/msg/TwistWithCovarianceStamped
```

`common` 下的 topic 名称就是对应输入链路的唯一开关：topic 非空时订阅，
topic 为空或缺失时不订阅，不再设置额外的传感器启用布尔参数。例如：

```yaml
common:
  rtk_velocity_topic: "/ins/velocity"  # 非空：更新 map 系水平速度
  wheel_odometry_topic: ""             # 为空：不订阅、不更新轮速观测
```

`localization.mode` 保留用于兼容服务和日志，不参与传感器开关判断；是否使用
某一种输入只由对应 topic 是否为空决定。当前定位滤波器是 12 维的“三维位姿 +
平面运动约束”EKF；
过程噪声、初始化不确定度、各观测的 fallback 标准差及卡方门限统一放在
`localization.ekf` 中。

RTK 位置和 RTK/INS 速度是两个相互独立的观测，不再组装成复合 RTK 消息。
每一条观测到来后，EKF 都先预测到对应 `Header.stamp`，再执行该观测
自己的更新。
厂家 INS 中的 `pitch/roll/courseang` 不再转换、订阅或用于初始化。这些量不被
当作车体相对 ENU 的绝对姿态，也不能用于求固定的 `map <- ENU` 关系。
`/ins/velocity` 的 ENU 东、北速度先旋转到 map，然后直接更新水平速度；天向
速度不进入滤波器。轮速定位入口继续保留，但 `ProcessWheelOdometry()` 当前是
明确的空入口。IMU topic 订阅专门服务于建图；定位侧的 `ProcessImu()` 兼容
入口仍保留为空，在线和离线定位运行时都不向它路由 IMU。`/imu_data` 的线
加速度和角速度均不作为定位观测，建图模块原有的 IMU 去畸变/LIO 链路不受
影响。状态顺序为：

```text
x = [p_map(3), rpy_map(3), v_map(3), omega(3)]
```

位置和姿态完整保留 `x/y/z` 与 `roll/pitch/yaw`。速度和角速度也用三维容器
存储，但每次初始化、预测和更新后都施加硬约束：

```text
v_map.z = 0
omega.x = 0
omega.y = 0
```

确定性预测模型为：

```text
x_next = x + vx_map*dt
y_next = y + vy_map*dt
z_next = z
roll_next = roll
pitch_next = pitch
yaw_next = yaw + omega_z*dt
vx_next = vx
vy_next = vy
vz_next = 0
omega_x_next = 0
omega_y_next = 0
omega_z_next = omega_z
```

NDT 更新完整的 map 系三维位置和 `roll/pitch/yaw`；GNSS 使用当前完整三维姿态
旋转并扣除杆臂后只更新三维位置，不能修改姿态或角速度；RTK/INS ENU 速度也
不能修改姿态或角速度，只更新 map 系 `vx/vy`。`z`、
`roll/pitch` 没有对应的速度状态，预测时保持上一值，同时用独立随机游走过程
噪声允许新的三维位置/姿态观测修正它们。`omega_z` 没有直接传感器观测，只能
通过连续 NDT yaw 观测形成的 yaw—角速度交叉协方差被间接估计。

在线模式的缓存规则只有两条：

- 在线建图：LiDAR 只保留最新帧，IMU 保留所有帧；
- 在线定位：LiDAR、GNSS 位置、RTK/INS 速度和轮速各自只保留一个最新值，
  新消息覆盖尚未消费的同类旧消息；IMU 不进入定位输入槽位。

离线 bag 按记录顺序逐条处理。离线建图注册 IMU 回调并保留全部 IMU 数据；
离线定位不注册 IMU 回调，只读取已启用的定位观测。

`cx16.yaml` 中 `localization.map_from_enu.lat/lon/alt` 是建图起点 GNSS 天线的
WGS84 参考坐标；`map_from_true_enu_yaw_deg` 是另外标定得到的固定
`map <- true ENU` 右手系旋转角。
它不是 `/INS_data` 的 `courseang`，也不能从 `pitch/roll/courseang` 推导出来。
定位启动时用这一固定地图标定和 `lever_arm_tracking` 建立 `map <- UTM/ENU`
关系；UTM 位置会在参考点消除网格收敛角，ENU 速度则直接旋转到 map。
定位模块不读取或乘入 `lio_sam.extrinsicRPY`。


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

在线定位保持原来的显式服务流程：`set_mode` 只切换模式，`set_map_path`
只加载地图并进入 `READY` 状态；随后必须调用 `set_location` 手动提供 map
坐标系中的初始猜测位姿，在线定位线程才会启动。是否启用 GNSS、速度或
LiDAR 观测都不会绕过这一步。

当 `lidar_topic` 和 `livox_lidar_topic` 都为空时，不创建点云订阅，也不产生
NDT 观测；但 `set_map_path` 仍会加载地图并创建定位 UI。RTK 位置、RTK/INS
速度或 NDT 每次更新后，ROS 定位话题和 UI 红色轨迹都使用同一个最终 EKF 状态。
原始 NDT 位姿在所有模式下都只作为 EKF 的一项观测，不再直接写入 UI 轨迹。
调试时可同时查看 `/lightning/localization/debug/raw_rtk_path`、
`/lightning/localization/debug/raw_ndt_path` 和最终
`/lightning/localization/path`，用来区分坐标预处理与滤波本身的问题。
Pangolin UI 中红线是最终 EKF 状态轨迹，绿线是进入 EKF 更新前、已经完成
WGS84/UTM/ENU 到 map 转换的原始 RTK 天线位置轨迹，黄线是激光扫描位姿轨迹。

只配置 `rtk_fix_topic` 时，在线定位仍然必须先调用 `set_location`；离线定位
使用自动设置的 map 原点初值。第一帧有效 `NavSatFix` 随后更新位置 EKF，
后续每帧继续更新位置并输出红色轨迹；绿色
轨迹始终是未经滤波的 GNSS 天线位置。没有 LiDAR/NDT 时不存在姿态观测，代码
会禁用杆臂补偿，避免 GNSS 位置残差通过杆臂雅可比虚构姿态和旋转；这时红线
表示天线位置。启用 LiDAR/NDT 后完整三维姿态可观，才使用三维杆臂补偿。

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_localization'}"

ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/localization/set_location lightning_interfaces/srv/SetLocation "{x: 0.0, y: 0.0, z: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0}"
ros2 service call /lightning/localization/get_map_path lightning_interfaces/srv/GetMapPath "{}"
ros2 service call /lightning/localization/get_localization_quality lightning_interfaces/srv/GetLocalizationQuality "{}"


ros2 service call /lightning/localization/finish_localization lightning_interfaces/srv/FinishLocalization "{}"
```

## 离线定位

离线定位可以先设置 bag，也可以先设置地图。两者都准备好后，运行时自动使用
map 原点单位位姿 `(0, 0, 0, 0, 0, 0)` 作为初始猜测并开始逐条读取；所有
消息按 bag 内原始顺序处理且不跳帧。离线定位不需要调用 `set_location`，这项
自动零初值只存在于离线定位，不影响在线定位和任何建图流程。

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
