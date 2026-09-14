# lightning service 架构


`save_path` 是地图的输出保存路径. `map_path` 是定位流程种用于NDT 匹配的地图路径.
## 启动程序

```bash
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/my_mapping.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/cx16.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/s3e_campus_road_1_bob.yaml
```



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


```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_localization'}"

ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/chaosuan_map'}"
ros2 service call /lightning/localization/set_location lightning_interfaces/srv/SetLocation "{x: 0.0, y: 0.0, z: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0}"
ros2 service call /lightning/localization/get_map_path lightning_interfaces/srv/GetMapPath "{}"
ros2 service call /lightning/localization/get_localization_quality lightning_interfaces/srv/GetLocalizationQuality "{}"


ros2 service call /lightning/localization/finish_localization lightning_interfaces/srv/FinishLocalization "{}"
```

## 离线定位


```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_localization'}"
ros2 service call /lightning/localization/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260905-02_standard'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/chaosuan_map'}"

ros2 service call /lightning/localization/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/S3E_Campus_Road_1'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/S3E_Campus_Road_1_map'}"
```

## 取消任务

```bash
ros2 service call /lightning/cancel_task lightning_interfaces/srv/CancelTask "{}"
```

## GPS/INS 与三维 Map–ENU 初始化

定位使用一个 `sensor_msgs/msg/NavSatFix` 位置 topic 和一个
`geometry_msgs/msg/TwistWithCovarianceStamped` 姿态 topic。姿态消息借用
`twist.angular.x/y/z` 保存 GPS 设备坐标系在 ENU 中的 roll、pitch、yaw，单位
均为 rad；它不是角速度观测，也不会在运行阶段直接更新 EKF 姿态。

第一帧有效 GPS 建立局部 ENU 数值原点。首个可靠 NDT 位姿（无雷达时为服务
设置的初始位姿）建立初始 `MAP<-BODY`。车辆保持静止时，程序近似同步前 N 组
GPS 与姿态消息，先用静态外参把 GPS 输出点还原为 body 原点：

```text
l_B_A = t_B_G + R_B_G * l_G_A
R_E_B = R_E_G * R_B_G^T
p_E_B = p_E_A - R_E_B * l_B_A
T_E_M = T_E_B0 * inverse(T_M_B0)
```

其中 `A` 是 `NavSatFix` 的实际输出点，`G` 是 GPS 设备坐标系，`B` 是
`common.base_link_frame`。位置直接取平均，四元数先统一符号再平均归一化，得到
固定的完整三维 `ENU<-MAP`。安装关系全部来自配置，不在算法中硬编码 90°：

```yaml
common:
  gps_topic: "/fdilink/gnss_fix"
  orientation_topic: "/ins/orientation"

localization:
  gps_ins:
    gps_output_lever_arm_gps: [0.0, 0.0, 0.0]
    gps_translation_tracking_gps: [0.0, 0.0, 0.0]
    gps_rotation_tracking_gps_rpy: [0.0, 0.0, 1.5707963267948966]
    gps_initialization_sync_tolerance_sec: 0.05
    gps_initialization_sample_count: 10
```

初始化完成后，姿态消息退出融合链。每帧 GPS 位置不再等待姿态或 NDT，直接以
三维 ENU 输出点观测进入 EKF：

```text
h(x) = R_E_M * (p_M_B + R_M_B(rpy) * l_B_A) + t_E_M
r    = p_E_A(measured) - h(x)
```

EKF 保留原来的预测、LDLT、马氏距离门控、增益和 Joseph 协方差更新。非零杆臂
通过观测雅可比与 RPY 产生耦合；杆臂为零时，GPS 只约束位置。NDT 仍以自身位姿
和 Hessian 协方差独立进入 EKF。UI 中绿色为 GPS 原始输出点转换到 map 后的轨迹，
黄色为 NDT 原始观测，红色为最终定位结果；三条轨迹按各自观测到达时独立追加。

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
