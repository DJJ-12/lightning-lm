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

ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/cx16_map'}"
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

## 三维 Map–ENU 自动标定

标定是一个完整的独立模块。它在内部持有一个 `LocalizationSystem`，只通过
`Init`、`SetMapPath`、`SetInitialGuess`、`ProcessCloud` 和
`GetLatestResult` 这些公开接口把定位模块当作黑盒使用。运行层只负责把点云、
GPS1 和 GPS2 转交给标定模块；普通定位持有的 `LocalizationSystem` 不参与
标定，也没有标定专用开关或回调。`common.gps1_topic` 和
`common.gps2_topic` 是唯一两路 GPS 输入：正常定位将它们近似同步成一个
双天线观测，标定则把同一对消息作为主、从天线数据。在线定位对每根天线只
保留最新一帧，`LocalizationSystem` 内也只保留每根天线尚未配对的一帧；只有
标定模块会为了插值保留一段有界历史。

正常定位不会先使用 NDT 姿态消除 GPS 杆臂。双天线直接构成联合观测：

```text
z = [GPS1_map, GPS2_map]
h(x) = [p_map + R_map_body * lever_GPS1,
        p_map + R_map_body * lever_GPS2]
```

因此 GPS 观测噪声只来自两帧 `NavSatFix`、Map–ENU 标定协方差和已知杆臂；
NDT 仍使用自己的位姿和 Hessian 协方差。两种观测分别进入 EKF。需要注意，
一条双天线基线不能观测绕基线自身的旋转，EKF 会保留这一不可观方向，而不
人为构造一个虚假的完整三维姿态观测。

使用前先填写：

- `common.gps1_topic` 和 `common.gps2_topic` 两个
  `sensor_msgs/msg/NavSatFix` topic；
- `localization.gps_ins.lever_arm_tracking`（GPS1）和
  `gps2_lever_arm_tracking`（GPS2）在 `common.base_link_frame` 中的
  完整 XYZ 坐标；
- 正常定位的 `dual_gps_sync_tolerance_sec` 和
  `dual_gps_baseline_length_tolerance_m`；
- 两个天线共同使用的 WGS84 ENU 原点；
- 如有固定时延，填写 `gnss_time_offset_sec`。代码使用
  `GNSS校正时间 = header.stamp + gnss_time_offset_sec`。

标定和建图、定位一样，是独立的运行模式。开始服务只调用标定模块；标定模块
自己创建定位黑盒、加载 NDT 地图并设置 map 原点零位姿，运行层随后启动在线
topic 或离线 bag 数据源。不需要再调用普通定位的 `load_bag`、`set_map_path`
或 `set_location` 服务。GPS1/GPS2 只进入标定算法，不会从标定运行层送进其
内部的定位黑盒。

离线标定：

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_calibration'}"
ros2 service call /lightning/calibration/start_map_enu_calibration lightning_interfaces/srv/StartMapEnuCalibration "{map_path: '/home/mt/maps/chaosuan_map', bag_path: '/home/mt/dataset/dual_gnss_standard'}"
# bag 在后台处理；只有全局状态 finished=true 后才能 finish 标定。
ros2 service call /lightning/get_status lightning_interfaces/srv/GetStatus "{}"
ros2 service call /lightning/calibration/get_map_enu_calibration_status lightning_interfaces/srv/GetMapEnuCalibrationStatus "{}"
ros2 service call /lightning/calibration/finish_map_enu_calibration lightning_interfaces/srv/FinishMapEnuCalibration "{}"
```

在线标定：

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_calibration'}"
ros2 service call /lightning/calibration/start_map_enu_calibration lightning_interfaces/srv/StartMapEnuCalibration "{map_path: '/home/mt/maps/chaosuan_map', bag_path: ''}"
# 车辆完成足够运动后查看采样质量并结束标定。
ros2 service call /lightning/calibration/get_map_enu_calibration_status lightning_interfaces/srv/GetMapEnuCalibrationStatus "{}"
ros2 service call /lightning/calibration/finish_map_enu_calibration lightning_interfaces/srv/FinishMapEnuCalibration "{}"
```

`finish` 服务直接返回严格定义的 `enu_from_map`、运行时使用的
`map_from_enu`、ENU 原点、6×6 标定协方差和残差/可观性指标，不再创建
中间文件。需要在后续定位中使用结果时，将响应里的值写入主配置并重启：

```yaml
localization:
  map_from_enu:
    enu_origin: {lat: 34.0, lon: 113.0, alt: 100.0}
    translation: {x: 0.0, y: 0.0, z: 0.0}
    quaternion: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    calibration_covariance: [36 个按行展开的服务返回值]
```

标定模块在每次点云处理后读取内部 `LocalizationSystem` 的最新有效定位结果，
再与 GPS1/GPS2 插值配对。它不访问 `LocalizationSystem` 的内部 NDT、EKF 或
UI 对象，也不向普通定位对象注册旁路回调。

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
