# lightning service 架构


`save_path` 是地图的输出保存路径. `map_path` 是定位流程种用于NDT 匹配的地图路径.
## 启动程序

```bash
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/my_mapping.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/cx16.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/s3e_campus_road_1_bob.yaml
```

所有刚体外参统一采用下面的人工可读格式，平移单位是米，姿态角单位是度：

```yaml
# [tx_m, ty_m, tz_m, roll_deg, pitch_deg, yaw_deg]
common:
  extrinsicBaseLidar: [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
```

程序读取后会自动把角度转换为弧度，并按照
`R = Rz(yaw) * Ry(pitch) * Rx(roll)` 生成旋转矩阵。参数文件不再接收
`extrinsicBaseLidarTrans` 和 `extrinsicBaseLidarRot`。



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
ros2 service call /lightning/mapping/start_mapping lightning_interfaces/srv/StartMapping "{save_path: '/home/mt/maps/test_map'}"
ros2 service call /lightning/mapping/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260921_01_standard'}"
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

离线定位使用建图原始 bag 时，第一帧 `MAP<-BODY` 固定为单位位姿。`load_bag`
和 `set_map_path` 可以按任意顺序调用；第二个条件就绪后程序会自动设置零位姿并
启动 bag，不需要调用 `set_location`。在线定位仍必须显式调用 `set_location`。


```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_localization'}"
ros2 service call /lightning/localization/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260921_01_standard'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/test_map'}"

ros2 service call /lightning/localization/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/S3E_Campus_Road_1'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/S3E_Campus_Road_1_map'}"
```

## 取消任务

```bash
ros2 service call /lightning/cancel_task lightning_interfaces/srv/CancelTask "{}"
```

## GPS/INS 与 Map–ENU 偏航标定

定位使用一个 `sensor_msgs/msg/NavSatFix` 位置 topic 和一个
`geometry_msgs/msg/TwistWithCovarianceStamped` 姿态 topic。姿态消息借用
`twist.angular.x/y/z` 保存厂家 GPS 设备的 roll、pitch、course，单位均为 rad；
它不是角速度观测。`course` 是从 ENU 正北方向顺时针旋转到 GPS 设备 `+Y`
轴的夹角；当 `course=0`（或 `360°`）时，设备 `+X` 指东、`+Y` 指北，设备系
与 ENU 重合。Lightning 在初始化和每帧姿态更新的公共入口把它转换成标准右手
ENU yaw：

```text
yaw_enu = wrap(-courseang)
```

前 10 帧有效 GPS 的经纬高取平均后建立局部 ENU 数值原点。标定期间，第一个可靠
NDT 位姿作为 0 m 起点加入；后续可靠位姿距上一个采样点超过 1 m 时加入。GPS 只保存
NDT 采样时刻前后 50 ms 内的少量数据。车辆距起点达到 50 m 后，程序按照有序时间戳，
为每个 NDT 样本选择时间上最近且相差小于 50 ms 的 GPS ENU 坐标。Map 和 ENU 都是
重力对齐的固定坐标系，因此只对两个 `2×N` 水平点矩阵执行固定尺度的二维 Umeyama，
求出 ENU 到 map 的 yaw 与 xy 平移；roll、pitch 固定为 0，z 平移由匹配点高度差均值
得到：

```text
T_map_enu_xy = Eigen::umeyama(gps_xy, ndt_xy, false)
R_map_enu    = Rz(yaw_map_enu)
t_map_enu.xy = T_map_enu_xy.block<2, 1>(0, 2)
t_map_enu.z  = mean(z_map - z_enu)
p_map        = R_map_enu * p_enu + t_map_enu
```

标定成功后，程序会把结果写回本次 `--config` 传入的同一个参数文件，只更新
`localization.map_enu_calibration` 这一小段，其他参数、注释和排版保持不变：

```yaml
localization:
  map_enu_calibration:
    # p_map = R_map_enu * p_enu + translation_map_enu
    translation_map_enu: [tx, ty, tz]
    rotation_map_enu_rpy_deg: [roll, pitch, yaw]
```

这里保存的是本次标定结果，RPY 角度单位为度，平移单位为米。该写回功能不改变现有
采样和标定流程；程序每次启动仍按原流程重新采样并在成功后覆盖这两个结果字段。
如果参数文件没有写权限，定位仍继续使用内存中已经算出的标定结果，同时日志中的
`config_saved=false` 会明确指出写回失败。

`NavSatFix` 经纬高表示 GPS 设备坐标系原点，该点直接作为 EKF 的位置观测，
不再定义天线点或位置杆臂。只有可选的 INS 姿态观测需要 GPS 设备坐标系到
body 坐标系的旋转：

```yaml
common:
  gps_topic: "/fdilink/gnss_fix"
  orientation_topic: "/ins/orientation"

localization:
  gps_ins:
    # R_body_gps 的 [roll_deg, pitch_deg, yaw_deg]，仅用于 INS 姿态观测
    gps_rotation_body_gps_rpy_deg: [0.0, 0.0, 0.0]
```

标定完成后，每帧 GPS 位置及其 ENU 协方差都先转换到 map：

```text
p_M       = R_M_E * p_E + t_M_E
Cov_M     = R_M_E * Cov_E * R_M_E^T
r         = p_M(measured) - state.position_map
H_position = I3
```

平移不参与协方差变换。消息协方差有效时保留其完整 ENU 三维矩阵（包括交叉项）；
无效时才使用配置的标准差。`gps_position_std_z` 同时是高程标准差下限，当前配置为
`100 m`，即高程方差至少为 `10000 m^2`，使 map 高程主要由 NDT 决定。

本项目的 `/fdilink/gnss_fix` 对协方差采用设备接口的实际语义：虽然消息类型是
`NavSatFix`，`position_covariance` 中保存的是 ECEF XYZ 方差，单位为 `m²`，并非
标准消息通常约定的 ENU 方差。Lightning 使用第一帧 GPS 建立局部 ENU 原点时生成
固定的 `R_enu_ecef`，先执行：

```text
Cov_ENU = R_enu_ecef * Cov_ECEF * R_enu_ecef^T
```

再执行：

```text
Cov_MAP = R_map_enu * Cov_ENU * R_map_enu^T
```

ECEF→ENU 使用的是 ENU 原点处的固定旋转，而不是每一帧重新建立一个局部切平面；
这样得到的协方差和 `ForwardDegrees()` 输出的位置处于同一个固定 ENU 坐标系。

每帧 INS 姿态先在 `LocalizationSystem` 中转换成与 EKF 姿态状态完全相同的
`MAP<-BODY` 欧拉角：

```text
R_E_G(measured) = RpyToRotation(roll, pitch, yaw_enu)
R_M_B(measured) = R_M_E * R_E_G(measured) * R_B_G^T
z_rpy           = RotationToRpy(R_M_B(measured))
r_rpy           = wrap(z_rpy - state.rpy_map)
```

坐标变换不放进 EKF。进入 EKF 时观测量和状态量都已经是 `MAP<-BODY` RPY，
因此观测函数就是 `h(x)=state.rpy_map`，观测雅可比的姿态块严格为 3x3 单位阵，
不再使用数值微分。姿态观测不与 GPS 位置拼成一帧
虚假的六维 pose。三维姿态消息协方差用于该观测，并由配置中的
`gps_orientation_std_floor_*_deg` 提供最小标准差。

EKF 保留原来的预测、LDLT、马氏距离门控、增益和 Joseph 协方差更新。GPS
位置只约束位置，但可以通过 EKF 中现有的交叉协方差间接影响其他状态。NDT 仍以自身位姿
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
