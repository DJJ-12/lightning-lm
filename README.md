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
ros2 service call /lightning/mapping/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260905-02_standard'}"
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
`twist.angular.x/y/z` 保存厂家 GPS 设备的 roll、pitch、course，单位均为 rad；
它不是角速度观测。`course` 仍保持厂家“北向为 0、顺时针为正”的定义，Lightning
在初始化和每帧姿态更新的公共入口把它转换成“东向为 0、逆时针为正”的标准右手
ENU yaw：

```text
yaw_enu = wrap(pi / 2 - courseang)
```

第一帧有效 GPS 建立局部 ENU 数值原点。首个可靠 NDT 位姿（无雷达时为服务
设置的初始位姿）建立初始 `MAP<-BODY`。车辆保持静止时，程序近似同步前 N 组
GPS 与姿态消息，先用静态外参把 GPS 输出点还原为 body 原点：

```text
p_B_A = t_B_G + R_B_G * p_G_A
R_E_B = R_E_G * R_B_G^T
p_E_B = p_E_A - R_E_B * p_B_A
T_E_M = T_E_B0 * inverse(T_M_B0)
```

其中 `A` 是 `NavSatFix` 的实际输出点，`G` 是 GPS 设备坐标系，`B` 是
车体坐标系（其 ROS frame 名由 `common.base_link_frame` 指定）。位置直接取平均，
四元数先统一符号再平均归一化，得到
固定的完整三维 `ENU<-MAP`。安装关系全部来自配置，不在算法中硬编码 90°：

```yaml
common:
  gps_topic: "/fdilink/gnss_fix"
  orientation_topic: "/ins/orientation"

localization:
  gps_ins:
    # p_G_A: /fdilink/gnss_fix 所表示的天线点 A 在 GPS 设备坐标系 G 下的坐标，单位 m
    gps_output_lever_arm_gps: [0.0, 0.0, 0.0]
    # [tx_m, ty_m, tz_m, roll_deg, pitch_deg, yaw_deg]
    gps_extrinsic_body_gps: [0.0, 0.0, 0.0, 0.0, 0.0, 90.0]
    gps_initialization_sync_tolerance_sec: 0.05
    gps_initialization_sample_count: 10
```

这里的杆臂不是 body 坐标，也不是 GPS 设备原点在 body 下的位置。它严格定义为
`NavSatFix` 经纬高所代表的天线点 `A` 在 GPS 设备坐标系 `G` 下的坐标
`p_G_A`。程序再通过完整外参 `T_B_G` 计算：

```text
p_B_A = t_B_G + R_B_G * p_G_A
```

初始化时从天线位置恢复 body 原点，运行时则从 EKF body 位姿预测同一个天线点：

```text
p_E_B          = p_E_A - R_E_B * p_B_A
p_E_A(predicted) = R_E_M * (p_M_B + R_M_B * p_B_A) + t_E_M
```

初始化完成后，GPS 位置和 INS 姿态按照各自时间戳分别进入 EKF，二者不再互相
等待。每帧 GPS 位置直接以三维 ENU 输出点观测进入 EKF：

```text
h(x) = R_E_M * (p_M_B + R_M_B(rpy) * p_B_A) + t_E_M
r    = p_E_A(measured) - h(x)
```

每帧 INS 姿态先在 `LocalizationSystem` 中转换成与 EKF 姿态状态完全相同的
`MAP<-BODY` 欧拉角：

```text
R_E_G(measured) = RpyToRotation(roll, pitch, yaw_enu)
R_M_B(measured) = R_E_M^T * R_E_G(measured) * R_B_G^T
z_rpy           = RotationToRpy(R_M_B(measured))
r_rpy           = wrap(z_rpy - state.rpy_map)
```

坐标变换不放进 EKF。进入 EKF 时观测量和状态量都已经是 `MAP<-BODY` RPY，
因此观测函数就是 `h(x)=state.rpy_map`，观测雅可比的姿态块严格为 3x3 单位阵，
不再使用数值微分。姿态观测不使用杆臂位置雅可比，也不与 GPS 位置拼成一帧
虚假的六维 pose。三维姿态消息协方差用于该观测，并由配置中的
`gps_orientation_std_floor_*_deg` 提供最小标准差。

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
