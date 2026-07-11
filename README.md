# lightning clean service architecture v2

程序启动后常驻，客户只能通过 `/lightning/*` 服务控制。

## 层级

```text
run_lightning
  -> lightning
  -> service
  -> mode + task + input
  -> Mapping / Localization
  -> SaveMap
```

## 三种模式

```text
offline_mapping: BagInput   -> Mapping      -> SaveMap
online_mapping : TopicInput -> Mapping      -> SaveMap
localization   : TopicInput -> Localization
```

## 命名

- `lightning`: 唯一常驻主程序/调度类
- `service`: 注册所有服务
- `mode`: 当前模式
- `task`: 当前任务状态和进度
- `BagInput`: 读取 rosbag
- `TopicInput`: 订阅 topic
- `Mapping`: 建图算法主模块，只调用 LIO-SAM/原前端，不关心离线/在线
- `Localization`: 定位算法主模块，内部保留 robot_localizer::Localizer 和 MapLoader
- `SaveMap`: 地图保存模块

## 坐标约定

建图和定位统一由外层读取：

```text
common.extrinsicBaseLidarTrans
common.extrinsicBaseLidarRot
```

`MappingSystem` 会先把输入点云从 LiDAR 坐标系转换到 `base_link`，再交给 LIO-SAM 或 FAST-LIO。两个前端的输入点云都视为 `base_link` 坐标系，输出状态都视为 `T_map_base`。FAST-LIO 内部不再读取 `fasterlio.extrinsic_T/R`，其内部 lidar-to-IMU 外参固定为单位阵，避免同一帧点云被重复套用外参。

## SetMapPath 与 MapLoader 的关系

`/lightning/set_map_path` 只负责让客户指定地图根路径，例如：

```text
/home/mt/maps/new_map
```

`core/localization/MapLoader` 仍然属于 robot_localizer 定位算法内部，用于定位时根据当前位姿加载附近 BlockMap 地图块做 NDT 匹配。

这两个概念不要混淆。

## 主要服务

```text
/lightning/set_mode
/lightning/get_status
/lightning/cancel_task

/lightning/get_offline_mapping_progress

/lightning/start_mapping
/lightning/finish_mapping
/lightning/save_map

/lightning/set_map_path
/lightning/get_map_path
/lightning/set_location
/lightning/get_localization_quality
```

## 启动

```bash
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/my_mapping.yaml
ros2 run lightning run_lightning --config /home/mt/workspace/src/lightning-lm/config/cx16.yaml
```

## 典型流程

### 离线建图

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_mapping'}"
ros2 service call /lightning/start_mapping lightning_interfaces/srv/StartMapping "{bag_path: '/home/mt/dataset/20260329', save_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/get_status lightning_interfaces/srv/GetStatus "{}"
ros2 service call /lightning/get_offline_mapping_progress lightning_interfaces/srv/GetOfflineMappingProgress "{}"
```

### 在线建图

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_mapping'}"
ros2 service call /lightning/start_mapping lightning_interfaces/srv/StartMapping "{bag_path: '', save_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/finish_mapping lightning_interfaces/srv/FinishMapping "{}"
```

### 在线定位

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'localization'}"
ros2 service call /lightning/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/set_location lightning_interfaces/srv/SetLocation "{x: 0.0, y: 0.0, z: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0}"
ros2 service call /lightning/get_localization_quality lightning_interfaces/srv/GetLocalizationQuality "{}"

ros2 service call /lightning/cancel_task
```
### 取消任务

```bash
ros2 service call /lightning/cancel_task lightning_interfaces/srv/CancelTask "{}"
```
