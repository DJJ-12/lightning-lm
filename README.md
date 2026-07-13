# lightning service architecture

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

`save_path` is used only by mapping and means the map output directory. `map_path` is used only by localization and means the NDT map directory.

## Online localization

The point cloud subscription is created only after `set_location`, so the first received cloud is used for localization initialization.

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_localization'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/localization/set_location lightning_interfaces/srv/SetLocation "{x: 0.0, y: 0.0, z: -0.2, roll: 0.0, pitch: 0.0, yaw: 0.0}"
ros2 service call /lightning/localization/get_map_path lightning_interfaces/srv/GetMapPath "{}"
ros2 service call /lightning/localization/get_localization_quality lightning_interfaces/srv/GetLocalizationQuality "{}"
```

```bash
ros2 service call /lightning/localization/finish_localization lightning_interfaces/srv/FinishLocalization "{}"
```

## Offline localization

`load_bag` only records the bag path. Reading starts after `set_location` succeeds.

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_localization'}"
ros2 service call /lightning/localization/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260616'}"
ros2 service call /lightning/localization/set_map_path lightning_interfaces/srv/SetMapPath "{map_path: '/home/mt/maps/hesai_map'}"
ros2 service call /lightning/localization/set_location lightning_interfaces/srv/SetLocation "{x: 0.0, y: 0.0, z: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0}"
```

## Online mapping

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'online_mapping'}"
ros2 service call /lightning/mapping/start_mapping lightning_interfaces/srv/StartMapping "{save_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/mapping/finish_mapping lightning_interfaces/srv/FinishMapping "{save_map: true}"
ros2 service call /lightning/mapping/get_map_path lightning_interfaces/srv/GetMapPath "{}"
```

## Offline mapping

The map is saved automatically when the bag finishes. `finish_mapping` is for early termination; set `save_map` to `false` to stop without saving.

```bash
ros2 service call /lightning/set_mode lightning_interfaces/srv/SetMode "{mode: 'offline_mapping'}"
ros2 service call /lightning/mapping/start_mapping lightning_interfaces/srv/StartMapping "{save_path: '/home/mt/maps/cx16_map'}"
ros2 service call /lightning/mapping/load_bag lightning_interfaces/srv/LoadBag "{bag_path: '/home/mt/dataset/20260329'}"
ros2 service call /lightning/mapping/get_offline_mapping_progress lightning_interfaces/srv/GetOfflineMappingProgress "{}"
ros2 service call /lightning/mapping/finish_mapping lightning_interfaces/srv/FinishMapping "{save_map: true}"
```

## Cancel task

```bash
ros2 service call /lightning/cancel_task lightning_interfaces/srv/CancelTask "{}"
```

Cancelling does not save a map. Closing the Pangolin UI only closes rendering and does not shut down the ROS process.
