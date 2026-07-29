# LIO-SAM Mapping Chain

Current mapping data flow:

```text
bag/topic/livox input
  -> MappingSystem
     -> cachePointCloud
     -> deskewInfo / imuDeskewInfo
     -> projectPointCloud
     -> cloudExtraction
     -> LioSamMapping::Run(LioSamCloudInfo&)
        -> FeatureExtractor::Run(LioSamCloudInfo&)
        -> mapOptimization::Run(LioSamCloudInfo&)
           -> internal loop closure
```

`MappingSystem` owns sensor input adaptation, point cloud format normalization,
IMU buffering, deskew, projection, and packed `LioSamCloudInfo` transport.

`LioSamMapping` owns the LIO-SAM mapping pipeline orchestration only: feature
extraction, map optimization, state update, UI update, and exported keyframes.

`FeatureExtractor` consumes the projected and deskewed cloud info and produces
corner/surface feature clouds. `mapOptimization` consumes those feature clouds
and keeps loop closure inside the optimization module.
