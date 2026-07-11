cmake_minimum_required(VERSION 3.16)

project(lightning)

# Default to C++17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

# set to release by default
set(CMAKE_BUILD_TYPE Release)

# -------------------------------------------------------------
# 导入第三方库，同时做一些全局编译选项配置
add_definitions("-DPCL_NO_PRECOMPILE")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-parentheses)
    add_compile_options(-Wno-pedantic)  
    add_compile_options(-Wno-reorder)    
    add_compile_options(-Wno-sign-compare)  
    add_compile_options(-Wno-unused-but-set-variable)
    add_compile_options(-Wno-unused-parameter)    
    add_compile_options(-Wno-unused-result)
    add_compile_options(-Wno-unused-variable)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/packages.cmake)

install(DIRECTORY config
        DESTINATION lib/${PROJECT_NAME}/
)

# 首先构建 livox_ros_driver2 子项目
add_subdirectory(thirdparty/livox_ros_driver)

# 添加对 livox_interfaces2 的依赖
add_subdirectory(src)

ament_package()
