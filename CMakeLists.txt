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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp -fPIC -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -fopenmp -fPIC -g -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -ggdb -g -fopenmp -fPIC -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -g")


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

find_package(visualization_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenMP REQUIRED)
find_package(rosidl_default_generators REQUIRED)


# msg 和 srv
set(MY_MSG_FILES "")
set(MSG_DIR ${PROJECT_SOURCE_DIR}/msg)
file(GLOB_RECURSE MSG_FILES ${MSG_DIR}/*.msg)
foreach (MSG_FILE ${MSG_FILES})
    file(RELATIVE_PATH REL_PATH ${MSG_DIR} ${MSG_FILE})
    list(APPEND MY_MSG_FILES ${PROJECT_SOURCE_DIR}:msg/${REL_PATH})
endforeach ()

set(MY_ACTION_FILES "")

set(MY_SRV_FILES
        ${PROJECT_SOURCE_DIR}:srv/SetLocation.srv
)
set(SRV_DIR ${PROJECT_SOURCE_DIR}/srv)
file(GLOB_RECURSE SRV_FILES ${SRV_DIR}/*.srv)
foreach (SRV_FILE ${SRV_FILES})
    file(RELATIVE_PATH REL_PATH ${SRV_DIR} ${SRV_FILE})
    if (NOT REL_PATH STREQUAL "SetLocation.srv")
        list(APPEND MY_SRV_FILES ${PROJECT_SOURCE_DIR}:srv/${REL_PATH})
    endif ()
endforeach ()

rosidl_generate_interfaces(${PROJECT_NAME}
        ${MY_SRV_FILES}
        ${MY_MSG_FILES}
        ${MY_ACTION_FILES}
        DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs
)

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)

install(DIRECTORY config
        DESTINATION lib/${PROJECT_NAME}/
)

add_subdirectory(thirdparty/livox_ros_driver)

add_subdirectory(src)

ament_package()
