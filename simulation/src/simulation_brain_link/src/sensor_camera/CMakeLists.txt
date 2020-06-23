
##Declare Nodelet
add_executable(sensor_camera
  src/sensor_camera/sensor_camera.cpp
  src/sensor_camera/sensor_camera_node.cpp
)

##Declare Nodelet
add_library(sensor_camera_nodelet
  src/sensor_camera/sensor_camera.cpp
  src/sensor_camera/sensor_camera_node.cpp
)
## set compiler definitions for nodelet
set_target_properties(sensor_camera_nodelet PROPERTIES COMPILE_DEFINITIONS "BUILD_NODELET")

if(${PROJECT_NAME}_gencfg)
  add_dependencies(sensor_camera ${PROJECT_NAME}_gencfg)
  add_dependencies(sensor_camera_nodelet ${PROJECT_NAME}_gencfg)
endif(${PROJECT_NAME}_gencfg)

add_dependencies(sensor_camera ${catkin_EXPORTED_TARGETS})
add_dependencies(sensor_camera_nodelet ${catkin_EXPORTED_TARGETS})

target_link_libraries(sensor_camera
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(sensor_camera_nodelet
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(sensor_camera ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${common_LIBRARIES})
target_link_libraries(sensor_camera_nodelet ${catkin_LIBRARIES} ${common_LIBRARIES} ${OpenCV_LIBRARIES})

