
##Declare Nodelet
add_executable(tof_sensor_emulation
  src/tof_sensor_emulation/tof_sensor_emulation.cpp
  src/tof_sensor_emulation/tof_sensor_emulation_node.cpp
)

##Declare Nodelet
add_library(tof_sensor_emulation_nodelet
  src/tof_sensor_emulation/tof_sensor_emulation.cpp
  src/tof_sensor_emulation/tof_sensor_emulation_node.cpp
)
## set compiler definitions for nodelet
set_target_properties(tof_sensor_emulation_nodelet PROPERTIES COMPILE_DEFINITIONS "BUILD_NODELET")

if(${PROJECT_NAME}_gencfg)
  add_dependencies(tof_sensor_emulation ${PROJECT_NAME}_gencfg)
  add_dependencies(tof_sensor_emulation_nodelet ${PROJECT_NAME}_gencfg)
endif(${PROJECT_NAME}_gencfg)

add_dependencies(tof_sensor_emulation ${catkin_EXPORTED_TARGETS})
add_dependencies(tof_sensor_emulation_nodelet ${catkin_EXPORTED_TARGETS})

target_link_libraries(tof_sensor_emulation
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(tof_sensor_emulation_nodelet
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)

target_link_libraries(tof_sensor_emulation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${common_LIBRARIES})
target_link_libraries(tof_sensor_emulation_nodelet ${catkin_LIBRARIES} ${common_LIBRARIES} ${PCL_LIBRARIES})

