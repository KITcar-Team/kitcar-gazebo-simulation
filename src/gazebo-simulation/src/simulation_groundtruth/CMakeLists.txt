
##Declare Nodelet
add_executable(groundtruth_publisher
  src/groundtruth_publisher/groundtruth_publisher.cpp
  src/groundtruth_publisher/groundtruth_publisher_node.cpp
)

##Declare Nodelet
add_library(groundtruth_publisher_nodelet
  src/groundtruth_publisher/groundtruth_publisher.cpp
  src/groundtruth_publisher/groundtruth_publisher_node.cpp
)
## set compiler definitions for nodelet
set_target_properties(groundtruth_publisher_nodelet PROPERTIES COMPILE_DEFINITIONS "BUILD_NODELET")

if(${PROJECT_NAME}_gencfg)
  add_dependencies(groundtruth_publisher ${PROJECT_NAME}_gencfg)
  add_dependencies(groundtruth_publisher_nodelet ${PROJECT_NAME}_gencfg)
endif(${PROJECT_NAME}_gencfg)

add_dependencies(groundtruth_publisher ${catkin_EXPORTED_TARGETS})
add_dependencies(groundtruth_publisher_nodelet ${catkin_EXPORTED_TARGETS})

target_link_libraries(groundtruth_publisher
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(groundtruth_publisher_nodelet
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)

target_link_libraries(groundtruth_publisher ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${common_LIBRARIES})
target_link_libraries(groundtruth_publisher_nodelet ${catkin_LIBRARIES} ${common_LIBRARIES} ${PCL_LIBRARIES})

