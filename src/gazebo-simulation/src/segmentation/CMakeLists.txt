
##Declare Nodelet
add_executable(segmentation
  src/segmentation/segmentation.cpp
  src/segmentation/segmentation_node.cpp
)

##Declare Nodelet
add_library(segmentation_nodelet
  src/segmentation/segmentation.cpp
  src/segmentation/segmentation_node.cpp
)
## set compiler definitions for nodelet
set_target_properties(segmentation_nodelet PROPERTIES COMPILE_DEFINITIONS "BUILD_NODELET")

if(${PROJECT_NAME}_gencfg)
  add_dependencies(segmentation ${PROJECT_NAME}_gencfg)
  add_dependencies(segmentation_nodelet ${PROJECT_NAME}_gencfg)
endif(${PROJECT_NAME}_gencfg)

add_dependencies(segmentation ${catkin_EXPORTED_TARGETS})
add_dependencies(segmentation_nodelet ${catkin_EXPORTED_TARGETS})

target_link_libraries(segmentation
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(segmentation_nodelet
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(segmentation ${catkin_LIBRARIES} ${OpenCV_LIBS} ${common_LIBRARIES})
#target_link_libraries(segmentation_node ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_link_libraries(segmentation_nodelet ${catkin_LIBRARIES} ${common_LIBS} ${OpenCV_LIBRARIES})


catkin_add_gtest(segmentation_test
  test/test_pre_cropping.cpp
  src/segmentation/segmentation.cpp
)

