
##Declare Nodelet
add_executable(pre_cropping
  src/pre_cropping/pre_cropping.cpp
  src/pre_cropping/pre_cropping_node.cpp
)

##Declare Nodelet
add_library(pre_cropping_nodelet
  src/pre_cropping/pre_cropping.cpp
  src/pre_cropping/pre_cropping_node.cpp
)
## set compiler definitions for nodelet
set_target_properties(pre_cropping_nodelet PROPERTIES COMPILE_DEFINITIONS "BUILD_NODELET")

if(${PROJECT_NAME}_gencfg)
  add_dependencies(pre_cropping ${PROJECT_NAME}_gencfg)
  add_dependencies(pre_cropping_nodelet ${PROJECT_NAME}_gencfg)
endif(${PROJECT_NAME}_gencfg)

add_dependencies(pre_cropping ${catkin_EXPORTED_TARGETS})
add_dependencies(pre_cropping_nodelet ${catkin_EXPORTED_TARGETS})

target_link_libraries(pre_cropping
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(pre_cropping_nodelet
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(pre_cropping ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${common_LIBRARIES})
target_link_libraries(pre_cropping_nodelet ${catkin_LIBRARIES} ${common_LIBRARIES} ${OpenCV_LIBRARIES})

catkin_add_gtest(pre_cropping_test
  test/test_pre_cropping.cpp
  src/pre_cropping/pre_cropping.cpp
)

