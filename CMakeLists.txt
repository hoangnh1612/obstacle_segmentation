cmake_minimum_required(VERSION 3.5)
#Print cmake version 
message("CMake version: ${CMAKE_VERSION}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(obstacle_segmentation)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/qdt/g2o/g2o/cmake_modules/)


find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(SuiteSparse REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
find_package(fmt REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
set(G2O_LIBS
    g2o_core
    g2o_stuff
    g2o_types_slam2d
    g2o_types_slam3d
    g2o_solver_dense
    g2o_solver_eigen
    g2o_csparse_extension
    g2o_solver_csparse   
)
# Set external include directories
set(EXTERNAL_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${SUITESPARSE_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS} ${fmt_INCLUDE_DIRS} ${CSPARSE_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
set(EXTERNAL_LIBS ${OpenCV_LIBS} ${SUITESPARSE_LIBRARIES} ${G2O_LIBS} ${Boost_LIBRARIES} fmt::fmt ${CSPARSE_LIBRARY})
add_executable(test_clustering
               src/test.cpp src/obstacles.cpp src/ultis.cpp

)
add_executable(obs_segment src/obs_segment.cpp src/obstacles.cpp src/ultis.cpp) 
ament_target_dependencies(obs_segment rclcpp sensor_msgs nav_msgs tf2 tf2_geometry_msgs)
install(TARGETS
  obs_segment
  DESTINATION lib/${PROJECT_NAME})

add_executable(test_planner 
               src/test_planner.cpp 
               src/obstacles.cpp 
               src/ultis.cpp
               src/homotopy_class_planner/homotopy_class_planner.cpp
               src/homotopy_class_planner/graph_search.cpp
               src/homotopy_class_planner/timed_elastic_band.cpp
               src/optimization/optimal_planner.cpp
               src/global_planner/cubic_spline.cpp 
               src/global_planner/rrt_star.cpp 

)
add_executable(teb_ros2
               src/teb_ros2.cpp
               src/obstacles.cpp 
               src/ultis.cpp
               src/homotopy_class_planner/homotopy_class_planner.cpp
               src/homotopy_class_planner/graph_search.cpp
               src/homotopy_class_planner/timed_elastic_band.cpp
               src/optimization/optimal_planner.cpp
               src/global_planner/cubic_spline.cpp 
               src/global_planner/rrt_star.cpp 

)
ament_target_dependencies(teb_ros2 geometry_msgs rclcpp sensor_msgs nav_msgs tf2 tf2_geometry_msgs)
install(TARGETS
  teb_ros2
  DESTINATION lib/${PROJECT_NAME})          
add_executable(
    test_g2o src/test_g2o.cpp
)
include_directories(
    ${EXTERNAL_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR} 
    include
    include/homotopy_class_planner
    include/g2o_types
    include/global_planner
    include/optimization
)
target_link_libraries(test_clustering ${LIBRARIES} ${EXTERNAL_LIBS})
target_link_libraries(test_g2o ${LIBRARIES} ${EXTERNAL_LIBS})
target_link_libraries(test_planner 
    ${LIBRARIES} 
    g2o_csparse_extension
    ${EXTERNAL_LIBS} 
    ${G2O_LIBS}
    ${CSPARSE_LIBRARY} 
)
target_link_libraries(obs_segment ${LIBRARIES} ${EXTERNAL_LIBS})
target_link_libraries(teb_ros2
    ${LIBRARIES} 
    g2o_csparse_extension
    ${EXTERNAL_LIBS} 
    ${G2O_LIBS}
    ${CSPARSE_LIBRARY} 
    # g2o::solver_cholmod
    # ${CXSPARSE_LIBRARIES}
)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
