cmake_minimum_required(VERSION 3.10)
project(FT-Quadruped-Locomotion)

IF (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release) 
ENDIF ()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
endif()
add_compile_options(-std=c++17)

set(dirEigen "third_party/eigen3") # eigen-3.4.0
set(dirGlfw "third_party/glfw") # glfw-3
set(dirJson "third_party/jsoncpp") # for json read
set(dirQuill "third_party/quill") # for log

set(dirHpp "third_party/hpp") # for hpp-fcl
set(dirCere "third_party/cereal") # for cereal
set(dirProx "third_party/proxsuite") # for proxsuite
set(dirCasi "third_party/casadi") # for casadi
set(dirPino "third_party/pinocchio") # for pinocchio
set(dirCroc "third_party/crocoddyl") # for crocoddyl
set(dirConc "third_party/contactbench") # for contactbench

set(dirLib "third_party/lib") # for lib

set(libURDF "third_party/urdfdom") # for urdf

set(incFmt "third_party/quill/quill/bundled")
set(incBoost "third_party/boost") # for c++
set(incMujoco "third_party/mujoco") # mujoco-3.2.7

set(allDir ${dirEigen} ${dirGlfw} ${dirJson} ${dirQuill})
set(allLib ${allDir} ${libURDF} ${dirLib})
set(allInc ${allDir} ${incFmt} ${incBoost} ${incMujoco} ${dirHpp} ${dirCere} ${dirProx} ${dirCasi} ${dirPino} ${dirCroc} ${dirConc})

include_directories(${allInc})
include_directories("algorithm")

message(${CMAKE_SYSTEM_PROCESSOR})
list(APPEND allLib "third_party/mujoco/lin_x64")
link_directories(${allLib})

file(GLOB C_SOURCES *.c)
file(GLOB CPP_SOURCES *.cpp algorithm/*.cpp)
file(GLOB HEADER_FILES *.h algorithm/*.h)
set(SOURCES ${C_SOURCES} ${CPP_SOURCES} ${HEADER_FILES})


set(corLibs urdfdom_model tinyxml console_bridge jsoncpp quill)
set(simLibs glfw3)


set(sysCoreLibs)
set(sysSimLibs)

message("linux x64")
foreach(lib ${corLibs})
	list(APPEND sysCoreLibs ${lib}_lin_x64)
endforeach()
foreach(lib ${simLibs})
	list(APPEND sysSimLibs ${lib}_lin_x64)
endforeach()
message(${sysCoreLibs})


add_library(core ${SOURCES})
target_link_libraries(core ${sysCoreLibs} pthread)


option(BUILD_TESTS "Build the test programs" ON)
if(BUILD_TESTS)
	message("Build the test programs")

	#pinocchio
	#add_executable(test_pin test/test_pin.cpp)
	#target_link_libraries(test_pin core pinocchio_parsers ${sysSimLibs} dl)

	#crocoddyl
	#add_executable(test_cro test/test_cro.cpp)
	#target_link_libraries(test_cro core crocoddyl ${sysSimLibs} dl)

	#mujoco
	#add_executable(test_sim test/test_sim.cpp)
	#target_link_libraries(test_sim core pinocchio_parsers mujoco ${sysSimLibs} dl)

	#contactbench
	#add_executable(test_contact_solver test/test_contact_solver.cpp)
	#target_link_libraries(test_contact_solver core contactbench ${sysSimLibs} dl)

endif(BUILD_TESTS)

message("Build the demo programs")
add_executable(flip_cimpc demo/flip_cimpc.cpp)
target_link_libraries(flip_cimpc core casadi hpp-fcl pinocchio_parsers mujoco crocoddyl ${sysSimLibs} dl)