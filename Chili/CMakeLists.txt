#step 1 设置项目以及版本最小需求
cmake_minimum_required(VERSION 3.14)
project(Chili_Detect)
#step 2 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#step 3 设置编译类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
#step 4 设置编译选项
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -Werror -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -Werror -O3 ")
# libdnn.so 依赖设置
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
#step 5 添加外部依赖包
find_package(OpenCV REQUIRED)# OpenCV
find_package(Boost REQUIRED COMPONENTS filesystem system)
#step 6 设置RDK BPU库路径
set(DNN_PATH "/usr/include/dnn")      # BPU头文件路径
set(DNN_LIB_PATH "/usr/lib/")         # BPU库文件路径
#step 7 添加头文件路径
include_directories(
	${Boost_INCLUDE_DIRS}
    ${DNN_PATH}
    ${OpenCV_INCLUDE_DIRS}
)
#step 8 添加库文件路径
link_directories(
    ${DNN_LIB_PATH}
)
#step 9 添加源文件
add_executable(main 
    main.cpp
)
#step 10 链接依赖库
target_link_libraries(main
	${Boost_LIBRARIES}
    ${OpenCV_LIBS}     # OpenCV库
    dnn                # RDK BPU库
    pthread            # 线程库
    rt                 # 实时库
    dl                 # 动态链接库
)
#step 11 安装目标
install(TARGETS main
    RUNTIME DESTINATION bin
)
