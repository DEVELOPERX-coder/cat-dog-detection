cmake_minimum_required(VERSION 3.10)
project(CatDogDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(cat_dog_detector ${SOURCES})

# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(cat_dog_detector Threads::Threads)