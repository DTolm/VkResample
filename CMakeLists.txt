cmake_minimum_required(VERSION 3.11)
project(VkResample)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(${PROJECT_NAME} VkResample.cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

find_package(Vulkan REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glslang-master/glslang/Include/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/glslang-master)

add_library(VkFFT INTERFACE)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/)
target_include_directories(VkFFT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/glslang-master/glslang/Include/)

add_library(half INTERFACE)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/)

add_library(stb_image INTERFACE)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/stb_image/)

add_library(stb_image_write INTERFACE)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/stb_image/)

target_link_libraries(${PROJECT_NAME} PUBLIC SPIRV glslang stb_image stb_image_write Vulkan::Vulkan VkFFT half)