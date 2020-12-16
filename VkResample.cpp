// This file is part of VkResample, a Vulkan real-time FFT resampling tool
//
// Copyright (C) 2020 Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. 
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#include "vkFFT.h"
#include "vulkan/vulkan.h"
#include "half.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "glslang_c_interface.h"
using half_float::half;

typedef half half2[2];

const bool enableValidationLayers = false;

typedef struct {
	VkInstance instance;//a connection between the application and the Vulkan library 
	VkPhysicalDevice physicalDevice;//a handle for the graphics card used in the application
	VkPhysicalDeviceProperties physicalDeviceProperties;//bastic device properties
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;//bastic memory properties of the device
	VkDevice device;//a logical device, interacting with physical device
	VkDebugUtilsMessengerEXT debugMessenger;//extension for debugging
	uint32_t queueFamilyIndex;//if multiple queues are available, specify the used one
	VkQueue queue;//a place, where all operations are submitted
	VkCommandPool commandPool;//an opaque objects that command buffer memory is allocated from
	VkFence fence;//a vkGPU->fence used to synchronize dispatches
	uint32_t device_id;//an id of a device, reported by Vulkan device list
	std::vector<const char*> enabledDeviceExtensions;
} VkGPU;//an example structure containing Vulkan primitives

const char validationLayers[28] = "VK_LAYER_KHRONOS_validation";

typedef struct {
	//system size
	uint32_t localSize[3];
	uint32_t size[3];
	uint32_t inputStride[3];
	//how much memory is coalesced (in bytes) - 32 for Nvidia, 64 for Intel, 64 for AMD. Maximum value: 128
	uint32_t coalescedMemory;
	//bridging information, that allows shaders to freely access resources like buffers and images
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	//pipeline used for graphics applications, we only use compute part of it in this example
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	//input buffer
	VkDeviceSize inputBufferSize;//the size of buffer (in bytes)
	VkBuffer* inputBuffer;//pointer to the buffer object
	//VkDeviceMemory* inputBufferDeviceMemory;//pointer to the memory object, corresponding to the buffer
	//output buffer
	VkDeviceSize outputBufferSize;
	VkBuffer* outputBuffer;
	//VkDeviceMemory* outputBufferDeviceMemory;
	uint32_t numCoordinates;
	uint32_t precision; //0-single, 1-double, 2-half
	uint32_t r2c;
	char* code0;
} VkShiftApplication;//sample shader specific data

/*static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
	VkDebugReportFlagsEXT                       flags,
	VkDebugReportObjectTypeEXT                  objectType,
	uint64_t                                    object,
	size_t                                      location,
	int32_t                                     messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData) {

	printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

	return VK_FALSE;
}*/

VkResult CreateDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	//pointer to the function, as it is not part of the core. Function creates debugging messenger
	PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkGPU->instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != NULL) {
		return func(vkGPU->instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}
void DestroyDebugUtilsMessengerEXT(VkGPU* vkGPU, const VkAllocationCallbacks* pAllocator) {
	//pointer to the function, as it is not part of the core. Function destroys debugging messenger
	PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkGPU->instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != NULL) {
		func(vkGPU->instance, vkGPU->debugMessenger, pAllocator);
	}
}
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	printf("validation layer: %s\n", pCallbackData->pMessage);
	return VK_FALSE;
}


VkResult setupDebugMessenger(VkGPU* vkGPU) {
	//function that sets up the debugging messenger 
	if (enableValidationLayers == 0) return VK_SUCCESS;

	VkDebugUtilsMessengerCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;

	if (CreateDebugUtilsMessengerEXT(vkGPU, &createInfo, NULL, &vkGPU->debugMessenger) != VK_SUCCESS) {
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	return VK_SUCCESS;
}
VkResult checkValidationLayerSupport() {
	//check if validation layers are supported when an instance is created
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, NULL);

	VkLayerProperties* availableLayers = (VkLayerProperties*)malloc(sizeof(VkLayerProperties) * layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

	for (uint32_t i = 0; i < layerCount; i++) {
		if (strcmp("VK_LAYER_KHRONOS_validation", availableLayers[i].layerName) == 0) {
			free(availableLayers);
			return VK_SUCCESS;
		}
	}
	free(availableLayers);
	return VK_ERROR_LAYER_NOT_PRESENT;
}

std::vector<const char*> getRequiredExtensions() {
	std::vector<const char*> extensions;

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	extensions.push_back("VK_KHR_get_physical_device_properties2");

	return extensions;
}

VkResult createInstance(VkGPU* vkGPU) {
	//create instance - a connection between the application and the Vulkan library 
	VkResult res = VK_SUCCESS;
	//check if validation layers are supported
	if (enableValidationLayers == 1) {
		res = checkValidationLayerSupport();
		if (res != VK_SUCCESS) return res;
	}

	VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	applicationInfo.pApplicationName = "VkFFT";
	applicationInfo.applicationVersion = 1.0;
	applicationInfo.pEngineName = "VkFFT";
	applicationInfo.engineVersion = 1.0;
	applicationInfo.apiVersion = VK_API_VERSION_1_1;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = &applicationInfo;

	auto extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	if (enableValidationLayers) {
		//query for the validation layer support in the instance
		createInfo.enabledLayerCount = 1;
		const char* validationLayers = "VK_LAYER_KHRONOS_validation";
		createInfo.ppEnabledLayerNames = &validationLayers;
		debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugCreateInfo.pfnUserCallback = debugCallback;
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
	}
	else {
		createInfo.enabledLayerCount = 0;

		createInfo.pNext = nullptr;
	}

	res = vkCreateInstance(&createInfo, NULL, &vkGPU->instance);
	if (res != VK_SUCCESS) return res;

	return res;
}

VkResult findPhysicalDevice(VkGPU* vkGPU) {
	//check if there are GPUs that support Vulkan and select one
	VkResult res = VK_SUCCESS;
	uint32_t deviceCount;
	res = vkEnumeratePhysicalDevices(vkGPU->instance, &deviceCount, NULL);
	if (res != VK_SUCCESS) return res;
	if (deviceCount == 0) {
		return VK_ERROR_DEVICE_LOST;
	}

	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	res = vkEnumeratePhysicalDevices(vkGPU->instance, &deviceCount, devices);
	if (res != VK_SUCCESS) return res;
	vkGPU->physicalDevice = devices[vkGPU->device_id];
	free(devices);
	return VK_SUCCESS;
}
VkResult devices_list() {
	//this function creates an instance and prints the list of available devices
	VkResult res = VK_SUCCESS;
	VkInstance local_instance = { 0 };
	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.flags = 0;
	createInfo.pApplicationInfo = NULL;
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	createInfo.enabledLayerCount = 0;
	createInfo.enabledExtensionCount = 0;
	createInfo.pNext = NULL;
	res = vkCreateInstance(&createInfo, NULL, &local_instance);
	if (res != VK_SUCCESS) return res;

	uint32_t deviceCount;
	res = vkEnumeratePhysicalDevices(local_instance, &deviceCount, NULL);
	if (res != VK_SUCCESS) return res;

	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	res = vkEnumeratePhysicalDevices(local_instance, &deviceCount, devices);
	if (res != VK_SUCCESS) return res;
	for (uint32_t i = 0; i < deviceCount; i++) {
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(devices[i], &device_properties);
		printf("Device id: %d name: %s API:%d.%d.%d\n", i, device_properties.deviceName, (device_properties.apiVersion >> 22), ((device_properties.apiVersion >> 12) & 0x3ff), (device_properties.apiVersion & 0xfff));
	}
	free(devices);
	vkDestroyInstance(local_instance, NULL);
	return res;
}
VkResult getComputeQueueFamilyIndex(VkGPU* vkGPU) {
	//find a queue family for a selected GPU, select the first available for use
	uint32_t queueFamilyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(vkGPU->physicalDevice, &queueFamilyCount, NULL);

	VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(vkGPU->physicalDevice, &queueFamilyCount, queueFamilies);
	uint32_t i = 0;
	for (; i < queueFamilyCount; i++) {
		VkQueueFamilyProperties props = queueFamilies[i];

		if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			break;
		}
	}
	free(queueFamilies);
	if (i == queueFamilyCount) {
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	vkGPU->queueFamilyIndex = i;
	return VK_SUCCESS;
}

VkResult createDevice(VkGPU* vkGPU) {
	//create logical device representation
	VkResult res = VK_SUCCESS;
	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	res = getComputeQueueFamilyIndex(vkGPU);
	if (res != VK_SUCCESS) return res;
	queueCreateInfo.queueFamilyIndex = vkGPU->queueFamilyIndex;
	queueCreateInfo.queueCount = 1; 
	float queuePriorities = 1.0;
	queueCreateInfo.pQueuePriorities = &queuePriorities;
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceFeatures.shaderFloat64 = true;
	VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
	VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16 = {};
	shaderFloat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
	shaderFloat16.shaderFloat16 = true;
	shaderFloat16.shaderInt8 = true;
	deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	deviceFeatures2.pNext = &shaderFloat16;
	deviceFeatures2.features = deviceFeatures;
	vkGetPhysicalDeviceFeatures2(vkGPU->physicalDevice, &deviceFeatures2);
	deviceCreateInfo.pNext = &deviceFeatures2;
	vkGPU->enabledDeviceExtensions.push_back("VK_KHR_16bit_storage");
	deviceCreateInfo.enabledExtensionCount = vkGPU->enabledDeviceExtensions.size();
	deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pEnabledFeatures = NULL;
	res = vkCreateDevice(vkGPU->physicalDevice, &deviceCreateInfo, NULL, &vkGPU->device);
	if (res != VK_SUCCESS) return res;
	vkGetDeviceQueue(vkGPU->device, vkGPU->queueFamilyIndex, 0, &vkGPU->queue);
		
	return res;
}
VkResult createFence(VkGPU* vkGPU) {
	//create fence for synchronization 
	VkResult res = VK_SUCCESS;
	VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	fenceCreateInfo.flags = 0;
	res = vkCreateFence(vkGPU->device, &fenceCreateInfo, NULL, &vkGPU->fence);
	return res;
}
VkResult createCommandPool(VkGPU* vkGPU) {
	//create a place, command buffer memory is allocated from
	VkResult res = VK_SUCCESS;
	VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = vkGPU->queueFamilyIndex;
	res = vkCreateCommandPool(vkGPU->device, &commandPoolCreateInfo, NULL, &vkGPU->commandPool);
	return res;
}

VkResult findMemoryType(VkGPU* vkGPU, uint32_t memoryTypeBits, uint32_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex) {
	VkPhysicalDeviceMemoryProperties memoryProperties = { 0 };

	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
		if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) && (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size >= memorySize))
		{
			memoryTypeIndex[0] = i;
			return VK_SUCCESS;
		}
	}
	return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult allocateFFTBuffer(VkGPU* vkGPU, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
	//allocate the buffer used by the GPU with specified properties
	VkResult res = VK_SUCCESS;
	uint32_t queueFamilyIndices;
	VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.queueFamilyIndexCount = 1;
	bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usageFlags;
	res = vkCreateBuffer(vkGPU->device, &bufferCreateInfo, NULL, buffer);
	if (res != VK_SUCCESS) return res;
	VkMemoryRequirements memoryRequirements = { 0 };
	vkGetBufferMemoryRequirements(vkGPU->device, buffer[0], &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	res = findMemoryType(vkGPU, memoryRequirements.memoryTypeBits, memoryRequirements.size, propertyFlags, &memoryAllocateInfo.memoryTypeIndex);
	if (res != VK_SUCCESS) return res;
	res = vkAllocateMemory(vkGPU->device, &memoryAllocateInfo, NULL, deviceMemory);
	if (res != VK_SUCCESS) return res;
	res = vkBindBufferMemory(vkGPU->device, buffer[0], deviceMemory[0], 0);
	if (res != VK_SUCCESS) return res;
	return res;
}
VkResult transferDataFromCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
	//a function that transfers data from the CPU to the GPU using staging buffer, because the GPU memory is not host-coherent
	VkResult res = VK_SUCCESS;
	VkDeviceSize stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	res = allocateFFTBuffer(vkGPU, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (res != VK_SUCCESS) return res;
	void* data;
	res = vkMapMemory(vkGPU->device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (res != VK_SUCCESS) return res;
	memcpy(data, arr, stagingBufferSize);
	vkUnmapMemory(vkGPU->device, stagingBufferMemory);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != VK_SUCCESS) return res;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != VK_SUCCESS) return res;
	VkBufferCopy copyRegion = { 0 };
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
	res = vkEndCommandBuffer(commandBuffer);
	if (res != VK_SUCCESS) return res;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return res;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	vkDestroyBuffer(vkGPU->device, stagingBuffer, NULL);
	vkFreeMemory(vkGPU->device, stagingBufferMemory, NULL);
	return res;
}
VkResult transferDataToCPU(VkGPU* vkGPU, void* arr, VkBuffer* buffer, VkDeviceSize bufferSize) {
	//a function that transfers data from the GPU to the CPU using staging buffer, because the GPU memory is not host-coherent
	VkResult res = VK_SUCCESS;
	VkDeviceSize stagingBufferSize = bufferSize;
	VkBuffer stagingBuffer = { 0 };
	VkDeviceMemory stagingBufferMemory = { 0 };
	res = allocateFFTBuffer(vkGPU, &stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);
	if (res != VK_SUCCESS) return res;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = { 0 };
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != VK_SUCCESS) return res;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != VK_SUCCESS) return res;
	VkBufferCopy copyRegion = { 0 };
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = stagingBufferSize;
	vkCmdCopyBuffer(commandBuffer, buffer[0], stagingBuffer, 1, &copyRegion);
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return res;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != VK_SUCCESS) return res;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	void* data;
	res = vkMapMemory(vkGPU->device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
	if (res != VK_SUCCESS) return res;
	memcpy(arr, data, stagingBufferSize);
	vkUnmapMemory(vkGPU->device, stagingBufferMemory);
	vkDestroyBuffer(vkGPU->device, stagingBuffer, NULL);
	vkFreeMemory(vkGPU->device, stagingBufferMemory, NULL);
	return res;
}


static inline void shaderGenShift(VkShiftApplication* app) {
	sprintf(app->code0, "#version 450\n");
	if (app->precision == 2) {
		sprintf(app->code0 + strlen(app->code0), "#extension GL_EXT_shader_16bit_storage : require\n");
	}
	sprintf(app->code0 + strlen(app->code0), "layout (local_size_x = %d, local_size_y = %d, local_size_z = %d) in;\n", app->localSize[0], app->localSize[1], app->localSize[2]);

	char vecType[10];
	switch (app->precision) {
	case 0: {
		sprintf(vecType, "vec2");
		break;
	}
	case 1: {
		sprintf(vecType, "dvec2");
		break;
	}
	case 2: {
		sprintf(vecType, "f16vec2");
		break;
	}
	}
	sprintf(app->code0 + strlen(app->code0), "\
layout(std430, binding = 0) buffer Input\n\
{\n\
	%s inputs[];\n\
};\n\
layout(std430, binding = 1) buffer Output\n\
{\n\
	%s outputs[];\n\
};\n", vecType, vecType);
	sprintf(app->code0 + strlen(app->code0), "\
uint index(uint index_x, uint index_y) {\n\
	return index_x + index_y * %d + gl_GlobalInvocationID.z * %d;\n\
}\n", app->inputStride[0], app->inputStride[2]);
	if (app->r2c)
	{
		sprintf(app->code0 + strlen(app->code0), "\
void main()\n\
{\n\
	if (gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*%d < %d){\n\
			outputs[index(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*%d + %d, %d)] = inputs[index(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y*%d + %d, %d)];\n\
	}\n\
	if ((gl_GlobalInvocationID.y < %d)) return; \n", app->size[0], app->size[1] / 2, app->size[0], app->inputStride[1] - app->size[1]/2, app->inputStride[1], app->size[0], app->size[1] / 2, app->inputStride[1], app->size[1] / 2);
		sprintf(app->code0 + strlen(app->code0), "\
	uint id=index(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);\n\
			uint id_out = index(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y + %d);\n\
		outputs[id_out] = inputs[id];\n\
	}", app->inputStride[1] - app->size[1]);
	}
	else {
		sprintf(app->code0 + strlen(app->code0), "\
void main()\n\
{\n\
	uint id = index(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);\n\
	if ((gl_GlobalInvocationID.x < %d) && (gl_GlobalInvocationID.y < %d)) return;\n\
	uint id_out;\n", app->size[0]/ 2, app->size[1] / 2);
		sprintf(app->code0 + strlen(app->code0), "\
	if ((gl_GlobalInvocationID.x >= %d) && (gl_GlobalInvocationID.y < %d))\n\
		id_out = index(gl_GlobalInvocationID.x + %d, gl_GlobalInvocationID.y);\n", app->size[0] / 2, app->size[1] / 2, app->inputStride[0]- app->size[0]);
		sprintf(app->code0 + strlen(app->code0), "\
	if ((gl_GlobalInvocationID.x >= %d) && (gl_GlobalInvocationID.y >= %d))\n\
		id_out = index(gl_GlobalInvocationID.x + %d, gl_GlobalInvocationID.y + %d);\n", app->size[0] / 2, app->size[1] / 2, app->inputStride[0] - app->size[0], app->inputStride[1] - app->size[1]);
		sprintf(app->code0 + strlen(app->code0), "\
	if ((gl_GlobalInvocationID.x < %d) && (gl_GlobalInvocationID.y >= %d))\n\
		id_out = index(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y + %d);\n\
	outputs[id_out] = inputs[id];\n\
}", app->size[0] / 2, app->size[1] / 2, app->inputStride[1] - app->size[1]);
	}
}

VkResult createShiftApp(VkGPU* vkGPU, VkShiftApplication* app) {
	//create an application interface to Vulkan. This function binds the shader to the compute pipeline, so it can be used as a part of the command buffer later
	VkResult res = VK_SUCCESS;
	//we have two storage buffer objects in one set in one pool
	VkDescriptorPoolSize descriptorPoolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	descriptorPoolSize.descriptorCount = 2;

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	descriptorPoolCreateInfo.poolSizeCount = 1;
	descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;
	descriptorPoolCreateInfo.maxSets = 1;
	res = vkCreateDescriptorPool(vkGPU->device, &descriptorPoolCreateInfo, NULL, &app->descriptorPool);
	if (res != VK_SUCCESS) return res;
	//specify each object from the set as a storage buffer
	const VkDescriptorType descriptorType[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	VkDescriptorSetLayoutBinding* descriptorSetLayoutBindings = (VkDescriptorSetLayoutBinding*)malloc(descriptorPoolSize.descriptorCount * sizeof(VkDescriptorSetLayoutBinding));
	for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {
		descriptorSetLayoutBindings[i].binding = i;
		descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
		descriptorSetLayoutBindings[i].descriptorCount = 1;
		descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	descriptorSetLayoutCreateInfo.bindingCount = descriptorPoolSize.descriptorCount;
	descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
	//create layout
	res = vkCreateDescriptorSetLayout(vkGPU->device, &descriptorSetLayoutCreateInfo, NULL, &app->descriptorSetLayout);
	if (res != VK_SUCCESS) return res;
	free(descriptorSetLayoutBindings);
	//provide the layout with actual buffers and their sizes
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	descriptorSetAllocateInfo.descriptorPool = app->descriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = 1;
	descriptorSetAllocateInfo.pSetLayouts = &app->descriptorSetLayout;
	res = vkAllocateDescriptorSets(vkGPU->device, &descriptorSetAllocateInfo, &app->descriptorSet);
	if (res != VK_SUCCESS) return res;
	for (uint32_t i = 0; i < descriptorPoolSize.descriptorCount; ++i) {


		VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
		if (i == 0) {
			descriptorBufferInfo.buffer = app->inputBuffer[0];
			descriptorBufferInfo.range = app->inputBufferSize;
			descriptorBufferInfo.offset = 0;
		}
		if (i == 1) {
			descriptorBufferInfo.buffer = app->outputBuffer[0];
			descriptorBufferInfo.range = app->outputBufferSize;
			descriptorBufferInfo.offset = 0;
		}

		VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		writeDescriptorSet.dstSet = app->descriptorSet;
		writeDescriptorSet.dstBinding = i;
		writeDescriptorSet.dstArrayElement = 0;
		writeDescriptorSet.descriptorType = descriptorType[i];
		writeDescriptorSet.descriptorCount = 1;
		writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
		vkUpdateDescriptorSets(vkGPU->device, 1, &writeDescriptorSet, 0, NULL);
	}



	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &app->descriptorSetLayout;
	//specify how many push constants can be specified when the pipeline is bound to the command buffer
	VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
	pushConstantRange.offset = 0;
	pushConstantRange.size = 0;
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
	//create pipeline layout
	res = vkCreatePipelineLayout(vkGPU->device, &pipelineLayoutCreateInfo, NULL, &app->pipelineLayout);
	if (res != VK_SUCCESS) return res;
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };

	VkComputePipelineCreateInfo computePipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	//specify specialization constants - structure that sets constants in the shader after first compilation (done by glslangvalidator, for example) but before final shader module creation
	//first three values - workgroup dimensions 

	pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	//create a shader module from the byte code
	uint32_t filelength;
	app->code0 = (char*)malloc(sizeof(char) * 100000);
	shaderGenShift(app);
	//printf("%s\n", app->code0);
	const glslang_resource_t default_resource = {
		/* .MaxLights = */ 32,
		/* .MaxClipPlanes = */ 6,
		/* .MaxTextureUnits = */ 32,
		/* .MaxTextureCoords = */ 32,
		/* .MaxVertexAttribs = */ 64,
		/* .MaxVertexUniformComponents = */ 4096,
		/* .MaxVaryingFloats = */ 64,
		/* .MaxVertexTextureImageUnits = */ 32,
		/* .MaxCombinedTextureImageUnits = */ 80,
		/* .MaxTextureImageUnits = */ 32,
		/* .MaxFragmentUniformComponents = */ 4096,
		/* .MaxDrawBuffers = */ 32,
		/* .MaxVertexUniformVectors = */ 128,
		/* .MaxVaryingVectors = */ 8,
		/* .MaxFragmentUniformVectors = */ 16,
		/* .MaxVertexOutputVectors = */ 16,
		/* .MaxFragmentInputVectors = */ 15,
		/* .MinProgramTexelOffset = */ -8,
		/* .MaxProgramTexelOffset = */ 7,
		/* .MaxClipDistances = */ 8,
		/* .MaxComputeWorkGroupCountX = */ 65535,
		/* .MaxComputeWorkGroupCountY = */ 65535,
		/* .MaxComputeWorkGroupCountZ = */ 65535,
		/* .MaxComputeWorkGroupSizeX = */ 1024,
		/* .MaxComputeWorkGroupSizeY = */ 1024,
		/* .MaxComputeWorkGroupSizeZ = */ 64,
		/* .MaxComputeUniformComponents = */ 1024,
		/* .MaxComputeTextureImageUnits = */ 16,
		/* .MaxComputeImageUniforms = */ 8,
		/* .MaxComputeAtomicCounters = */ 8,
		/* .MaxComputeAtomicCounterBuffers = */ 1,
		/* .MaxVaryingComponents = */ 60,
		/* .MaxVertexOutputComponents = */ 64,
		/* .MaxGeometryInputComponents = */ 64,
		/* .MaxGeometryOutputComponents = */ 128,
		/* .MaxFragmentInputComponents = */ 128,
		/* .MaxImageUnits = */ 8,
		/* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
		/* .MaxCombinedShaderOutputResources = */ 8,
		/* .MaxImageSamples = */ 0,
		/* .MaxVertexImageUniforms = */ 0,
		/* .MaxTessControlImageUniforms = */ 0,
		/* .MaxTessEvaluationImageUniforms = */ 0,
		/* .MaxGeometryImageUniforms = */ 0,
		/* .MaxFragmentImageUniforms = */ 8,
		/* .MaxCombinedImageUniforms = */ 8,
		/* .MaxGeometryTextureImageUnits = */ 16,
		/* .MaxGeometryOutputVertices = */ 256,
		/* .MaxGeometryTotalOutputComponents = */ 1024,
		/* .MaxGeometryUniformComponents = */ 1024,
		/* .MaxGeometryVaryingComponents = */ 64,
		/* .MaxTessControlInputComponents = */ 128,
		/* .MaxTessControlOutputComponents = */ 128,
		/* .MaxTessControlTextureImageUnits = */ 16,
		/* .MaxTessControlUniformComponents = */ 1024,
		/* .MaxTessControlTotalOutputComponents = */ 4096,
		/* .MaxTessEvaluationInputComponents = */ 128,
		/* .MaxTessEvaluationOutputComponents = */ 128,
		/* .MaxTessEvaluationTextureImageUnits = */ 16,
		/* .MaxTessEvaluationUniformComponents = */ 1024,
		/* .MaxTessPatchComponents = */ 120,
		/* .MaxPatchVertices = */ 32,
		/* .MaxTessGenLevel = */ 64,
		/* .MaxViewports = */ 16,
		/* .MaxVertexAtomicCounters = */ 0,
		/* .MaxTessControlAtomicCounters = */ 0,
		/* .MaxTessEvaluationAtomicCounters = */ 0,
		/* .MaxGeometryAtomicCounters = */ 0,
		/* .MaxFragmentAtomicCounters = */ 8,
		/* .MaxCombinedAtomicCounters = */ 8,
		/* .MaxAtomicCounterBindings = */ 1,
		/* .MaxVertexAtomicCounterBuffers = */ 0,
		/* .MaxTessControlAtomicCounterBuffers = */ 0,
		/* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
		/* .MaxGeometryAtomicCounterBuffers = */ 0,
		/* .MaxFragmentAtomicCounterBuffers = */ 1,
		/* .MaxCombinedAtomicCounterBuffers = */ 1,
		/* .MaxAtomicCounterBufferSize = */ 16384,
		/* .MaxTransformFeedbackBuffers = */ 4,
		/* .MaxTransformFeedbackInterleavedComponents = */ 64,
		/* .MaxCullDistances = */ 8,
		/* .MaxCombinedClipAndCullDistances = */ 8,
		/* .MaxSamples = */ 4,
		/* .maxMeshOutputVerticesNV = */ 256,
		/* .maxMeshOutputPrimitivesNV = */ 512,
		/* .maxMeshWorkGroupSizeX_NV = */ 32,
		/* .maxMeshWorkGroupSizeY_NV = */ 1,
		/* .maxMeshWorkGroupSizeZ_NV = */ 1,
		/* .maxTaskWorkGroupSizeX_NV = */ 32,
		/* .maxTaskWorkGroupSizeY_NV = */ 1,
		/* .maxTaskWorkGroupSizeZ_NV = */ 1,
		/* .maxMeshViewCountNV = */ 4,
		/* .maxDualSourceDrawBuffersEXT = */ 1,

		/* .limits = */ {
			/* .nonInductiveForLoops = */ 1,
			/* .whileLoops = */ 1,
			/* .doWhileLoops = */ 1,
			/* .generalUniformIndexing = */ 1,
			/* .generalAttributeMatrixVectorIndexing = */ 1,
			/* .generalVaryingIndexing = */ 1,
			/* .generalSamplerIndexing = */ 1,
			/* .generalVariableIndexing = */ 1,
			/* .generalConstantMatrixVectorIndexing = */ 1,
		} };
	glslang_target_client_version_t client_version = (app->precision==2) ? GLSLANG_TARGET_VULKAN_1_1 : GLSLANG_TARGET_VULKAN_1_0;
	glslang_target_language_version_t target_language_version = (app->precision == 2) ? GLSLANG_TARGET_SPV_1_3 : GLSLANG_TARGET_SPV_1_0;
	const glslang_input_t input =
	{
		GLSLANG_SOURCE_GLSL,
		GLSLANG_STAGE_COMPUTE,
		GLSLANG_CLIENT_VULKAN,
		client_version,
		GLSLANG_TARGET_SPV,
		target_language_version,
		app->code0,
		450,
		GLSLANG_NO_PROFILE,
		1,
		0,
		GLSLANG_MSG_DEFAULT_BIT,
		&default_resource,
	};

	glslang_shader_t* shader = glslang_shader_create(&input);
	const char* err;
	if (!glslang_shader_preprocess(shader, &input))
	{
		err = glslang_shader_get_info_log(shader);
		printf("%s\n", app->code0);
		printf("%s\n", err);
		glslang_shader_delete(shader);
		free(app->code0);
		return VK_ERROR_INITIALIZATION_FAILED;
		
	}

	if (!glslang_shader_parse(shader, &input))
	{
		err = glslang_shader_get_info_log(shader);
		printf("%s\n", app->code0);
		printf("%s\n", err);
		glslang_shader_delete(shader);
		free(app->code0);
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	glslang_program_t* program = glslang_program_create();
	glslang_program_add_shader(program, shader);
	if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
	{
		err = glslang_program_get_info_log(program);
		printf("%s\n", app->code0);
		printf("%s\n", err);
		glslang_shader_delete(shader);
		free(app->code0);
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	glslang_program_SPIRV_generate(program, input.stage);

	if (glslang_program_SPIRV_get_messages(program))
	{
		printf("%s", glslang_program_SPIRV_get_messages(program));
	}

	glslang_shader_delete(shader);
	free(app->code0);

	VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	createInfo.pCode = glslang_program_SPIRV_get_ptr(program);
	createInfo.codeSize = glslang_program_SPIRV_get_size(program) * sizeof(uint32_t);
	vkCreateShaderModule(vkGPU->device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);

	pipelineShaderStageCreateInfo.pName = "main";
	pipelineShaderStageCreateInfo.pSpecializationInfo = 0;// &specializationInfo;
	computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
	computePipelineCreateInfo.layout = app->pipelineLayout;



	vkCreateComputePipelines(vkGPU->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, &app->pipeline);
	vkDestroyShaderModule(vkGPU->device, pipelineShaderStageCreateInfo.module, NULL);
	glslang_program_delete(program);
	return res;
}
void deleteShiftApp(VkGPU* vkGPU, VkShiftApplication* app) {
	//destroy previously allocated resources of the application
	vkDestroyDescriptorPool(vkGPU->device, app->descriptorPool, NULL);
	vkDestroyDescriptorSetLayout(vkGPU->device, app->descriptorSetLayout, NULL);
	vkDestroyPipelineLayout(vkGPU->device, app->pipelineLayout, NULL);
	vkDestroyPipeline(vkGPU->device, app->pipeline, NULL);
}
void appendShiftApp(VkShiftApplication* app, VkCommandBuffer commandBuffer) {
	//this function appends to the command buffer: push constants, binds pipeline, descriptors, the shader's program dispatch call and the barrier between two compute stages to avoid race conditions 
	VkMemoryBarrier memory_barrier = {
				VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
	};
	//bind compute pipeline to the command buffer
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, app->pipeline);
	//bind descriptors to the command buffer
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, app->pipelineLayout, 0, 1, &app->descriptorSet, 0, NULL);
	//record dispatch call to the command buffer - specifies the total amount of workgroups
	vkCmdDispatch(commandBuffer, app->size[0] / app->localSize[0], app->size[1] / app->localSize[1], app->numCoordinates);
	//memory synchronization between two compute dispatches
	vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

}
void performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, uint32_t batch) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	//Record commands batch times. Allows to perform multiple convolutions/transforms in one submit.
	for (uint32_t i = 0; i < batch; i++) {
		VkFFTAppend(app, commandBuffer);
	}
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	auto timeSubmit = std::chrono::system_clock::now();
	vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	//printf("Pure submit execution time per batch: %.3f ms\n", totTime / batch);
	vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
}
double performVulkanUpscale(VkGPU* vkGPU, VkFFTApplication* app_forward, VkShiftApplication* appShift, VkFFTApplication* app_inverse, uint32_t batch) {
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	//Record commands batch times. Allows to perform multiple convolutions/transforms in one submit.
	for (uint32_t i = 0; i < batch; i++) {
		VkFFTAppend(app_forward, commandBuffer);
		appendShiftApp(appShift, commandBuffer);
		VkFFTAppend(app_inverse, commandBuffer);
	}
	vkEndCommandBuffer(commandBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	auto timeSubmit = std::chrono::system_clock::now();
	vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	auto timeEnd = std::chrono::system_clock::now();
	double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
	//printf("Pure submit execution time per batch: %.3f ms\n", totTime / batch);
	vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
	return totTime/batch;
}
VkResult launchResample(VkGPU* vkGPU, char* png_input_name, char* png_output_name, uint32_t upscale, uint32_t precision, uint32_t numIter) {
	//Sample Vulkan project GPU initialization.
	VkResult res = VK_SUCCESS;
	//create instance - a connection between the application and the Vulkan library 
	res = createInstance(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Instance creation failed, error code: %d\n", res);
		return res;
	}
	//set up the debugging messenger 
	res = setupDebugMessenger(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Debug messenger creation failed, error code: %d\n", res);
		return res;
	}
	//check if there are GPUs that support Vulkan and select one
	res = findPhysicalDevice(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Physical device not found, error code: %d\n", res);
		return res;
	}
	//create logical device representation
	res = createDevice(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Device creation failed, error code: %d\n", res);
		return res;
	}
	//create fence for synchronization 
	res = createFence(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Fence creation failed, error code: %d\n", res);
		return res;
	}
	//create a place, command buffer memory is allocated from
	res = createCommandPool(vkGPU);
	if (res != VK_SUCCESS) {
		printf("Fence creation failed, error code: %d\n", res);
		return res;
	}
	vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);

	glslang_initialize_process();//compiler can be initialized before VkFFT
	uint32_t isCompilerInitialized = 1;

	printf("VkResample - FFT based upscaling\n");
	uint32_t complexSize = 2* sizeof(float);
	switch (precision) {
	case 0: {
		complexSize = 2 * sizeof(float);
		break;
	}
	case 1: {
		complexSize = 2 * sizeof(double);
		break;
	}
	case 2: {
		complexSize = 2 * sizeof(half);
		break;
	}
	}

	//Configuration + FFT application .
	VkFFTConfiguration forward_configuration = defaultVkFFTConfiguration;
	VkFFTConfiguration inverse_configuration = defaultVkFFTConfiguration;
	VkFFTApplication app_forward;
	VkFFTApplication app_inverse;

	int width;
	int height;
	int channels;
	unsigned char* png_input = stbi_load(png_input_name, &width, &height, &channels, 3); 
	if (png_input == 0) {
		printf("Image not found\n");
		return VK_INCOMPLETE;
	}
	if ((pow(2, (int)log2(width)) != width) || (pow(2, (int)log2(height)) != height)) {
		printf("VkFFT supports only power of 2 image dimensions for now\n");
		return VK_ERROR_FORMAT_NOT_SUPPORTED;
	}
	//Setting up FFT configuration for forward and inverse FFT.
	switch (vkGPU->physicalDeviceProperties.vendorID) {
	case 0x10DE://NVIDIA
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = true;
		break;
	case 0x8086://INTEL
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = true;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	case 0x1002://AMD
		forward_configuration.coalescedMemory = 32;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 64;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 19;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	default:
		forward_configuration.coalescedMemory = 64;
		forward_configuration.useLUT = false;
		forward_configuration.warpSize = 32;
		forward_configuration.registerBoost = 1;
		forward_configuration.registerBoost4Step = 1;
		forward_configuration.swapTo3Stage4Step = 0;
		forward_configuration.performHalfBandwidthBoost = false;
		break;
	}
	forward_configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
	forward_configuration.size[0] = width; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
	forward_configuration.size[1] = height;
	forward_configuration.size[2] = 1;
	forward_configuration.isInputFormatted = true;
	forward_configuration.inputBufferStride[0] = forward_configuration.size[0];
	forward_configuration.inputBufferStride[1] = forward_configuration.size[1];
	forward_configuration.inputBufferStride[2] = 1;
	forward_configuration.bufferStride[0] = upscale * forward_configuration.size[0];
	forward_configuration.bufferStride[1] = upscale * forward_configuration.size[1];
	forward_configuration.bufferStride[2] = 1;
	forward_configuration.halfPrecision = (precision == 2) ? true : false;
	forward_configuration.doublePrecision = (precision == 1) ? true : false;
	uint32_t temporaryScaleIntel = (vkGPU->physicalDeviceProperties.vendorID == 0x8086) ? 2 : 1;//Temporary measure, until L1 overutilization is enabled
	forward_configuration.performR2C = (forward_configuration.bufferStride[0] > vkGPU->physicalDeviceProperties.limits.maxComputeSharedMemorySize / (complexSize) / temporaryScaleIntel) ? false : true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	forward_configuration.coordinateFeatures = 3; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc). 
	forward_configuration.inverse = false; //Direction of FFT. false - forward, true - inverse.
	forward_configuration.reorderFourStep = true;//set to true if you want data to return to correct layout after FFT. Set to false if you use convolution routine. Requires additional tempBuffer of bufferSize (see below) to do reordering
	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [VkDeviceSize *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [VkDeviceSize *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	forward_configuration.device = &vkGPU->device;
	forward_configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	forward_configuration.fence = &vkGPU->fence;
	forward_configuration.commandPool = &vkGPU->commandPool;
	forward_configuration.physicalDevice = &vkGPU->physicalDevice;
	forward_configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

	//Allocate buffer for the input data.
	VkDeviceSize inputBufferSize = (forward_configuration.performR2C) ? ((uint64_t)forward_configuration.coordinateFeatures) * complexSize * (forward_configuration.size[0] / 2 + 1) * forward_configuration.size[1] * forward_configuration.size[2] : ((uint64_t)forward_configuration.coordinateFeatures) * complexSize * forward_configuration.size[0] * forward_configuration.size[1] * forward_configuration.size[2];
	VkDeviceSize bufferSize = (forward_configuration.performR2C) ? ((uint64_t)forward_configuration.coordinateFeatures) * complexSize * (forward_configuration.bufferStride[0] / 2 + 1) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2] : ((uint64_t)forward_configuration.coordinateFeatures) * complexSize * forward_configuration.bufferStride[0] * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];
	//VkDeviceSize bufferSize = ((uint64_t)forward_configuration.coordinateFeatures) * sizeof(scalar) * 2 * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2];;
	VkBuffer inputBuffer = {};
	VkDeviceMemory inputBufferDeviceMemory = {};
	VkBuffer buffer = {};
	VkDeviceMemory bufferDeviceMemory = {};
	VkBuffer tempBuffer = {};
	VkDeviceMemory tempBufferDeviceMemory = {};
	allocateFFTBuffer(vkGPU, &inputBuffer, &inputBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, inputBufferSize);
	allocateFFTBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	allocateFFTBuffer(vkGPU, &tempBuffer, &tempBufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);

	forward_configuration.buffer = &buffer;
	forward_configuration.tempBuffer = &tempBuffer;
	forward_configuration.inputBuffer = &inputBuffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
	forward_configuration.outputBuffer = &buffer;

	forward_configuration.bufferSize = &bufferSize;
	forward_configuration.tempBufferSize = &bufferSize;
	forward_configuration.inputBufferSize = &inputBufferSize;
	forward_configuration.outputBufferSize = &bufferSize;

	//Now we will create a similar configuration for inverse FFT and change inverse parameter to true.
	inverse_configuration = forward_configuration;
	inverse_configuration.isInputFormatted = false;
	inverse_configuration.inputBuffer = &buffer; //you can specify first buffer to read data from to be different from the buffer FFT is performed on. FFT is still in-place on the second buffer, this is here just for convenience.
	inverse_configuration.inputBufferSize = &bufferSize;
	inverse_configuration.size[0] = inverse_configuration.bufferStride[0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
	inverse_configuration.size[1] = inverse_configuration.bufferStride[1];
	inverse_configuration.size[2] = 1;
	inverse_configuration.inverse = true;
	inverse_configuration.performZeropaddingInput[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
	inverse_configuration.performZeropaddingInput[1] = true;
	if (forward_configuration.performR2C) {
		inverse_configuration.fft_zeropad_left_read[0] = forward_configuration.size[0] / 2;
		inverse_configuration.fft_zeropad_right_read[0] = inverse_configuration.size[0] / 2;
		inverse_configuration.fft_zeropad_left_read[1] = inverse_configuration.size[1] / (2 * upscale);
		inverse_configuration.fft_zeropad_right_read[1] = (2 * upscale - 1) * inverse_configuration.size[1] / (2 * upscale);
	}
	else {
		inverse_configuration.fft_zeropad_left_read[0] = forward_configuration.size[0] / 2;
		inverse_configuration.fft_zeropad_right_read[0] = (2 * upscale - 1) * inverse_configuration.size[0] / (2 * upscale);
		inverse_configuration.fft_zeropad_left_read[1] = inverse_configuration.size[1] / (2 * upscale);
		inverse_configuration.fft_zeropad_right_read[1] = (2 * upscale - 1) * inverse_configuration.size[1] / (2 * upscale);
	}
	inverse_configuration.performZeropaddingInput[2] = false;
	//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
	void* buffer_input_void = (void*)malloc(inputBufferSize);
	switch (precision) {
	case 0: {
		float* buffer_input = (float*)buffer_input_void;
		for (uint32_t i = 0; i < inputBufferSize / sizeof(float); i++)
			buffer_input[i] = 0;
		break;
	}
	case 1: {
		double* buffer_input = (double*)buffer_input_void;
		for (uint32_t i = 0; i < inputBufferSize / sizeof(double); i++)
			buffer_input[i] = 0;
		break;
	}
	case 2: {
		half* buffer_input = (half*)buffer_input_void;
		for (uint32_t i = 0; i < inputBufferSize / sizeof(half); i++)
			buffer_input[i] = 0;
		break;
	}
	}
				
	for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; v++) {
		for (uint32_t k = 0; k < forward_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < forward_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < forward_configuration.size[0]; i++) {
					double temp;
					uint32_t png_id = v+i*channels + j * width*channels;
					temp = (double)(png_input[png_id])-127.5;
					switch (precision) {
					case 0: {
						float* buffer_input = (float*)buffer_input_void;
						if (forward_configuration.performR2C)
							buffer_input[(i + j * forward_configuration.size[0] + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2])] = (float)temp;
						else
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = (float)temp;

						break;
					}
					case 1: {
						double* buffer_input = (double*)buffer_input_void;
						if (forward_configuration.performR2C)
							buffer_input[(i + j * forward_configuration.size[0] + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2])] = (double)temp;
						else
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = (double)temp;

						break;
					}
					case 2: {
						half* buffer_input = (half*)buffer_input_void;
						if (forward_configuration.performR2C)
							buffer_input[(i + j * forward_configuration.size[0] + k * (forward_configuration.size[0] + 2) * forward_configuration.size[1] + v * (forward_configuration.size[0] + 2) * forward_configuration.size[1] * forward_configuration.size[2])] = (half)temp;
						else
							buffer_input[2 * (i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * forward_configuration.size[1] + v * (forward_configuration.size[0]) * forward_configuration.size[1] * forward_configuration.size[2])] = (half)temp;

						break;
					}
					}
					}
			}
		}
	}
	stbi_image_free(png_input);
	//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
	transferDataFromCPU(vkGPU, buffer_input_void, &inputBuffer, inputBufferSize);
	//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	res = initializeVulkanFFT(&app_forward, forward_configuration);
	if (res != VK_SUCCESS) return res;
	res = initializeVulkanFFT(&app_inverse, inverse_configuration);
	if (res != VK_SUCCESS) return res;

	VkShiftApplication appShift = { 0 };
	appShift.r2c = forward_configuration.performR2C;
	appShift.precision = precision;
	if (forward_configuration.performR2C)
		appShift.size[0] = forward_configuration.size[0] / 2;
	else
		appShift.size[0] = forward_configuration.size[0];
	appShift.size[1] = forward_configuration.size[1];
	appShift.size[2] = forward_configuration.size[2];
	appShift.localSize[0] = forward_configuration.coalescedMemory / complexSize;
	appShift.localSize[1] = forward_configuration.coalescedMemory / complexSize;
	if (appShift.localSize[0] < 8) appShift.localSize[0] = 8;
	if (appShift.localSize[1] < 8) appShift.localSize[1] = 8;
	appShift.localSize[2] = 1;
	if (forward_configuration.performR2C) {
		appShift.inputStride[0] = forward_configuration.bufferStride[0] / 2;
		appShift.inputStride[1] = forward_configuration.bufferStride[1];
		appShift.inputStride[2] = (forward_configuration.bufferStride[0] / 2 + 1) * forward_configuration.bufferStride[1];
	}
	else
	{
		appShift.inputStride[0] = forward_configuration.bufferStride[0];
		appShift.inputStride[1] = forward_configuration.bufferStride[1];
		appShift.inputStride[2] = (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1];
	}
	appShift.numCoordinates = channels;
	appShift.inputBuffer = &buffer;
	appShift.inputBufferSize = bufferSize;
	appShift.outputBuffer = &buffer;
	appShift.outputBufferSize = bufferSize;

	createShiftApp(vkGPU, &appShift);
	//Submit FFT+shift+iFFT.

	double totTime = performVulkanUpscale(vkGPU, &app_forward, &appShift, &app_inverse, numIter);
	void* buffer_output_void = (void*)malloc(bufferSize);
				
				
	printf("VkResample %dx upscale: %dx%d to %dx%d Time: %0.3f ms\n", upscale, width, height, upscale * width, upscale * height, totTime);

	//Transfer data from GPU using staging buffer.
	transferDataToCPU(vkGPU, buffer_output_void, &buffer, bufferSize);
	
	bool png_output_name_set = false;
	if (png_output_name == 0) {
		png_output_name_set = true;
		png_output_name = (char*)malloc(50 * sizeof(char));
		sprintf(png_output_name, "%d_%d_upscaled.png", upscale*forward_configuration.size[0], upscale * forward_configuration.size[1]);
	}
	unsigned char* png_output = (unsigned char* ) malloc(upscale * upscale * width * height * channels * sizeof(char));
	for (uint32_t v = 0; v < inverse_configuration.coordinateFeatures; v++) {

		for (uint32_t k = 0; k < inverse_configuration.size[2]; k++) {
			for (uint32_t j = 0; j < inverse_configuration.size[1]; j++) {
				for (uint32_t i = 0; i < inverse_configuration.size[0]; i++) {
					//scalar scale = (scalar) (inverse_configuration.size[0] / forward_configuration.size[0] * inverse_configuration.size[1] / forward_configuration.size[1]);
					
					double temp = 0;// 127.5 + upscale * upscale * (float)(buffer_output[(i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
					switch (precision) {
					case 0: {
						float* buffer_output = (float*)buffer_output_void;
						if (forward_configuration.performR2C)
							temp = 127.5 + upscale * upscale * (double)(buffer_output[(i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
						else
							temp = 127.5 + upscale * upscale * (double)(buffer_output[2*(i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
						break;
					}
					case 1: {
						double* buffer_output = (double*)buffer_output_void;
						if (forward_configuration.performR2C)
							temp = 127.5 + upscale * upscale * (double)(buffer_output[(i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
						else
							temp = 127.5 + upscale * upscale * (double)(buffer_output[2 * (i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
						break;
					}
					case 2: {
						half* buffer_output = (half*)buffer_output_void;
						if (forward_configuration.performR2C)
							temp = 127.5 + upscale * upscale * (double)(buffer_output[(i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0] + 2) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
						else
							temp = 127.5 + upscale * upscale * (double)(buffer_output[2 * (i + j * forward_configuration.bufferStride[0] + k * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] + v * (forward_configuration.bufferStride[0]) * forward_configuration.bufferStride[1] * forward_configuration.bufferStride[2])]);
						break;
					}
					}
					if (temp < 0) temp = 0;
					if (temp > 255) temp = 255.0;
					png_output[v+i*channels+j* upscale * width* channels]= (unsigned char)(temp);
				}
			}
		}
	}
	stbi_write_png(png_output_name, upscale * width, upscale * height, channels, png_output, upscale * width*channels);
	if (png_output_name_set == true) {
		free(png_output_name);
	}
	free(png_output);
	free(buffer_input_void);
	free(buffer_output_void);
	vkDestroyBuffer(vkGPU->device, inputBuffer, NULL);
	vkFreeMemory(vkGPU->device, inputBufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, buffer, NULL);
	vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
	vkDestroyBuffer(vkGPU->device, tempBuffer, NULL);
	vkFreeMemory(vkGPU->device, tempBufferDeviceMemory, NULL);
	deleteVulkanFFT(&app_forward);
	deleteVulkanFFT(&app_inverse);
	deleteShiftApp(vkGPU, &appShift);

	//free(buffer_input);
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
	vkDestroyFence(vkGPU->device, vkGPU->fence, NULL);
	vkDestroyCommandPool(vkGPU->device, vkGPU->commandPool, NULL);
	vkDestroyDevice(vkGPU->device, NULL);
	DestroyDebugUtilsMessengerEXT(vkGPU, NULL);
	vkDestroyInstance(vkGPU->instance, NULL);
	glslang_finalize_process();//destroy compiler after use

	return VK_SUCCESS;
}

bool findFlag(char** start, char** end, const std::string& flag) {
	return (std::find(start, end, flag) != end);
}
char* getFlagValue(char** start, char** end, const std::string& flag)
{
	char** value = std::find(start, end, flag);
	value++;
	if (value != end)
	{
		return *value;
	}
	return 0;
}
int main(int argc, char* argv[])
{
	VkGPU vkGPU = {};
	char* png_input_name=0;
	char* png_output_name=0;
	uint32_t upscale = 1;
	uint32_t precision = 0;
	uint32_t numIter = 1;
	if (findFlag(argv, argv + argc, "-h"))
	{
		//print help
		printf("VkResample v1.0.0 (16-12-2020). Author: Tolmachev Dmitrii\n");
		printf("Works with png images only, for now!\n");
		printf("	-h: print help\n");
		printf("	-devices: print the list of available GPU devices\n");
		printf("	-d X: select GPU device (default 0)\n");
		printf("	-i NAME: specify input png file path (power of 2 dimensions)\n");
		printf("	-o NAME: specify output png file path (default X_X_upscale.png)\n");
		printf("	-u X: specify upscale factor (power of 2, default 1)\n");
		printf("	-p X: specify precision (0 - single, 1 - double, 2 - half, default - single)\n");
		printf("	-n X: specify how many times to perform upscale. This removes dispatch overhead and will show the real application performance (default 1)\n");
		return 0;
	}
	if (findFlag(argv, argv + argc, "-devices"))
	{
		//print device list
		VkResult res = devices_list();
		return res;
	}
	if (findFlag(argv, argv + argc, "-d"))
	{
		//select device_id
		char* value = getFlagValue(argv, argv + argc, "-d");
		if (value != 0) {
			sscanf(value, "%d", &vkGPU.device_id);
		}
		else {
			printf("No device is selected with -d flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-n"))
	{
		char* value = getFlagValue(argv, argv + argc, "-n");
		if (value != 0) {
			sscanf(value, "%d", &numIter);
		}
		else {
			printf("No number is selected with -n flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-p"))
	{
		char* value = getFlagValue(argv, argv + argc, "-p");
		if (value != 0) {
			sscanf(value, "%d", &precision);
		}
		else {
			printf("No precision is selected with -p flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-i"))
	{
		png_input_name = getFlagValue(argv, argv + argc, "-i");

		if (png_input_name==0) {
			printf("No input file is selected with -i flag\n");
			return 1;
		}
	}
	else
	{
		printf("No input file is selected with -i flag\n");
		return 1;
	}

	if (findFlag(argv, argv + argc, "-u"))
	{
		char* value = getFlagValue(argv, argv + argc, "-u");
		if (value != 0) {
			sscanf(value, "%d", &upscale);
		}
		else {
			printf("No proper upscale factor is selected with -u flag, default 1\n");
		}
	}
	else {
		printf("No upscale factor is selected with -u flag, default 1\n");
	}
	if (findFlag(argv, argv + argc, "-o"))
	{
		png_output_name = getFlagValue(argv, argv + argc, "-o");
		
		if (png_output_name == 0){
			printf("No output file is selected with -o flag\n");
			return 1;
		}
	}
	
	launchResample(&vkGPU, png_input_name, png_output_name, upscale, precision, numIter);
	
	return VK_SUCCESS;
	
}
