#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"

#include <iostream>
#include <cstdint>
#include <algorithm>
#include <set>
#include <vector>
#include <array>
#include <cassert>
#include <optional>

#include <vulkan/vulkan.h>

#ifdef _DEBUG
VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData);
#endif

VulkanObjectHandler::VulkanObjectHandler(VulkanCreateInfo vulkanCreateInfo)
{
	createVulkanInstance(vulkanCreateInfo);

#ifdef _DEBUG
	m_debugMessenger = setupDebugMessenger(m_instance);
#endif

	createWindowSurface(vulkanCreateInfo.windowPtr);

	createPhysicalDevice(vulkanCreateInfo);

	m_physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	m_physicalDeviceDescriptorBufferProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
	m_physicalDeviceProperties.pNext = &m_physicalDeviceDescriptorBufferProperties;
	vkGetPhysicalDeviceProperties2(m_physicalDevice, &m_physicalDeviceProperties);

	m_physicalDeviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_physicalDeviceFeatures);
	m_physicalDeviceMemoryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
	vkGetPhysicalDeviceMemoryProperties2(m_physicalDevice, &m_physicalDeviceMemoryProperties);

	createLogicalDeviceAndQueues(vulkanCreateInfo);

	createSwapchain(vulkanCreateInfo);

	retrieveSwapchainImagesAndViews();
}

VulkanObjectHandler::~VulkanObjectHandler()
{
	vkDeviceWaitIdle(m_logicalDevice);


	for (auto imageView : m_swapchainImageViews)
	{
		vkDestroyImageView(m_logicalDevice, imageView, nullptr);
	}

	vkDestroySwapchainKHR(m_logicalDevice, m_swapchain, nullptr);

	vkDestroyDevice(m_logicalDevice, nullptr);

	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

#ifdef _DEBUG
	DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
#endif

	vkDestroyInstance(m_instance, nullptr);
}

//Vulkan instance creation
bool layersAreSupported(const std::vector<const char*>& layers);
bool instanceExtensionsAreSupported(const std::vector<const char*>& extensions);

const VkQueue VulkanObjectHandler::getQueue(QueueType type) const
{
	VkQueue retQueue{};
	switch (type)
	{
	case GRAPHICS_QUEUE_TYPE:
		retQueue = m_graphicsQueue;
		break;
	case COMPUTE_QUEUE_TYPE:
		retQueue = m_computeQueue;
		break;
	case TRANSFER_QUEUE_TYPE:
		retQueue = m_transferQueue;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Unknown function type")
	}
	return retQueue;
}

void VulkanObjectHandler::createVulkanInstance(const VulkanCreateInfo& vulkanCreateInfo)
{
	if (!layersAreSupported(vulkanCreateInfo.layers))
	{
		std::cerr << "[Vulkan] : Requested layers are not supported" << std::endl;
		assert(false);
	}

	if (!instanceExtensionsAreSupported(vulkanCreateInfo.instanceExtensions))
	{
		std::cerr << "[Vulkan] : Requested extensions are not supported" << std::endl;
		assert(false);
	}

	VkInstanceCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	if (!vulkanCreateInfo.layers.empty())
	{
		info.enabledLayerCount = static_cast<uint32_t>(vulkanCreateInfo.layers.size());
		info.ppEnabledLayerNames = vulkanCreateInfo.layers.data();
	}
	else
	{
		info.enabledLayerCount = 0;
	}
	if (!vulkanCreateInfo.instanceExtensions.empty())
	{
		info.enabledExtensionCount = static_cast<uint32_t>(vulkanCreateInfo.instanceExtensions.size());
		info.ppEnabledExtensionNames = vulkanCreateInfo.instanceExtensions.data();
	}
	else
	{
		info.enabledExtensionCount = 0;
	}
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.apiVersion = vulkanCreateInfo.apiVersion;
	info.pApplicationInfo = &appInfo;

#ifdef _DEBUG
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
	debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	debugCreateInfo.pfnUserCallback = debugCallback;

	info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
#endif

	VkInstance instance{};
	if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS)
	{
		std::cerr << "[Vulkan] : Instance creation failed" << std::endl;
		assert(false);
	}
	m_instance = instance;
}

//Vulkan surface creation
void VulkanObjectHandler::createWindowSurface(GLFWwindow* window)
{
	VkResult result{ glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) };
	if (result != VK_SUCCESS)
	{
		std::cerr << "[GLFW] : Window surface creation failed" << std::endl;
		std::cerr << result << std::endl;
		assert(false);
	}
}

//Vulkan physical device creation
bool isSwapchainSupportAdequate(VkPhysicalDevice device, VkSurfaceKHR surface);

void VulkanObjectHandler::createPhysicalDevice(const VulkanCreateInfo& vulkanCreateInfo)
{
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);

	if (deviceCount == 0)
	{
		std::cerr << "[Vulkan] : No physical devices found" << std::endl;
		assert(false);
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

	for (auto device : devices)
	{
		if (isDeviceSuitable(device, vulkanCreateInfo))
		{
			if (checkQueueFamilies(device, vulkanCreateInfo) && isSwapchainSupportAdequate(device, m_surface))
			{
				m_physicalDevice = device;
				return;
			}
		}
	}
	std::cerr << "[Vulkan] : No suitable physical devices found" << std::endl;
	assert(false);
}

//Vulkan logical device creation
void VulkanObjectHandler::createLogicalDeviceAndQueues(const VulkanCreateInfo& vulkanCreateInfo)
{
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
	std::array uniqueQueueFamiliesIndices{ m_graphicsQueueFamilyIndex, m_computeQueueFamilyIndex, m_transferQueueFamilyIndex };
	constexpr float queuePriority{ 1.0f };
	constexpr uint32_t queueCount{ 1 };
	for (auto index : uniqueQueueFamiliesIndices)
	{
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = index;
		queueCreateInfo.queueCount = queueCount;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkDeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();

	//TODO: Fix this shit
	//
	VkPhysicalDeviceFeatures2 deviceFeatures{ vulkanCreateInfo.deviceFeaturesToEnable };
	deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	VkPhysicalDeviceSynchronization2Features syncFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, .synchronization2 = true };
	VkPhysicalDeviceDynamicRenderingFeatures drFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, .dynamicRendering = true };
	VkPhysicalDeviceDescriptorBufferFeaturesEXT dbFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT, .descriptorBuffer = true };
	VkPhysicalDeviceBufferDeviceAddressFeatures feturesBufAddr{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, .bufferDeviceAddress = true };
	deviceCreateInfo.pNext = reinterpret_cast<void*>(&deviceFeatures);
	deviceFeatures.pNext = &syncFeatures;
	syncFeatures.pNext = &drFeatures;
	drFeatures.pNext = &dbFeatures;
	dbFeatures.pNext = &feturesBufAddr;
	//


	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(vulkanCreateInfo.deviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = vulkanCreateInfo.deviceExtensions.data();

	if (vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_logicalDevice) != VK_SUCCESS)
	{
		std::cerr << "[Vulkan] : Device could not be created" << std::endl;
		assert(false);
	}
	vkGetDeviceQueue(m_logicalDevice, m_graphicsQueueFamilyIndex, 0, &m_graphicsQueue);
	vkGetDeviceQueue(m_logicalDevice, m_computeQueueFamilyIndex, 0, &m_computeQueue);
	vkGetDeviceQueue(m_logicalDevice, m_transferQueueFamilyIndex, 0, &m_transferQueue);
}

//Vulkan swapchain creation
VkSurfaceFormatKHR getSurfaceFormat(VkPhysicalDevice device, VkSurfaceKHR surface, VkFormat preferredFormat, VkColorSpaceKHR preferredColorspace);
VkPresentModeKHR getPresentMode(VkPhysicalDevice device, VkSurfaceKHR surface, VkPresentModeKHR preferredMode);
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);
void VulkanObjectHandler::createSwapchain(const VulkanCreateInfo& vulkanCreateInfo)
{
	VkSurfaceFormatKHR surfaceFormat{ getSurfaceFormat(m_physicalDevice, m_surface, vulkanCreateInfo.swapchainPreferredFormat, vulkanCreateInfo.swapchainPreferredColorspace) };
	m_swapchainImageFormat = surfaceFormat.format;

	VkPresentModeKHR presentMode{ getPresentMode(m_physicalDevice, m_surface, vulkanCreateInfo.swapchainPreferredPresentMode) };

	VkSurfaceCapabilitiesKHR surfaceCapabilities{};
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfaceCapabilities);

	VkExtent2D swapExtent{ chooseSwapExtent(surfaceCapabilities, vulkanCreateInfo.windowPtr)};
	m_swapchainExtent = swapExtent;

	uint32_t swapchainImageCount{ 
		surfaceCapabilities.minImageCount + 1 <= surfaceCapabilities.maxImageCount || surfaceCapabilities.maxImageCount == 0 
		? 
		surfaceCapabilities.minImageCount + 1 
		: 
		surfaceCapabilities.minImageCount 
	};

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = m_surface;
	createInfo.minImageCount = swapchainImageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = swapExtent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	//App does not allow separate graphics and present queue so we use exclusive queue mode
	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.queueFamilyIndexCount = 0;
	createInfo.pQueueFamilyIndices = nullptr;

	createInfo.preTransform = surfaceCapabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; //Maybe include it into vulkanCreateInfo
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE; //Maybe include it into vulkanCreateInfo
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(m_logicalDevice, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS)
	{
		std::cerr << "[Vulkan] : Swapchain could not be created" << std::endl;
		assert(false);
	}
}

void VulkanObjectHandler::retrieveSwapchainImagesAndViews()
{
	uint32_t imageCount{};
	vkGetSwapchainImagesKHR(m_logicalDevice, m_swapchain, &imageCount, nullptr);
	m_swapchainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(m_logicalDevice, m_swapchain, &imageCount, m_swapchainImages.data());


	m_swapchainImageViews.resize(m_swapchainImages.size());

	VkImageViewCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	createInfo.format = m_swapchainImageFormat;
	createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	createInfo.subresourceRange.baseMipLevel = 0;
	createInfo.subresourceRange.levelCount = 1;
	createInfo.subresourceRange.baseArrayLayer = 0;
	createInfo.subresourceRange.layerCount = 1;

	for (int32_t i = 0; i < m_swapchainImageViews.size(); ++i)
	{
		createInfo.image = m_swapchainImages[i];
		if (vkCreateImageView(m_logicalDevice, &createInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) 
		{
			std::cerr << "[Vulkan] : Swapchain image view " << i << " could not be created" << std::endl;
			assert(false);
		}
	}

}




bool layersAreSupported(const std::vector<const char*>& layers)
{
	if (layers.empty())
		return true;

	uint32_t layerPropCount{};
	vkEnumerateInstanceLayerProperties(&layerPropCount, nullptr);
	std::vector<VkLayerProperties> layerProperties(layerPropCount);
	vkEnumerateInstanceLayerProperties(&layerPropCount, layerProperties.data());
	
	std::set<std::string> requiredLayers{ layers.begin(), layers.end() };

	for (const VkLayerProperties& layerProperty : layerProperties)
	{
		requiredLayers.erase(layerProperty.layerName);
	}

	return requiredLayers.empty();
}

bool instanceExtensionsAreSupported(const std::vector<const char*>& extensions)
{
	if (extensions.empty())
		return true;

	uint32_t extensionsCount{};
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionsCount, nullptr);
	std::vector<VkExtensionProperties> extensionProperties(extensionsCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionsCount, extensionProperties.data());

	std::set<std::string> requiredExtensions{ extensions.begin(), extensions.end() };

	for (const VkExtensionProperties& extensionProperty : extensionProperties)
	{
		requiredExtensions.erase(extensionProperty.extensionName);
	}
	return requiredExtensions.empty();
}

//Validation layers creation functions for debugging Vulkan
#ifdef _DEBUG

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);

VkDebugUtilsMessengerEXT setupDebugMessenger(VkInstance instance)
{
	VkDebugUtilsMessengerEXT debugMessenger{};

	VkDebugUtilsMessengerCreateInfoEXT createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
	
	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) 
	{
		std::cerr << "[Vulkan] : Failed to create debug messenger" << std::endl;
		assert(false);
	}

	return debugMessenger;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) 
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) 
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else 
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) 
	{
		func(instance, debugMessenger, pAllocator);
	}
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) 
{
	std::cerr << "Vulkan validation layer message: \n" << pCallbackData->pMessage << std::endl << std::endl;

	return VK_FALSE;
}

#endif



bool VulkanObjectHandler::isDeviceSuitable(VkPhysicalDevice device, const VulkanCreateInfo& vulkanCreateInfo)
{
	VkPhysicalDeviceProperties2 deviceProperties{};
	deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	vkGetPhysicalDeviceProperties2(device, &deviceProperties);

	VkPhysicalDeviceFeatures2 deviceFeatures{};
	deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	vkGetPhysicalDeviceFeatures2(device, &deviceFeatures);

	bool propertyCompatible{ deviceProperties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU };

	//TODO: Make independent to change by adding VkPhysicalDeviceFeatures to VulkanCreateInfo and iterate with offset
	bool featureCompatible{ 
		deviceFeatures.features.multiDrawIndirect &&
		deviceFeatures.features.drawIndirectFirstInstance &&
		deviceFeatures.features.geometryShader &&
		deviceFeatures.features.tessellationShader &&
		deviceFeatures.features.vertexPipelineStoresAndAtomics &&
		deviceFeatures.features.fragmentStoresAndAtomics &&
		deviceFeatures.features.sparseBinding &&
		deviceFeatures.features.sparseResidencyBuffer &&
		deviceFeatures.features.imageCubeArray &&
		deviceFeatures.features.samplerAnisotropy
	};

	uint32_t extensionCount{};
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
	std::set<std::string> requiredExtensions(vulkanCreateInfo.deviceExtensions.begin(), vulkanCreateInfo.deviceExtensions.end());
	for (const auto& extension : availableExtensions) 
	{
		requiredExtensions.erase(extension.extensionName);
	}
	bool extensionsSupported{ requiredExtensions.empty() };

	return propertyCompatible && featureCompatible && extensionsSupported;
}

bool VulkanObjectHandler::checkQueueFamilies(VkPhysicalDevice device, const VulkanCreateInfo& vulkanCreateInfo)
{
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
	bool requiredGraphicsQueueFamilyIsFound{ false };
	bool requiredComputeQueueFamilyIsFound{ false };
	bool requiredTransferQueueFamilyIsFound{ false };

	typedef std::pair<std::vector<uint32_t>, std::vector<uint32_t>> included_excluded_flags_pair;
	auto queueFamilySuitabilityTest{ [&queueFamilies, queueFamilyCount](bool& queueFamilyIsFound, const included_excluded_flags_pair& requirements, uint32_t& queueFamilyIndex, const uint32_t index)
		{

		if (!queueFamilyIsFound)
		{
			for (uint32_t j{ 0 }; j < requirements.first.size(); ++j)
			{
				if (!(queueFamilies[index].queueFlags & requirements.first[j]))
					return;
			}
			for (uint32_t j{ 0 }; j < requirements.second.size(); ++j)
			{
				if (queueFamilies[index].queueFlags & requirements.second[j])
					return;
			}
			queueFamilyIsFound = true;
			queueFamilyIndex = index;
		}

		}
	};

	VkBool32 graphicsQueueFamilySupportsPresent{ VK_FALSE };
	for (uint32_t i{ 0 }; i < queueFamilyCount; ++i)
	{
		queueFamilySuitabilityTest(requiredGraphicsQueueFamilyIsFound, vulkanCreateInfo.graphicsQueueRequirementsFlags, m_graphicsQueueFamilyIndex, i);
		if (requiredGraphicsQueueFamilyIsFound && graphicsQueueFamilySupportsPresent == VK_FALSE)
		{
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &graphicsQueueFamilySupportsPresent);
			if (graphicsQueueFamilySupportsPresent == VK_FALSE)
				requiredGraphicsQueueFamilyIsFound = false;
		}

		queueFamilySuitabilityTest(requiredComputeQueueFamilyIsFound, vulkanCreateInfo.computeQueueRequirementsFlags, m_computeQueueFamilyIndex, i);

		queueFamilySuitabilityTest(requiredTransferQueueFamilyIsFound, vulkanCreateInfo.transferQueueRequirementsFlags, m_transferQueueFamilyIndex, i);
	}

	return requiredGraphicsQueueFamilyIsFound && requiredComputeQueueFamilyIsFound && requiredTransferQueueFamilyIsFound;
}

bool isSwapchainSupportAdequate(VkPhysicalDevice device, VkSurfaceKHR surface)
{
	std::vector<VkSurfaceFormatKHR> formats{};
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	std::vector<VkPresentModeKHR> presentModes{};
	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	return static_cast<bool>(formatCount) && static_cast<bool>(presentModeCount);
}



VkSurfaceFormatKHR getSurfaceFormat(VkPhysicalDevice device, VkSurfaceKHR surface, VkFormat preferredFormat, VkColorSpaceKHR preferredColorspace)
{
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	std::vector<VkSurfaceFormatKHR> availableFormats(formatCount);
	if (formatCount != 0) 
	{
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, availableFormats.data());
		for (const auto& format : availableFormats)
		{
			if (format.format == preferredFormat && format.colorSpace == preferredColorspace)
			{
				return format;
			}
		}
		std::cout << "[App] : Default format and colorspace were picked" << std::endl;
		return availableFormats[0];
	}
	std::cerr << "[Vulkan] : No surface formats available" << " (shouldn't has happened)" << std::endl;
	assert(false);
	return VkSurfaceFormatKHR{};
}

VkPresentModeKHR getPresentMode(VkPhysicalDevice device, VkSurfaceKHR surface, VkPresentModeKHR preferredMode)
{
	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	std::vector<VkPresentModeKHR> availableModes(presentModeCount);
	if (presentModeCount != 0) 
	{
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, availableModes.data());

		for (const auto& mode : availableModes)
		{
			if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return mode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}
	std::cerr << "[Vulkan] : No present modes available" << " (shouldn't has happened)" << std::endl;
	assert(false);
	return VkPresentModeKHR{};
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) 
	{
		return capabilities.currentExtent;
	}
	else 
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent
		{
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

const VkInstance VulkanObjectHandler::getInstance() const
{
	return m_instance;
}

const VkPhysicalDevice VulkanObjectHandler::getPhysicalDevice() const
{
	return m_physicalDevice;
}

const VkDevice VulkanObjectHandler::getLogicalDevice() const
{
	return m_logicalDevice;
}

const VkPhysicalDeviceLimits& VulkanObjectHandler::getPhysDevLimits() const
{
	return m_physicalDeviceProperties.properties.limits;
}
const VkPhysicalDeviceMemoryProperties& VulkanObjectHandler::getPhysDevMemoryProperties() const
{
	return m_physicalDeviceMemoryProperties.memoryProperties;
}

const VkPhysicalDeviceDescriptorBufferPropertiesEXT& VulkanObjectHandler::getPhysDevDescBufferProperties() const
{
	return m_physicalDeviceDescriptorBufferProperties;
}

const uint32_t VulkanObjectHandler::getGraphicsFamilyIndex() const
{
	return m_graphicsQueueFamilyIndex;
}

const uint32_t VulkanObjectHandler::getComputeFamilyIndex() const
{
	return m_computeQueueFamilyIndex;
}

const uint32_t VulkanObjectHandler::getTransferFamilyIndex() const
{
	return m_transferQueueFamilyIndex;
}

const std::tuple<VkImage, VkImageView, uint32_t> VulkanObjectHandler::getSwapchainImageData(uint32_t index) const
{
	return { m_swapchainImages[index], m_swapchainImageViews[index], index };
}
