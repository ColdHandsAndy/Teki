#ifndef VULKAN_OBJECTS_HANDLER_CLASS_HEADER
#define VULKAN_OBJECTS_HANDLER_CLASS_HEADER

#include <vulkan/vulkan.h>
#include <vector>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "src/tools/asserter.h"

struct VulkanCreateInfo;

class [[nodiscard]] VulkanObjectHandler
{
private:
	VkInstance m_instance{};
	VkPhysicalDevice m_physicalDevice{};
	VkDevice m_logicalDevice{};


	VkSurfaceKHR m_surface{};

	VkPhysicalDeviceProperties2 m_physicalDeviceProperties{};
	VkPhysicalDeviceFeatures2 m_physicalDeviceFeatures{};
	VkPhysicalDeviceMemoryProperties2 m_physicalDeviceMemoryProperties{};
	VkPhysicalDeviceDescriptorBufferPropertiesEXT m_physicalDeviceDescriptorBufferProperties{};

	uint32_t m_graphicsQueueFamilyIndex{};
	uint32_t m_computeQueueFamilyIndex{};
	uint32_t m_transferQueueFamilyIndex{};

	VkQueue m_graphicsQueue{};
	VkQueue m_computeQueue{};
	VkQueue m_transferQueue{};

	VkSwapchainKHR m_swapchain;
	VkFormat m_swapchainImageFormat;
	VkExtent2D m_swapchainExtent;
	std::vector<VkImage> m_swapchainImages;
	std::vector<VkImageView> m_swapchainImageViews;

#ifdef _DEBUG
	VkDebugUtilsMessengerEXT m_debugMessenger{};
#endif

public:
	VulkanObjectHandler() = delete;
	VulkanObjectHandler(const VulkanCreateInfo& vulkanCreateInfo);

	~VulkanObjectHandler();

	//Place for "get" functions
	//limits, physDev properies, features, queues
	const VkInstance getInstance() const;
	const VkPhysicalDevice getPhysicalDevice() const;
	const VkDevice getLogicalDevice() const;
	const VkPhysicalDeviceLimits& getPhysDevLimits() const;
	const VkPhysicalDeviceMemoryProperties& getPhysDevMemoryProperties() const;
	const VkPhysicalDeviceDescriptorBufferPropertiesEXT& getPhysDevDescBufferProperties() const;
	const uint32_t getGraphicsFamilyIndex() const;
	const uint32_t getComputeFamilyIndex() const;
	const uint32_t getTransferFamilyIndex() const;

	//Temp solution
	const std::tuple<VkImage, VkImageView, uint32_t> getSwapchainImageData(uint32_t index) const;
	const VkSwapchainKHR getSwapchain() const { return m_swapchain; };

	enum QueueType
	{
		GRAPHICS_QUEUE_TYPE,
		COMPUTE_QUEUE_TYPE,
		TRANSFER_QUEUE_TYPE
	};
	const VkQueue getQueue(QueueType type) const;

private:
	void createVulkanInstance(const VulkanCreateInfo& vulkanCreateInfo);

	void createWindowSurface(GLFWwindow* window);

	void createPhysicalDevice(const VulkanCreateInfo& vulkanCreateInfo);
	bool isDeviceSuitable(VkPhysicalDevice device, const VulkanCreateInfo& vulkanCreateInfo);
	bool checkQueueFamilies(VkPhysicalDevice device, const VulkanCreateInfo& vulkanCreateInfo);

	void createLogicalDeviceAndQueues(const VulkanCreateInfo& vulkanCreateInfo);

	void loadFunctions();

	void createSwapchain(const VulkanCreateInfo& vulkanCreateInfo);

	void retrieveSwapchainImagesAndViews();

};



struct VulkanCreateInfo
{
public:
	uint32_t apiVersion{ VK_API_VERSION_1_3 };

	std::vector<const char*> layers{};
	std::vector<const char*> instanceExtensions{};
	std::vector<const char*> deviceExtensions{};

	GLFWwindow* windowPtr{};

	typedef std::pair<std::vector<uint32_t>, std::vector<uint32_t>> included_excluded_flags_pair;
	const included_excluded_flags_pair graphicsQueueRequirementsFlags{ { VK_QUEUE_GRAPHICS_BIT }, { } };
	const included_excluded_flags_pair computeQueueRequirementsFlags{ { VK_QUEUE_COMPUTE_BIT }, { VK_QUEUE_GRAPHICS_BIT } };
	const included_excluded_flags_pair transferQueueRequirementsFlags{ { VK_QUEUE_TRANSFER_BIT }, { VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_COMPUTE_BIT } };

	VkPhysicalDeviceFeatures2 deviceFeaturesToEnable{};
	VkPhysicalDeviceSynchronization2Features* syncFeatures{ new VkPhysicalDeviceSynchronization2Features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, .synchronization2 = true} };
	VkPhysicalDeviceDynamicRenderingFeatures* drFeatures{ new VkPhysicalDeviceDynamicRenderingFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, .dynamicRendering = true} };
	VkPhysicalDeviceDescriptorBufferFeaturesEXT* dbFeatures{ new VkPhysicalDeviceDescriptorBufferFeaturesEXT{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT, .descriptorBuffer = true} };
	VkPhysicalDeviceBufferDeviceAddressFeatures* feturesBufAddr{ new VkPhysicalDeviceBufferDeviceAddressFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, .bufferDeviceAddress = true} };

	VkFormat swapchainPreferredFormat{ VK_FORMAT_B8G8R8A8_SRGB };
	VkColorSpaceKHR swapchainPreferredColorspace{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	VkPresentModeKHR swapchainPreferredPresentMode{ VK_PRESENT_MODE_MAILBOX_KHR };

public:
	VulkanCreateInfo()
	{
		getRequiredLayers(layers);
		getRequiredInstanceExtensions(instanceExtensions);
		getRequiredDeviceExtensions(deviceExtensions);

		deviceFeaturesToEnable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeaturesToEnable.features.multiDrawIndirect = VK_TRUE;
		deviceFeaturesToEnable.features.drawIndirectFirstInstance = VK_TRUE;
		deviceFeaturesToEnable.features.imageCubeArray = VK_TRUE;
		deviceFeaturesToEnable.features.samplerAnisotropy = VK_TRUE;

		deviceFeaturesToEnable.pNext = syncFeatures;
		syncFeatures->pNext = drFeatures;
		drFeatures->pNext = dbFeatures;
		dbFeatures->pNext = feturesBufAddr;
	}

	~VulkanCreateInfo()
	{
		delete syncFeatures;
		delete drFeatures;
		delete dbFeatures;
		delete feturesBufAddr;

	}

private:
	void getRequiredLayers(std::vector<const char*>& layers)
	{
#ifdef _DEBUG
		layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
	}
#define REQUIRED_DEVCIE_EXTENSIONS \
	{ \
	VK_KHR_SWAPCHAIN_EXTENSION_NAME, \
	VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME \
	}
	void getRequiredDeviceExtensions(std::vector<const char*>& extensions)
	{
#ifdef REQUIRED_DEVCIE_EXTENSIONS
		extensions.insert(extensions.end(), REQUIRED_DEVCIE_EXTENSIONS);
#endif
	}
//#define REQUIRED_INSTANCE_EXTENSIONS {}
	void getRequiredInstanceExtensions(std::vector<const char*>& extensions)
	{
		uint32_t glfwExtensionCount{};
		const char** glfwExtensions{ glfwGetRequiredInstanceExtensions(&glfwExtensionCount) };
		extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef REQUIRED_INSTANCE_EXTENSIONS
		extensions.insert(extensions.end(), REQUIRED_INSTANCE_EXTENSIONS);
#endif

#ifdef _DEBUG
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
	}
};

//////////////FUNCTIONS TO LOAD/////////////////

inline PFN_vkGetDescriptorSetLayoutSizeEXT lvkGetDescriptorSetLayoutSizeEXT;
inline PFN_vkGetDescriptorSetLayoutBindingOffsetEXT lvkGetDescriptorSetLayoutBindingOffsetEXT;
inline PFN_vkCmdBindDescriptorBuffersEXT lvkCmdBindDescriptorBuffersEXT;
inline PFN_vkCmdSetDescriptorBufferOffsetsEXT lvkCmdSetDescriptorBufferOffsetsEXT;
inline PFN_vkGetDescriptorEXT lvkGetDescriptorEXT;
inline PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT lvkCmdBindDescriptorBufferEmbeddedSamplersEXT;

///////////////////////////////////////////////

#endif