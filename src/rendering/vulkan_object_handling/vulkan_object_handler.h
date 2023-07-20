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

	VkSwapchainKHR m_swapchain{};
	VkFormat m_swapchainFormat{};
	VkColorSpaceKHR m_preferredColorspace{};
	VkPresentModeKHR m_prefferedPresentMode{};
	VkExtent2D m_swapchainExtent{};
	std::vector<VkImage> m_swapchainImages{};
	std::vector<VkImageView> m_swapchainImageViews{};

	GLFWwindow* m_window{};

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
	const VkFormat getSwapchainFormat() const { return m_swapchainFormat; };

	bool checkSwapchain(VkResult swapchainOpRes)
	{
		if (swapchainOpRes == VK_ERROR_OUT_OF_DATE_KHR || swapchainOpRes == VK_SUBOPTIMAL_KHR)
		{
			recreateSwapchain();
			return false;
		}
		else if (swapchainOpRes != VK_SUCCESS) 
		{
			EASSERT(false, "Vulkan", "Swapchain operation failed.")
		}
		return true;
	}

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

	void createSwapchain();
	void retrieveSwapchainImagesAndViews();
	void cleanupSwapchain() 
	{
		for (int i{ 0 }; i < m_swapchainImageViews.size(); ++i)
		{
			vkDestroyImageView(m_logicalDevice, m_swapchainImageViews[i], nullptr);
		}

		vkDestroySwapchainKHR(m_logicalDevice, m_swapchain, nullptr);
	}

	void recreateSwapchain()
	{
		int width{ 0 };
		int height{ 0 };
		glfwGetFramebufferSize(m_window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(m_window, &width, &height);
			glfwWaitEvents();
		}
		vkDeviceWaitIdle(m_logicalDevice);

		cleanupSwapchain();

		createSwapchain();
		retrieveSwapchainImagesAndViews();
	}
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

	VkPhysicalDeviceFeatures2 deviceFeaturesToEnable{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.features{
			.imageCubeArray = VK_TRUE,
			.multiDrawIndirect = VK_TRUE,
			.drawIndirectFirstInstance = VK_TRUE,
			.fillModeNonSolid = VK_TRUE,
			.wideLines = VK_TRUE,
			.samplerAnisotropy = VK_TRUE,
			.shaderInt64 = VK_TRUE,
			.shaderInt16 = VK_TRUE
		}
	};
	VkPhysicalDeviceVulkan11Features* vulkan11Features{ new VkPhysicalDeviceVulkan11Features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, 
		.storageBuffer16BitAccess = VK_TRUE,
		.uniformAndStorageBuffer16BitAccess = VK_TRUE,
		.shaderDrawParameters = VK_TRUE} };
	VkPhysicalDeviceVulkan12Features* vulkan12Features{ new VkPhysicalDeviceVulkan12Features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, 
		.storageBuffer8BitAccess = VK_TRUE,
		.uniformAndStorageBuffer8BitAccess = VK_TRUE,
		.shaderInt8 = VK_TRUE,
		.descriptorBindingPartiallyBound = VK_TRUE, 
		.runtimeDescriptorArray = VK_TRUE,
		.bufferDeviceAddress = VK_TRUE} };
	VkPhysicalDeviceVulkan13Features* vulkan13Features{ new VkPhysicalDeviceVulkan13Features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, 
		.synchronization2 = VK_TRUE, 
		.dynamicRendering = VK_TRUE} };
	VkPhysicalDeviceDescriptorBufferFeaturesEXT* dbFeatures{ new VkPhysicalDeviceDescriptorBufferFeaturesEXT{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT, .descriptorBuffer = VK_TRUE} };
	
	VkFormat swapchainPreferredFormat{ VK_FORMAT_B8G8R8A8_UNORM };
	VkColorSpaceKHR swapchainPreferredColorspace{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	VkPresentModeKHR swapchainPreferredPresentMode{ VK_PRESENT_MODE_MAILBOX_KHR };

public:
	VulkanCreateInfo()
	{
		getRequiredLayers(layers);
		getRequiredInstanceExtensions(instanceExtensions);
		getRequiredDeviceExtensions(deviceExtensions);

		deviceFeaturesToEnable.pNext = vulkan11Features;
		vulkan11Features->pNext = vulkan12Features;
		vulkan12Features->pNext = vulkan13Features;
		vulkan13Features->pNext = dbFeatures;
	}

	~VulkanCreateInfo()
	{
		delete vulkan11Features;
		delete vulkan12Features;
		delete vulkan13Features;
		delete dbFeatures;
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