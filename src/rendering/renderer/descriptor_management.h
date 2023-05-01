#ifndef DESCRIPTOR_MANAGEMENT_HEADER
#define DESCRIPTOR_MANAGEMENT_HEADER

#include <vector>
#include <array>
#include <span>

#include "src/rendering/data_management/buffer_class.h"

extern PFN_vkGetDescriptorSetLayoutSizeEXT lvkGetDescriptorSetLayoutSizeEXT;
extern PFN_vkGetDescriptorSetLayoutBindingOffsetEXT lvkGetDescriptorSetLayoutBindingOffsetEXT;
extern PFN_vkCmdBindDescriptorBuffersEXT lvkCmdBindDescriptorBuffersEXT;
extern PFN_vkCmdSetDescriptorBufferOffsetsEXT lvkCmdSetDescriptorBufferOffsetsEXT;
extern PFN_vkGetDescriptorEXT lvkGetDescriptorEXT;
extern PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT lvkCmdBindDescriptorBufferEmbeddedSamplersEXT;

#define DESCRIPTOR_BUFFER_DEFAULT_SIZE 1024 //51200

class ResourceSet;

class DescriptorManager
{
private:
	VkDevice m_device{};

	struct DescriptorBuffer
	{
		VmaVirtualBlock memoryProxy{};
        BufferBaseHostAccessible descriptorBuffer;
		VkDeviceAddress deviceAddress{};
	};
	std::vector<DescriptorBuffer> m_descriptorBuffers{};
	inline static uint32_t m_descriptorBufferAlignment;
	inline static std::array<uint32_t, 2> m_queueFamilyIndices;

	inline static thread_local std::vector<uint32_t> m_descBuffersBindings{};
	inline static thread_local std::vector<uint32_t> m_bufferIndicesToBind{};
	inline static thread_local std::vector<VkDeviceSize> m_offsetsToSet{};

	//should be protected from concurrent access
	struct DescriptorAllocation
	{
		VmaVirtualAllocation memProxyAlloc{};
		uint32_t bufferIndex{};
	};
	std::list<DescriptorAllocation> m_descriptorSetAllocations{};

public:
	DescriptorManager(VulkanObjectHandler& vulkanObjectHandler);
	~DescriptorManager();

	void cmdSubmitResource(VkCommandBuffer cb, VkPipelineLayout layout, ResourceSet& resource);
    void cmdSubmitPipelineResources(VkCommandBuffer cb, std::vector<ResourceSet>& resourceSets, VkPipelineLayout pipelineLayout, uint32_t frameIndex = 0u);
private:
	void createNewDescriptorBuffer();

	void insertResourceSetInBuffer(ResourceSet& resourceSet);
	void removeResourceSetFromBuffer(std::list<DescriptorAllocation>::const_iterator allocationIter);


	friend class ResourceSet;

	DescriptorManager() = delete;
	DescriptorManager(DescriptorManager&) = delete;
	void operator=(DescriptorManager&) = delete;
};



class ResourceSetSharedData
{
protected:
    inline static DescriptorManager* m_assignedDescriptorManager{ nullptr };
    inline static const VkPhysicalDeviceDescriptorBufferPropertiesEXT* m_descriptorBufferProperties{ nullptr };

    ResourceSetSharedData() = default;
public:
    static void initializeResourceManagement(VulkanObjectHandler& vulkanObjectHandler, DescriptorManager& descriptorManager)
    {
        m_descriptorBufferProperties = &vulkanObjectHandler.getPhysDevDescBufferProperties();
        m_assignedDescriptorManager = &descriptorManager;
    }
};

class ResourceSet : public ResourceSetSharedData
{
private:
    VkDescriptorSetLayout m_layout{};

    struct DescriptorAllocation;
    struct DescriptorBuffer;
    std::list<DescriptorManager::DescriptorAllocation>::const_iterator m_allocationIter{};

    //descriptor data stored as a vector where data for each binding has "frameCopies" of VkDescriptorDataEXT structs
    struct DescriptorData
    {
        uint32_t frameCopies{};
        std::vector<VkDescriptorDataEXT> descriptorSetDataPerFrame{};
    };
    DescriptorData m_descriptorData;

    struct BindingData
    {
        uint32_t binding{};
        VkDescriptorType type{};
        uint32_t count{};
        VkShaderStageFlags stages{};
        VkDeviceSize inSetOffset{};
    };
    std::vector<BindingData> m_resources{};


    VkDeviceSize m_descSetByteSize{};
    VkDeviceSize m_descSetAlignedByteSize{};

    bool m_initializationState{ false };
    VkDeviceSize m_descBufferOffset{};
    uint8_t* m_resourcePayload{ nullptr };

    VkDevice m_device{};
    uint32_t m_setIndex{};

    bool m_invalid{ false };

public:
    ResourceSet(VkDevice device, uint32_t setIndex, VkDescriptorSetLayoutCreateFlags flags, std::span<const VkDescriptorSetLayoutBinding> bindings, uint32_t frameCopies, std::span<const VkDescriptorDataEXT> descriptorData);
    ResourceSet(ResourceSet&& srcResourceSet) noexcept;
    ~ResourceSet();

    uint32_t getFrameCount() const;
    uint32_t getSetIndex() const;
    const VkDescriptorSetLayout& getSetLayout() const;

private:
    bool isInitialized() const;
    void initializeSet(uint32_t descriptorBufferOffsetAlignment);
    uint32_t getDescBufferIndex() const;
    VkDeviceSize getDescriptorSetAlignedSize();
    void setDescBufferOffset(VkDeviceSize offset);
    VkDeviceSize getDescriptorSetOffset(uint32_t frameInFlight)  const;
    const void* getResourcePayload() const;
    uint32_t getDescriptorTypeSize(VkDescriptorType type) const;

    ResourceSet() = delete;
    ResourceSet(ResourceSet&) = delete;
    void operator=(ResourceSet&) = delete;

    friend class DescriptorManager;
};


#endif