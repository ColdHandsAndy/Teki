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

#define DESCRIPTOR_BUFFER_DEFAULT_SIZE 51200 //1024

class ResourceSet;

class DescriptorManager
{
private:
	VkDevice m_device{};

    enum DescriptorBufferType
    {
        RESOURCE_TYPE,
        SAMPLER_TYPE,
        TYPES_MAX_NUM
    };
	struct DescriptorBuffer
	{
		VmaVirtualBlock memoryProxy{};
        BufferBaseHostAccessible descriptorBuffer;
		VkDeviceAddress deviceAddress{};
        DescriptorBufferType type{};
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

    void cmdSubmitPipelineResources(VkCommandBuffer cb, VkPipelineBindPoint bindPoint, const std::vector<std::reference_wrapper<const ResourceSet>>& resourceSets, const std::vector<uint32_t>& resourceIndices, VkPipelineLayout pipelineLayout);

private:
	void createNewDescriptorBuffer(DescriptorBufferType type);

	void insertResourceSetInBuffer(ResourceSet& resourceSet, bool containsSampledData);
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

    struct BindingData
    {
        uint32_t binding{};
        VkDescriptorType type{};
        uint32_t count{};
        VkShaderStageFlags stages{};
        VkDeviceSize inSetOffset{};
    };
    std::vector<BindingData> m_resources{};

    uint32_t m_resCopies{};

    VkDeviceSize m_descSetByteSize{};
    VkDeviceSize m_descSetAlignedByteSize{};

    VkDeviceSize m_descBufferOffset{};
    uint8_t* m_resourcePayload{ nullptr };

    VkDevice m_device{};

    bool m_invalid{ true };

public:
    ResourceSet() = default;
    template<std::ranges::contiguous_range Range1, std::ranges::contiguous_range Range2, std::ranges::contiguous_range Range3>
    ResourceSet(VkDevice device,
        uint32_t resCopies,
        VkDescriptorSetLayoutCreateFlags flags,
        const Range1& bindings,
        const Range2& bindingFlags,
        const Range3& bindingsDescriptorData,
        bool containsSampledData)
    {
        initializeSet(device, resCopies, flags, bindings, bindingFlags, bindingsDescriptorData, containsSampledData);
    }
    ResourceSet(ResourceSet&& srcResourceSet) noexcept;
    ~ResourceSet();

    template<std::ranges::contiguous_range Range1, std::ranges::contiguous_range Range2, std::ranges::contiguous_range Range3>
    void initializeSet(VkDevice device, uint32_t resCopies, VkDescriptorSetLayoutCreateFlags flags, const Range1& bindings, const Range2& bindingFlags, const Range3& bindingsDescriptorData, bool containsSampledData)
    {
        if (!m_invalid)
            return;

        m_device = device;
        m_resCopies = resCopies;

        EASSERT(bindings.size() == bindingsDescriptorData.size(), "App", "Descriptors are not provided for every binding");

        VkDescriptorSetLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        layoutCI.flags = flags;

        m_resources.reserve(bindings.size());
        for (auto it = bindings.begin(); it != bindings.end(); ++it)
        {
            m_resources.push_back(BindingData{ .binding = it->binding, .type = it->descriptorType, .count = it->descriptorCount, .stages = it->stageFlags });
        }

        layoutCI.bindingCount = bindings.size();
        layoutCI.pBindings = bindings.data();

        if (bindingFlags.empty())
        {
            EASSERT(vkCreateDescriptorSetLayout(m_device, &layoutCI, nullptr, &m_layout) == VK_SUCCESS, "Vulkan", "Descriptor set layout creation failed");
        }
        else
        {
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCI{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
            bindingFlagsCI.bindingCount = bindingFlags.size();
            bindingFlagsCI.pBindingFlags = bindingFlags.data();
            layoutCI.pNext = &bindingFlagsCI;
            EASSERT(vkCreateDescriptorSetLayout(m_device, &layoutCI, nullptr, &m_layout) == VK_SUCCESS, "Vulkan", "Descriptor set layout creation failed");
        }


        for (uint32_t i{ 0 }; i < bindings.size(); ++i)
        {
            lvkGetDescriptorSetLayoutBindingOffsetEXT(m_device, m_layout, m_resources[i].binding, &m_resources[i].inSetOffset);
        }
        lvkGetDescriptorSetLayoutSizeEXT(m_device, m_layout, &m_descSetByteSize);

        m_descSetAlignedByteSize = (m_descSetByteSize + (m_assignedDescriptorManager->m_descriptorBufferAlignment - 1)) & ~(m_assignedDescriptorManager->m_descriptorBufferAlignment - 1);
        m_resourcePayload = { new uint8_t[m_descSetAlignedByteSize * m_resCopies] };

        for (uint32_t copyIndex{ 0 }; copyIndex < m_resCopies; ++copyIndex)
        {
            for (uint32_t resourceIndx{ 0 }; resourceIndx < m_resources.size(); ++resourceIndx)
            {
                uint32_t resourceNumToGet{ static_cast<uint32_t>(bindingsDescriptorData[resourceIndx].size() / m_resCopies) };

                VkDescriptorGetInfoEXT descGetInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, .type = m_resources[resourceIndx].type };
                uint32_t descriptorTypeSize{ getDescriptorTypeSize(descGetInfo.type) };

                for (VkDeviceSize descriptorArrayNumber{ 0 }; descriptorArrayNumber < resourceNumToGet; ++descriptorArrayNumber)
                {
                    descGetInfo.data = bindingsDescriptorData[resourceIndx][descriptorArrayNumber + resourceNumToGet * copyIndex];
                    lvkGetDescriptorEXT(m_device, &descGetInfo, descriptorTypeSize, m_resourcePayload + copyIndex * m_descSetAlignedByteSize + m_resources[resourceIndx].inSetOffset + descriptorArrayNumber * descriptorTypeSize);
                }
            }
        }

        m_assignedDescriptorManager->insertResourceSetInBuffer(*this, containsSampledData);

        m_invalid = false;
    }

    uint32_t getCopiesCount() const;
    const VkDescriptorSetLayout& getSetLayout() const;

private:
    uint32_t getDescBufferIndex() const;
    VkDeviceSize getDescriptorSetAlignedSize();
    void setDescBufferOffset(VkDeviceSize offset);
    VkDeviceSize getDescriptorSetOffset(uint32_t frameInFlight)  const;
    const void* getResourcePayload() const;
    uint32_t getDescriptorTypeSize(VkDescriptorType type) const;

    ResourceSet(ResourceSet&) = delete;
    void operator=(ResourceSet&) = delete;

    friend class DescriptorManager;
};

#endif