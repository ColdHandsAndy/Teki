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

#define DESCRIPTOR_BUFFER_DEFAULT_SIZE  51200

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
struct DescriptorAllocation
{
    VmaVirtualAllocation memProxyAlloc{};
    uint32_t bufferIndex{};
};

class DescriptorManager
{
private:
    VkDevice m_device{};

    std::vector<DescriptorBuffer> m_descriptorBuffers{};
    uint32_t m_descriptorBufferAlignment{};
    std::array<uint32_t, 2> m_queueFamilyIndices{};

    inline static thread_local std::vector<uint32_t> m_descBuffersBindings{};
    inline static thread_local std::vector<uint32_t> m_bufferIndicesToBind{};
    inline static thread_local std::vector<VkDeviceSize> m_offsetsToSet{};

    std::list<DescriptorAllocation> m_descriptorSetAllocations{};

    const VkPhysicalDeviceDescriptorBufferPropertiesEXT* m_descriptorBufferProperties{ nullptr };

    void createNewDescriptorBuffer(DescriptorBufferType type)
    {
        VkBufferCreateInfo bufferCI{};
        bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        if (type == RESOURCE_TYPE)
            bufferCI.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        else
            bufferCI.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        bufferCI.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferCI.queueFamilyIndexCount = m_queueFamilyIndices.size();
        bufferCI.pQueueFamilyIndices = m_queueFamilyIndices.data();

        bufferCI.size = DESCRIPTOR_BUFFER_DEFAULT_SIZE;

        m_descriptorBuffers.push_back(DescriptorBuffer{ .memoryProxy = VmaVirtualBlock{}, .descriptorBuffer = BufferBaseHostAccessible{ m_device, bufferCI, BufferBase::NULL_FLAG, false, true}, .type = type });
        DescriptorBuffer& newBuffer{ m_descriptorBuffers.back() };
        VkBufferDeviceAddressInfo addrInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = newBuffer.descriptorBuffer.getBufferHandle() };
        newBuffer.deviceAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
        VmaVirtualBlockCreateInfo virtualBlockCI{};
        virtualBlockCI.size = DESCRIPTOR_BUFFER_DEFAULT_SIZE;
        EASSERT(vmaCreateVirtualBlock(&virtualBlockCI, &newBuffer.memoryProxy) == VK_SUCCESS, "VMA", "Virtual block creation failed.")
    }
    void removeResourceSetFromBuffer(std::list<DescriptorAllocation>::const_iterator allocationIter)
    {
        vmaVirtualFree(m_descriptorBuffers[allocationIter->bufferIndex].memoryProxy, allocationIter->memProxyAlloc);
        m_descriptorSetAllocations.erase(allocationIter);
    }

public:
    DescriptorManager(VulkanObjectHandler& vulkanObjectHandler)
    {
        if (m_device != VK_NULL_HANDLE)
            return;

        m_descriptorBufferAlignment = vulkanObjectHandler.getPhysDevDescBufferProperties().descriptorBufferOffsetAlignment;
        EASSERT((m_descriptorBufferAlignment & (m_descriptorBufferAlignment - 1)) == 0, "App", "Non power of two alignment isn't supported.");
        m_queueFamilyIndices = { vulkanObjectHandler.getGraphicsFamilyIndex(), vulkanObjectHandler.getComputeFamilyIndex() };

        m_device = vulkanObjectHandler.getLogicalDevice();
        m_descriptorBufferProperties = &vulkanObjectHandler.getPhysDevDescBufferProperties();

        createNewDescriptorBuffer(RESOURCE_TYPE);
    }
    ~DescriptorManager()
    {
        for (auto& descBuf : m_descriptorBuffers)
        {
            vmaDestroyVirtualBlock(descBuf.memoryProxy);
        }
        m_descriptorBuffers.clear();
    }

    friend class Pipeline;
    friend class ResourceSet;
};

class ResourceSet
{
private:
    static inline DescriptorManager* m_descManager{};
public:
    static void assignGlobalDescriptorManager(DescriptorManager& descManager)
    {
        m_descManager = &descManager;
    }

private:
    VkDevice m_device{};

    VkDescriptorSetLayout m_layout{};

    std::list<DescriptorAllocation>::const_iterator m_allocationIter{};

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

    uint32_t getCopiesCount() const;
    const VkDescriptorSetLayout& getSetLayout() const;

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

        m_descSetAlignedByteSize = (m_descSetByteSize + (m_descManager->m_descriptorBufferAlignment - 1)) & ~(m_descManager->m_descriptorBufferAlignment - 1);
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

        insertResourceSetInBuffer(containsSampledData);

        m_invalid = false;
    }

private:
    uint32_t getDescBufferIndex() const;
    VkDeviceSize getDescriptorSetAlignedSize();
    void setDescBufferOffset(VkDeviceSize offset);
    VkDeviceSize getDescriptorSetOffset(uint32_t resIndex)  const;
    const void* getResourcePayload() const;
    uint32_t getDescriptorTypeSize(VkDescriptorType type) const;
    void insertResourceSetInBuffer(bool containsSampledData);
    void rewriteDescriptor(uint32_t bindingIndex, uint32_t copyIndex, uint32_t arrayIndex, const VkDescriptorDataEXT& descriptorData) const;

    ResourceSet(ResourceSet&) = delete;
    void operator=(ResourceSet&) = delete;

    friend class DescriptorManager;
    friend class Pipeline;
};

#endif