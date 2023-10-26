#include <algorithm>

#include "src/rendering/renderer/descriptor_management.h"


DescriptorManager::DescriptorManager(VulkanObjectHandler& vulkanObjectHandler)
{
    m_descriptorBufferAlignment = vulkanObjectHandler.getPhysDevDescBufferProperties().descriptorBufferOffsetAlignment;
    EASSERT((m_descriptorBufferAlignment & (m_descriptorBufferAlignment - 1)) == 0, "App", "Non power of two alignment isn't supported.");
    m_queueFamilyIndices = { vulkanObjectHandler.getGraphicsFamilyIndex(), vulkanObjectHandler.getComputeFamilyIndex() };

    m_device = vulkanObjectHandler.getLogicalDevice();
    createNewDescriptorBuffer(RESOURCE_TYPE);
}
DescriptorManager::~DescriptorManager()
{
    for (auto& descBuf : m_descriptorBuffers)
    {
        vmaClearVirtualBlock(descBuf.memoryProxy);
        vmaDestroyVirtualBlock(descBuf.memoryProxy);
    }
}

void DescriptorManager::cmdSubmitPipelineResources(VkCommandBuffer cb, VkPipelineBindPoint bindPoint, const std::vector<std::reference_wrapper<const ResourceSet>>& resourceSets, const std::vector<uint32_t>& resourceIndices, VkPipelineLayout pipelineLayout)
{
    for (uint32_t i{ 0 }; i < resourceSets.size(); ++i)
    {
        const ResourceSet& currentSet{ resourceSets[i].get() };
        uint32_t currentResIndex{ resourceIndices[i] };
        EASSERT(currentSet.getCopiesCount() > 1 ? (currentResIndex < currentSet.getCopiesCount()) : true, "App", "Resource index is bigger than number of resources in a resource set")

        uint32_t resourceDescBufferIndex{ currentSet.getDescBufferIndex() };
        std::vector<uint32_t>::iterator beginIter{ m_descBuffersBindings.begin() };
        std::vector<uint32_t>::iterator indexIter{ std::find(beginIter, m_descBuffersBindings.end(), resourceDescBufferIndex) };
        if (indexIter == m_descBuffersBindings.end())
        {
            m_descBuffersBindings.push_back(resourceDescBufferIndex);
            beginIter = m_descBuffersBindings.begin();
            indexIter = m_descBuffersBindings.end() - 1;

            m_bufferIndicesToBind.push_back(static_cast<uint32_t>(indexIter - beginIter));
            m_offsetsToSet.push_back(currentSet.getDescriptorSetOffset(currentResIndex));
        }
        else
        {
            m_bufferIndicesToBind.push_back(static_cast<uint32_t>(indexIter - beginIter));
            m_offsetsToSet.push_back(currentSet.getDescriptorSetOffset(currentResIndex));
        }
    }
    uint32_t bufferCount{ static_cast<uint32_t>(m_descBuffersBindings.size()) };

    VkDescriptorBufferBindingInfoEXT* bindingInfos{ new VkDescriptorBufferBindingInfoEXT[bufferCount] };
    for (uint32_t i{ 0 }; i < bufferCount; ++i)
    {
        bindingInfos[i].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
        bindingInfos[i].pNext = nullptr;
        bindingInfos[i].address = m_descriptorBuffers[m_descBuffersBindings[i]].deviceAddress;
        switch (m_descriptorBuffers[m_descBuffersBindings[i]].type)
        {
        case DescriptorManager::RESOURCE_TYPE:
            bindingInfos[i].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
            break;
        case DescriptorManager::SAMPLER_TYPE:
            bindingInfos[i].usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
            break;
        default:
            EASSERT(false, "App", "Unknown descriptor buffer type. Should never happen.");
        }
    }

    lvkCmdBindDescriptorBuffersEXT(cb, bufferCount, bindingInfos);
    lvkCmdSetDescriptorBufferOffsetsEXT(cb, bindPoint, pipelineLayout, 0, m_offsetsToSet.size(), m_bufferIndicesToBind.data(), m_offsetsToSet.data());

    m_descBuffersBindings.clear();
    m_bufferIndicesToBind.clear();
    m_offsetsToSet.clear();
    delete[] bindingInfos;
}

void DescriptorManager::createNewDescriptorBuffer(DescriptorBufferType type)
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


void DescriptorManager::insertResourceSetInBuffer(ResourceSet& resourceSet, bool containsSampledData)
{
    VmaVirtualAllocationCreateInfo allocationCI{};
    allocationCI.size = resourceSet.getDescriptorSetAlignedSize() * resourceSet.getCopiesCount();
    allocationCI.alignment = m_descriptorBufferAlignment;
    allocationCI.flags = VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;

   
    VmaVirtualAllocation allocation{};
    uint32_t bufferIndex{};
    auto bufferIter{ m_descriptorBuffers.begin() };
    VkDeviceSize offset{};
    DescriptorBufferType requiredType{ containsSampledData ? SAMPLER_TYPE : RESOURCE_TYPE };
    while (bufferIter->type == requiredType ? vmaVirtualAllocate(bufferIter->memoryProxy, &allocationCI, &allocation, &offset) == VK_ERROR_OUT_OF_DEVICE_MEMORY : true)
    {
        if (++bufferIter == m_descriptorBuffers.end())
        {
            createNewDescriptorBuffer(requiredType);
            bufferIter = m_descriptorBuffers.end() - 1;
        }
    } 
    if (allocation == VK_NULL_HANDLE)
    {
        EASSERT(false, "App", "Descriptor set allocation failed. || Should never happen.")
    }
    resourceSet.setDescBufferOffset(offset);
    bufferIndex = static_cast<uint32_t>(bufferIter - m_descriptorBuffers.begin());
    m_descriptorSetAllocations.push_back({ allocation, bufferIndex });
    resourceSet.m_allocationIter = --m_descriptorSetAllocations.end();

    std::memcpy(reinterpret_cast<uint8_t*>(bufferIter->descriptorBuffer.getData()) + offset, resourceSet.getResourcePayload(), allocationCI.size);
}

void DescriptorManager::removeResourceSetFromBuffer(std::list<DescriptorAllocation>::const_iterator allocationIter)
{
    vmaVirtualFree(m_descriptorBuffers[allocationIter->bufferIndex].memoryProxy, allocationIter->memProxyAlloc);
    m_descriptorSetAllocations.erase(allocationIter);
}

ResourceSet::ResourceSet(ResourceSet&& srcResourceSet) noexcept
{
    m_layout = srcResourceSet.m_layout;

    m_allocationIter = srcResourceSet.m_allocationIter;

    m_resCopies = srcResourceSet.m_resCopies;
    
    m_resources = srcResourceSet.m_resources;


    m_descSetByteSize = srcResourceSet.m_descSetByteSize;
    m_descSetAlignedByteSize = srcResourceSet.m_descSetAlignedByteSize;

    m_descBufferOffset = srcResourceSet.m_descBufferOffset;
    m_resourcePayload = srcResourceSet.m_resourcePayload;

    m_device = srcResourceSet.m_device;

    srcResourceSet.m_invalid = true;

    m_invalid = false;
}

ResourceSet::~ResourceSet()
{
    if (!m_invalid)
    {
        delete[] m_resourcePayload;
        m_assignedDescriptorManager->removeResourceSetFromBuffer(m_allocationIter);
        vkDestroyDescriptorSetLayout(m_device, m_layout, nullptr);
    }
}

uint32_t ResourceSet::getCopiesCount() const
{
    return m_resCopies;
}

const VkDescriptorSetLayout& ResourceSet::getSetLayout() const
{
    return m_layout;
}

uint32_t ResourceSet::getDescBufferIndex() const
{
    return m_allocationIter->bufferIndex;
}

VkDeviceSize ResourceSet::getDescriptorSetAlignedSize()
{
    return m_descSetAlignedByteSize;
}

void ResourceSet::setDescBufferOffset(VkDeviceSize offset)
{
    m_descBufferOffset = offset;
}

VkDeviceSize ResourceSet::getDescriptorSetOffset(uint32_t resIndex)  const
{
    return m_descBufferOffset + (resIndex * m_descSetAlignedByteSize);
}

const void* ResourceSet::getResourcePayload() const
{
    return m_resourcePayload;
}

uint32_t ResourceSet::getDescriptorTypeSize(VkDescriptorType type) const
{
    switch (type)
    {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
        return m_descriptorBufferProperties->samplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return m_descriptorBufferProperties->combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return m_descriptorBufferProperties->sampledImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return m_descriptorBufferProperties->storageImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return m_descriptorBufferProperties->uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return m_descriptorBufferProperties->storageBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return m_descriptorBufferProperties->inputAttachmentDescriptorSize;
    case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return m_descriptorBufferProperties->accelerationStructureDescriptorSize;
    default:
        EASSERT(false, "App", "Descriptor isn't supported yet")
    }
    return UINT32_MAX;
}