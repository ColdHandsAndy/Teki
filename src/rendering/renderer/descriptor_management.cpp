#include <algorithm>

#include "src/rendering/renderer/descriptor_management.h"

ResourceSet::ResourceSet(ResourceSet&& srcResourceSet) noexcept
{
    m_device = srcResourceSet.m_device;

    m_layout = srcResourceSet.m_layout;

    m_allocationIter = srcResourceSet.m_allocationIter;

    m_resCopies = srcResourceSet.m_resCopies;
    
    m_resources = srcResourceSet.m_resources;


    m_descSetByteSize = srcResourceSet.m_descSetByteSize;
    m_descSetAlignedByteSize = srcResourceSet.m_descSetAlignedByteSize;

    m_descBufferOffset = srcResourceSet.m_descBufferOffset;
    m_resourcePayload = srcResourceSet.m_resourcePayload;

    srcResourceSet.m_invalid = true;

    m_invalid = false;
}

ResourceSet::~ResourceSet()
{
    if (!m_invalid)
    {
        delete[] m_resourcePayload;
        m_descManager->removeResourceSetFromBuffer(m_allocationIter);
        vkDestroyDescriptorSetLayout(m_descManager->m_device, m_layout, nullptr);
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
        return m_descManager->m_descriptorBufferProperties->samplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return m_descManager->m_descriptorBufferProperties->combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return m_descManager->m_descriptorBufferProperties->sampledImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return m_descManager->m_descriptorBufferProperties->storageImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return m_descManager->m_descriptorBufferProperties->uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return m_descManager->m_descriptorBufferProperties->storageBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return m_descManager->m_descriptorBufferProperties->inputAttachmentDescriptorSize;
    case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return m_descManager->m_descriptorBufferProperties->accelerationStructureDescriptorSize;
    default:
        EASSERT(false, "App", "Descriptor isn't supported yet")
    }
    return UINT32_MAX;
}

void ResourceSet::insertResourceSetInBuffer(bool containsSampledData)
{
    VmaVirtualAllocationCreateInfo allocationCI{};
    allocationCI.size = getDescriptorSetAlignedSize() * getCopiesCount();
    allocationCI.alignment = m_descManager->m_descriptorBufferAlignment;
    allocationCI.flags = VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;


    VmaVirtualAllocation allocation{};
    uint32_t bufferIndex{};
    auto bufferIter{ m_descManager->m_descriptorBuffers.begin() };
    VkDeviceSize offset{};
    DescriptorBufferType requiredType{ containsSampledData ? SAMPLER_TYPE : RESOURCE_TYPE };
    while (bufferIter->type == requiredType ? vmaVirtualAllocate(bufferIter->memoryProxy, &allocationCI, &allocation, &offset) == VK_ERROR_OUT_OF_DEVICE_MEMORY : true)
    {
        if (++bufferIter == m_descManager->m_descriptorBuffers.end())
        {
            m_descManager->createNewDescriptorBuffer(requiredType);
            bufferIter = m_descManager->m_descriptorBuffers.end() - 1;
        }
    }
    if (allocation == VK_NULL_HANDLE)
    {
        EASSERT(false, "App", "Descriptor set allocation failed. || Should never happen.")
    }
    setDescBufferOffset(offset);
    bufferIndex = static_cast<uint32_t>(bufferIter - m_descManager->m_descriptorBuffers.begin());
    m_descManager->m_descriptorSetAllocations.push_back({ allocation, bufferIndex });
    m_allocationIter = --m_descManager->m_descriptorSetAllocations.end();

    std::memcpy(reinterpret_cast<uint8_t*>(bufferIter->descriptorBuffer.getData()) + offset, getResourcePayload(), allocationCI.size);
}

void ResourceSet::rewriteDescriptor(uint32_t bindingIndex, uint32_t copyIndex, uint32_t arrayIndex, const VkDescriptorDataEXT& descriptorData) const
{
    EASSERT(bindingIndex < m_resources.size(), "App", "Incorrect binding index.");

    VkDescriptorGetInfoEXT descGetInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, .type = m_resources[bindingIndex].type };
    uint32_t descriptorTypeSize{ getDescriptorTypeSize(descGetInfo.type) };
    descGetInfo.data = descriptorData;
    uint64_t payloadOffset{ copyIndex * m_descSetAlignedByteSize + m_resources[bindingIndex].inSetOffset + arrayIndex * descriptorTypeSize };
    lvkGetDescriptorEXT(m_device, &descGetInfo, descriptorTypeSize, m_resourcePayload + payloadOffset);

    std::memcpy(reinterpret_cast<uint8_t*>(m_descManager->m_descriptorBuffers[m_allocationIter->bufferIndex].descriptorBuffer.getData()) + m_descBufferOffset + payloadOffset, m_resourcePayload + payloadOffset, descriptorTypeSize);
}