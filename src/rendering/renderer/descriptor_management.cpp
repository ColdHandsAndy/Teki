#include <algorithm>

#include "src/rendering/renderer/descriptor_management.h"

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
        removeResourceSetFromBuffer(m_allocationIter);
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