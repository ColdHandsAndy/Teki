#include <algorithm>

#include "descriptor_management.h"


DescriptorManager::DescriptorManager(VulkanObjectHandler& vulkanObjectHandler)
{
    m_descriptorBufferAlignment = vulkanObjectHandler.getPhysDevDescBufferProperties().descriptorBufferOffsetAlignment;
    m_queueFamilyIndices = { vulkanObjectHandler.getGraphicsFamilyIndex(), vulkanObjectHandler.getComputeFamilyIndex() };

    m_device = vulkanObjectHandler.getLogicalDevice();
    createNewDescriptorBuffer();
}
DescriptorManager::~DescriptorManager()
{
    for (auto& descBuf : m_descriptorBuffers)
    {
        vmaClearVirtualBlock(descBuf.memoryProxy);
        vmaDestroyVirtualBlock(descBuf.memoryProxy);
    }
}


void DescriptorManager::cmdSubmitResource(VkCommandBuffer cb, VkPipelineLayout layout, ResourceSet& resource)
{
    if (!resource.isInitialized())
    {
        insertResourceSetInBuffer(resource);
    }
    uint32_t resourceDescBufferIndex{ resource.getDescBufferIndex() };
    std::vector<uint32_t>::iterator beginIter{ m_descBuffersBindings.begin() };
    std::vector<uint32_t>::iterator indexIter{ std::find(beginIter, m_descBuffersBindings.end(), resourceDescBufferIndex) };
    if (indexIter == m_descBuffersBindings.end())
    {
        m_descBuffersBindings.push_back(resourceDescBufferIndex);
        beginIter = m_descBuffersBindings.begin();
        indexIter = m_descBuffersBindings.end() - 1;

        m_bufferIndicesToBind.push_back(static_cast<uint32_t>(indexIter - beginIter));
        m_offsetsToSet.push_back(resource.getDescriptorSetOffset(0));
    }
    else
    {
        m_bufferIndicesToBind.push_back(static_cast<uint32_t>(indexIter - beginIter));
        m_offsetsToSet.push_back(resource.getDescriptorSetOffset(0));
    }

    uint32_t bufferCount{ static_cast<uint32_t>(m_descBuffersBindings.size()) };

    VkDescriptorBufferBindingInfoEXT* bindingInfos{ new VkDescriptorBufferBindingInfoEXT[bufferCount] };
    for (int i{0}; i < bufferCount; ++i)
    {
        bindingInfos->sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
        bindingInfos->pNext = nullptr;
        bindingInfos->address = m_descriptorBuffers[m_descBuffersBindings[i]].deviceAddress;
        bindingInfos->usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    lvkCmdBindDescriptorBuffersEXT(cb, bufferCount, bindingInfos);
    lvkCmdSetDescriptorBufferOffsetsEXT(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, m_offsetsToSet.size(), m_bufferIndicesToBind.data(), m_offsetsToSet.data());

    m_descBuffersBindings.clear();
    m_bufferIndicesToBind.clear();
    m_offsetsToSet.clear();
    delete[] bindingInfos;
}

void DescriptorManager::createNewDescriptorBuffer()
{
    VkBufferCreateInfo bufferCI{};
    bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCI.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    bufferCI.sharingMode = VK_SHARING_MODE_CONCURRENT;
    bufferCI.queueFamilyIndexCount = m_queueFamilyIndices.size();
    bufferCI.pQueueFamilyIndices = m_queueFamilyIndices.data();

    bufferCI.size = DESCRIPTOR_BUFFER_DEFAULT_SIZE;

    m_descriptorBuffers.push_back(DescriptorBuffer{ VmaVirtualBlock{}, BufferMappable{ bufferCI, Buffer::NULL_FLAG, false, true} });
    DescriptorBuffer& newBuffer{ m_descriptorBuffers.back() };
    VkBufferDeviceAddressInfo addrInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = newBuffer.descriptorBuffer.getBufferHandle() };
    newBuffer.deviceAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
    VmaVirtualBlockCreateInfo virtualBlockCI{};
    virtualBlockCI.size = DESCRIPTOR_BUFFER_DEFAULT_SIZE;
    ASSERT_ALWAYS(vmaCreateVirtualBlock(&virtualBlockCI, &newBuffer.memoryProxy) == VK_SUCCESS, "VMA", "Virtual block creation failed.")
}

void DescriptorManager::insertResourceSetInBuffer(ResourceSet& resourceSet)
{
    resourceSet.initializeSet(m_descriptorBufferAlignment);

    VmaVirtualAllocationCreateInfo allocationCI{};
    allocationCI.size = resourceSet.getDescriptorSetAlignedSize() * resourceSet.getFrameCount();
    allocationCI.alignment = m_descriptorBufferAlignment;
    allocationCI.flags = VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;

   
    VmaVirtualAllocation allocation{};
    uint32_t bufferIndex{};
    auto bufferIter{ m_descriptorBuffers.begin() };
    VkDeviceSize offset{};
    while (vmaVirtualAllocate(bufferIter->memoryProxy, &allocationCI, &allocation, &offset) == VK_ERROR_OUT_OF_DEVICE_MEMORY)
    {
        if (++bufferIter == m_descriptorBuffers.end())
        {
            createNewDescriptorBuffer();
            bufferIter = m_descriptorBuffers.end() - 1;
        }
    } 
    if (allocation == VK_NULL_HANDLE)
    {
        ASSERT_ALWAYS(false, "App", "Descriptor set allocation failed. || Should never happen.")
    }
    resourceSet.setDescBufferOffset(offset);
    bufferIndex = static_cast<uint32_t>(bufferIter - m_descriptorBuffers.begin());
    m_descriptorSetAllocations.push_back({ allocation, bufferIndex });
    resourceSet.m_allocationIter = --m_descriptorSetAllocations.end();

    std::memcpy(bufferIter->descriptorBuffer.getData(), resourceSet.getResourcePayload(), allocationCI.size);
}

void DescriptorManager::removeResourceSetFromBuffer(std::list<DescriptorAllocation>::const_iterator allocationIter)
{
    vmaVirtualFree(m_descriptorBuffers[allocationIter->bufferIndex].memoryProxy, allocationIter->memProxyAlloc);
    m_descriptorSetAllocations.erase(allocationIter);
}



ResourceSet::ResourceSet(VkDevice device, uint32_t setIndex, VkDescriptorSetLayoutCreateFlags flags, std::span<VkDescriptorSetLayoutBinding> bindings, uint32_t frameCopies, std::span<VkDescriptorDataEXT> descriptorData)
{
    m_device = device;
    m_setIndex = setIndex;
    // insert bindings and descriptor data per frame compatibility check
    m_descriptorData.frameCopies = frameCopies;

    uint32_t descriptorsInSetNumber{0};
    for (auto& binding : bindings)
    {
        descriptorsInSetNumber += binding.descriptorCount * frameCopies;
    }
    ASSERT_ALWAYS(descriptorsInSetNumber == descriptorData.size(), "App", "Descriptor data is not compatible with set layout bindings.");

    m_descriptorData.descriptorSetDataPerFrame = { descriptorData.begin(), descriptorData.end() };

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

    ASSERT_DEBUG(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &m_layout) == VK_SUCCESS, "Vulkan", "Descriptor set layout creation failed");

    for (uint32_t i{ 0 }; i < bindings.size(); ++i)
    {
        lvkGetDescriptorSetLayoutBindingOffsetEXT(device, m_layout, m_resources[i].binding, &m_resources[i].inSetOffset);
    }
    lvkGetDescriptorSetLayoutSizeEXT(device, m_layout, &m_descSetByteSize);
}

ResourceSet::~ResourceSet()
{
    if (m_initializationState == true)
    {
        delete[] m_resourcePayload;
        m_assignedDescriptorManager->removeResourceSetFromBuffer(m_allocationIter);
    }
    vkDestroyDescriptorSetLayout(m_device, m_layout, nullptr);
}

uint32_t ResourceSet::getFrameCount() const
{
    return m_descriptorData.frameCopies;
}

uint32_t ResourceSet::getSetIndex() const
{
    return m_setIndex;
}

const VkDescriptorSetLayout& ResourceSet::getSetLayout() const
{
    return m_layout;
}

bool ResourceSet::isInitialized() const
{
    return m_initializationState;
}

void ResourceSet::initializeSet(uint32_t descriptorBufferOffsetAlignment)
{
    m_descSetAlignedByteSize = (m_descSetByteSize + (descriptorBufferOffsetAlignment - 1)) & ~(descriptorBufferOffsetAlignment - 1);
    m_resourcePayload = { new uint8_t[m_descSetAlignedByteSize * m_descriptorData.frameCopies] };

    for (uint32_t frameIndx{ 0 }; frameIndx < m_descriptorData.frameCopies; ++frameIndx)
    {
        uint32_t dataIndex{ frameIndx };
        for (uint32_t resourceIndx{ 0 }; resourceIndx < m_resources.size(); ++resourceIndx)
        {
            VkDescriptorGetInfoEXT descGetInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT };
            descGetInfo.type = m_resources[resourceIndx].type;
            uint32_t descriptorTypeSize{ getDescriptorTypeSize(m_resources[resourceIndx].type)};
            //descriptor info is stored as a payload with "frameCopies" of copies stored in a row(aligned)
            for (VkDeviceSize descriptorArrayNumber{0}; descriptorArrayNumber < m_resources[resourceIndx].count; ++descriptorArrayNumber)
            {
                descGetInfo.data = m_descriptorData.descriptorSetDataPerFrame[dataIndex];
                lvkGetDescriptorEXT(m_device, &descGetInfo, descriptorTypeSize, m_resourcePayload + frameIndx * m_descSetAlignedByteSize + m_resources[resourceIndx].inSetOffset + descriptorArrayNumber * descriptorTypeSize);
                dataIndex += m_descriptorData.frameCopies + frameIndx;
            }
        }
    }
    m_initializationState = true;
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

VkDeviceSize ResourceSet::getDescriptorSetOffset(uint32_t frameInFlight)  const
{
    ASSERT_ALWAYS(m_initializationState, "App", "Resource set is not initialized")
    return m_descBufferOffset + (frameInFlight * m_descSetAlignedByteSize);
}

const void* ResourceSet::getResourcePayload() const
{
    ASSERT_ALWAYS(m_initializationState, "App", "Resource set is not initialized")
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
        ASSERT_ALWAYS(false, "App", "Descriptor isn't supported yet")
    }
    return UINT32_MAX;
}