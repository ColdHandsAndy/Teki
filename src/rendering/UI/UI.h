#ifndef UI_CLASS_HEADER
#define UI_CLASS_HEADER

#include <imgui.h>
#include <imgui_impl_glfw.h>
//Header from the newer version to support dynamic rendering.
#include "dependencies/include/imgui_impl_vulkan.h"

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/command_management.h"

struct UiData
{
    bool drawBVs{ false };
    bool drawLightProxies{ true };
    bool drawSpaceLines{ false };
    bool hiZvis{ false };
    bool showOBBs{ false };
    int hiZmipLevel{ 0 };
    uint32_t frustumCulledCount{ 0 };
    BufferMapped finalDrawCount;
};

class UI
{
private:
    VkDevice m_device{};
    VkDescriptorPool m_descPool{};

    GLFWwindow* m_window{};

public:
	UI(GLFWwindow* window, VulkanObjectHandler& handler, CommandBufferSet& cmdBufferSet)
	{
        m_device = handler.getLogicalDevice();
        m_window = window;

        VkDescriptorPoolSize poolSizes[]
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 100 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 100 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 100 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 100 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 100 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 100 }
        };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 100 * IM_ARRAYSIZE(poolSizes);
        poolInfo.poolSizeCount = (uint32_t)IM_ARRAYSIZE(poolSizes);
        poolInfo.pPoolSizes = poolSizes;
        vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool);


        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        setStyle(ImGui::GetStyle());

        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.Instance = handler.getInstance();
        initInfo.PhysicalDevice = handler.getPhysicalDevice();
        initInfo.Device = handler.getLogicalDevice();
        initInfo.QueueFamily = handler.getGraphicsFamilyIndex();
        initInfo.Queue = handler.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE);
        initInfo.DescriptorPool = m_descPool;
        initInfo.MinImageCount = 2;
        initInfo.ImageCount = 2;
        initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.ColorAttachmentFormat = VK_FORMAT_B8G8R8A8_UNORM;
        initInfo.UseDynamicRendering = true;
        ImGui_ImplVulkan_Init(&initInfo, 0);

        VkQueue queue{ handler.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE) };

        VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };
            ImGui_ImplVulkan_CreateFontsTexture(cb);
        cmdBufferSet.endRecording(cb);

        VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        ImGui_ImplVulkan_DestroyFontUploadObjects();
	}
	~UI()
	{
        vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
	}

    void startUIPass(VkCommandBuffer cb, VkImageView outputAttachment)
    {
        VkRenderingAttachmentInfo attachmentInfo{};
        attachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentInfo.imageView = outputAttachment;

        VkRenderingInfo renderInfo{};
        renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        int width{};
        int height{};
        glfwGetWindowSize(m_window, &width, &height);
        renderInfo.renderArea = { .offset{0,0}, .extent{.width = static_cast<uint32_t>(width), .height = static_cast<uint32_t>(height)} };
        renderInfo.layerCount = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments = &attachmentInfo;

        vkCmdBeginRendering(cb, &renderInfo);
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
    void endUIPass(VkCommandBuffer cb)
    {
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
        vkCmdEndRendering(cb);
    }

    void begin(std::string name) 
    {
        ImGui::Begin(name.c_str());
    }
    void end()
    {
        ImGui::End();
    }
    void stats(UiData& data, uint32_t drawCount)
    {
        if (ImGui::TreeNode("Stats"))
        {
            ImGui::Text("Frustum culled meshes - %u", data.frustumCulledCount);
            ImGui::Text("Occlusion culled meshes - %u", drawCount - *reinterpret_cast<uint32_t*>(data.finalDrawCount.getData()) - data.frustumCulledCount);
            ImGui::TreePop();
        }
    }
    void lightSettings(UiData& data, LightTypes::PointLight& pLight, LightTypes::SpotLight& sLight)
    {
        if (ImGui::TreeNode("Light settings"))
        {
            ImGui::Checkbox("Light bounding volumes", &data.drawBVs);
            ImGui::Checkbox("Light proxies", &data.drawLightProxies);
            if (ImGui::TreeNode("Spot settings"))
            {
                {
                    static glm::vec3 pos{ 0.0f };
                    bool p = ImGui::DragFloat3("Position", &pos.x, 0.04f);
                    static glm::vec2 anglesAB{ 0.0f };
                    static glm::vec3 dir{ 0.0f, -1.0f, 0.0f };
                    bool d = ImGui::DragFloat2("Direction", &anglesAB.x, 0.5f, -180.0f, 180.0f);
                    static float length{ 1.0f };
                    bool l = ImGui::SliderFloat("Length", &length, 0.0f, 50.0f);
                    static glm::vec3 color{};
                    bool c = ImGui::ColorPicker3("Color", &color.x, ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoAlpha);
                    float power{ 1.0f };
                    bool pw = ImGui::SliderFloat("Power", &power, 0.0f, 30000.0f);
                    static float lightSize{ 0.1f };
                    bool s = ImGui::SliderFloat("Light size", &lightSize, 0.0f, 2.0f);
                    static float cutoff{ 22.5f };
                    bool cf = ImGui::SliderFloat("Cutoff", &cutoff, 0.0f, 90.0f);
                    static float falloff{ 20.0f };
                    bool ff = ImGui::SliderFloat("Falloff", &falloff, 0.0f, 90.0f);

                    if (s)
                        sLight.changeSize(lightSize);
                    if (c)
                        sLight.changeColor(color);
                    if (pw)
                        sLight.changePower(power);
                    if (d)
                        sLight.changeDirection(glm::rotateY(glm::rotateZ(dir, glm::radians(anglesAB.x)), glm::radians(anglesAB.y)));
                    if (l)
                        sLight.changeLength(length);
                    if (p)
                        sLight.changePosition(pos);
                    if (cf)
                        sLight.changeCutoff(glm::radians(cutoff));
                    if (ff)
                        sLight.changeFalloff(glm::radians(falloff));
                }
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Point settings"))
            {
                {
                    static glm::vec3 pos{};
                    bool p = ImGui::DragFloat3("Position", &pos.x, 0.04f);
                    static float radius{};
                    bool r = ImGui::SliderFloat("Radius", &radius, 0.0f, 50.0f);
                    static glm::vec3 color{};
                    bool c = ImGui::ColorPicker3("Color", &color.x, ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoAlpha);
                    float power{ 1.0f };
                    bool pw = ImGui::SliderFloat("Power", &power, 0.0f, 30000.0f);
                    static float lightSize{};
                    bool s = ImGui::SliderFloat("Light size", &lightSize, 0.0f, 2.0f);

                    if (pw)
                        pLight.changePower(power);
                    if (p)
                        pLight.changePosition(pos);
                    if (s)
                        pLight.changeSize(lightSize);
                    if (c)
                        pLight.changeColor(color);
                    if (r)
                        pLight.changeRadius(radius);

                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
    }
    void misc(UiData& data)
    {
        if (ImGui::TreeNode("Miscellanous"))
        {
            ImGui::Checkbox("Space lines", &data.drawSpaceLines);
            ImGui::Checkbox("OBBs", &data.showOBBs);
            ImGui::TreePop();
        }
    }

private:
    void setStyle(ImGuiStyle& style)
    {
        style.WindowRounding = 8.3f;
        style.FrameRounding = 4.3f;
        style.ScrollbarRounding = 0;
        style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    }
};

#endif