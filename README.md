# Teki
![](docs/images/banner.png)


Teki is my Vulkan-based toy-renderer. It is my primary environment for experimenting with real-time rendering techniques and general programming as well. 
### Renderer feature list
* [Global Illumination - Diffuse and Specular](docs/gi.md)
* [Light clustering](docs/clustering.md)  
* [Deferred rendering (texturing) - UV-buffer](docs/deferred.md)    
* [PBR](docs/pbr.md)
* [Shadow mapping for Spot and Point lights - PCSS, PCF](docs/shadows.md)
* [Hi-Z occlusion culling](docs/occlusion_culling.md)
* TAA  
* HBAO  
* Bindless materials   
* HDR pipeline      
* Reverse-Z  
* Frustum culling  
### Core feature list
* Buffer suballocation
* Descriptor buffers
* Asynchronous compute queue
* Parallel command buffer recording
* Block compressed textures support (ktx2)
* HDR cubemap support

Everything is developed and tested on my GTX 1050 Ti.
