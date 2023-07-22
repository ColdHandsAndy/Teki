## Teki's light clustering  

Teki uses light culling algorithm described in [this](https://www.activision.com/cdn/research/2017_Sig_Improved_Culling_final.pdf) Call of Duty presentation.  

#### Example from the renderer  
[![](https://img.youtube.com/vi/Z9wthpAi6As/0.jpg)](https://youtu.be/Z9wthpAi6As)

#### Brief overview of the algorithm.
1. Lights are culled against the view frustum.
![](images/clustering_desc0.png)
2. Remaining lights are sorted by their depth to the camera.
![](images/clustering_desc1.png)
![](images/clustering_desc2.png)
3. Frustum is divided by depth into multiple bins. For each bin the closest and furthest lights to the camera contained in it are found.
![](images/clustering_desc3.png)
4. Lights' bounding volumes are rasterized to reduced viewport(width/8, height/8) with MSAA. Fragment shader writes to the corresponding bit in the tile bit array.  
5. In the main fragment shader, each invocation accesses corresponding bin via linear depth and gets min and max light indices. Then it gets tile bit array using screen coordinates.
![](images/clustering_desc4.png)  
To avoid diverging in the fragment shader we scalarize the loop traversal by doing subgroup operations on min and max indices and light bit arrays.
For example, if we assume that all **red** fragment invocations on the image above belong to the same subgroup, their min index would be **1** and max would be **4**,
subgroup operation on light bit array would give **10010111**. It means that only light **3** is calculated.
