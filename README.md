# Open3d PointCloud Interpolator
## How to use
### 点云插值
1. 按`I`键插值点云， 会打开`Open3d VisualizerWithEditing`，教程如下：
```aiignore
  -- Editing control
    F            : Enter freeview mode.
    X            : Enter orthogonal view along X axis, press again to flip.
    Y            : Enter orthogonal view along Y axis, press again to flip.
    Z            : Enter orthogonal view along Z axis, press again to flip.
    K            : Lock / unlock camera.
    Ctrl + D     : Downsample point cloud with a voxel grid.
    Ctrl + R     : Reset geometry to its initial state.
    Shift + +/-  : Increase/decrease picked point size..
    Shift + mouse left button   : Pick a point and add in queue.
    Shift + mouse right button  : Remove last picked point from queue.

    -- When camera is locked --
    Mouse left button + drag    : Create a selection rectangle.
    Ctrl + mouse buttons + drag : Hold Ctrl key to draw a selection polygon.
                                  Left mouse button to add point. Right mouse
                                  button to remove point. Release Ctrl key to
                                  close the polygon.
    C                           : Crop the geometry with selection region.
```
2. 首先选择要插值的点云区域，建议选择一个较大的平面区域。在选之前按`K`锁定相机，然后按住`Ctrl`键，用鼠标左键选择一个多边形区域，然后按`C`键裁剪点云，重复上面的操作直到选择到满意的区域，最后按 `Q` 退出。
3. 在第二个弹出的窗口中，选择需要保留的插值区域，选择方法同上。
4. 然后在主界面中就能看到插值后的点云了，点击右侧的`Accept`按钮或者按`A`键保留这次插值结果，按`Decline`按钮或者`D`键取消这次插值结果。
### 点云裁剪
1. 按`C`键裁剪点云，同理用上面的方法选择裁剪区域，最后按`Q`键退出后便能看到裁剪后的结果。 

### P.S.
可以按`Ctrl+Z`键撤销上一次的裁剪操作， 按`Ctrl+Y`键重做上一次的操作。