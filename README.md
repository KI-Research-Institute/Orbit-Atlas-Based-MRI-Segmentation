# Orbit-Atlas-Based-MRI-Segmentation
An atlas-based segmentation of the Orbit T1 MR image

An example from the [IXI dataset](https://brain-development.org/ixi-dataset/)  with selected anatomical landmarks is provided.

To try out the method on your own images, you will need to define four anatomical points on the MRI. 
You can utilize [3D Slicer](https://www.slicer.org/) to that end. 
It is possible to change this list of points or the segmentation in the atlas dataset to modify this code for other organs.

For this implementation you will need to identify the following anatomical landmarks (in this order):
1. The center of the right eyeball
2. The center of the left eyeball
3. The nose bridge
4. Anterior point of the optic chiasm

The landmarks identification does not need to be accurate.
Here are screenshots demonstrating these locations.


<img width="427" height="274" alt="image" src="https://github.com/user-attachments/assets/d868f283-fc9b-4a0b-9030-3e11f31bd7f8" />
<img width="966" height="441" alt="image" src="https://github.com/user-attachments/assets/a8650ef2-ceeb-4275-869a-418398a1ffbf" />
<img width="954" height="732" alt="image" src="https://github.com/user-attachments/assets/d732071f-ce3a-4135-aa90-289e24da8231" />

