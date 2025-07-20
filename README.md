# Orbit-Atlas-Based-MRI-Segmentation
An atlas-based segmentation of the Orbit T1 MR image


# Installation guide
1. Download the [orbit atlas](https://drive.google.com/file/d/1nNXVWC9WTn4dlnc7vPMJXNI4aNc8lFc_/view?usp=sharing)
2. Download the example input data from this repository
3. Download/clone the Python code and update the paths to the atlas and the example folder
4. In Python install SimpleITK, pandas, numpy and other relevant packages.
5. Download and copy ANTs binaries from: https://github.com/ANTsX/ANTs/releases
   (ANTs: Advanced Normalization Tools; website and source: https://stnava.github.io/ANTs/)
   You will need to update the path to the binaries in the python code.
7. Execute the Python code, and the result will be saved in a results folder where the input data resides.

The example is based on an image from the [IXI dataset](https://brain-development.org/ixi-dataset/)  with the addition of selected anatomical landmarks.

To try out the method on your images, you will need to define four anatomical points on the MRI. 
You can utilize [3D Slicer](https://www.slicer.org/) to that end. 
It is possible to change this list of points or the segmentation in the atlas dataset to modify this code for other organs.

For this implementation, you will need to identify the following anatomical landmarks (in this order):
1. The center of the right eyeball
2. The center of the left eyeball
3. The nose bridge
4. Anterior point of the optic chiasm

The landmarks identification does not need to be accurate.
Here are screenshots demonstrating these locations.


<img width="427" height="274" alt="image" src="https://github.com/user-attachments/assets/d868f283-fc9b-4a0b-9030-3e11f31bd7f8" />
<img width="966" height="441" alt="image" src="https://github.com/user-attachments/assets/a8650ef2-ceeb-4275-869a-418398a1ffbf" />
<img width="954" height="732" alt="image" src="https://github.com/user-attachments/assets/d732071f-ce3a-4135-aa90-289e24da8231" />

