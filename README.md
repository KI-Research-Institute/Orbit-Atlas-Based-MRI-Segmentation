# Orbit Atlas-Based MRI T1 Segmentation

<img width="492" height="519" alt="image" src="https://github.com/user-attachments/assets/a7a11ba6-c348-46e2-8c91-ce2d4576c4d3" />

# The challenge:
Even when legal agreements are on hold, technical teams can move forward. The teams from KI Research Institute and Sheba Medical Center's Orbital Surgery Institute collaborated to implement the necessary infrastructure for a future joint project. During that preparation work, we anticipated a key challenge: a lack of publicly available segmented MRI datasets of the human orbit. Searching the internet, we found only one open segmentation of the right orbit section on a single MRI (T1w) image [1]. A common solution would be to hire a team for manual segmentation, a process that would be time-consuming (months) and expensive.

# Solution: atlas-based segmentation for initial data generation
As an alternative, we have developed an atlas-based segmentation method to semi-automatically generate 70 segmented head MRIs (T1w) of healthy subjects from IXI open dataset [2]. To this end, we have manually placed the single right orbit segmentation of [1] on the right and left head MRI MNI atlas (MNI-ICBM152) [3] using 3D Slicer [4]. Following the placement of few predefined anatomical landmarks, an affine transform was computed between the atlas MRI and each of the 70 subjects using SimpleITK and ANTs [5], [6]. Last, the orbit segments were transformed from the atlas coordinates to the subject's MRI. That way, we have produced the segmentation dataset of 70 healthy subjects in a single day. Later on, we have utilized nnU-Net [7] to train a segmentation network based on the 70 subjects and successfully applied it on Sheba’s Orbital Surgery Institute dataset (see image below). Early results are promising!

# Limitations of this method:

The accuracy of the orbit segmentation is lower than manual annotation. 

Performance drops with abnormal/deformed anatomy.

# References
[1]	J. Barranco et al., “Eye-Opening Advances: Automated 3D Segmentation, Key Biomarkers Extraction, and the First Large-Scale MRI Eye Atlas,” Aug. 19, 2024, bioRxiv. doi: 10.1101/2024.08.15.608051.
[2]	“IXI Dataset – Brain Development.” Accessed: Jul. 13, 2025. [Online]. Available: https://brain-development.org/ixi-dataset/
[3]	V. Fonov, A. Evans, R. McKinstry, C. Almli, and D. Collins, “Unbiased nonlinear average age-appropriate brain templates from birth to adulthood,” NeuroImage, vol. 47, p. S102, Jul. 2009, doi: 10.1016/s1053-8119(09)70884-5.
[4]	“3D Slicer image computing platform,” 3D Slicer. Accessed: Jul. 13, 2025. [Online]. Available: https://slicer.org/
[5]	“SimpleITK - Home.” Accessed: Jul. 13, 2025. [Online]. Available: https://simpleitk.org/
[6]	“ANTs by stnava.” Accessed: Jul. 14, 2025. [Online]. Available: https://stnava.github.io/ANTs/
[7]	“nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation | Nature Methods.” Accessed: Jul. 13, 2025. [Online]. Available: https://www.nature.com/articles/s41592-020-01008-z



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

