import os.path
import pandas as pd
import SimpleITK as sitk
import numpy as np
import subprocess
import csv
import ants
import copy
import shutil


def get_points(points_filename):
    points = []
    print(os.listdir(os.path.dirname(points_filename)))
    with open(points_filename) as file:
        for row in csv.reader(file):
            if row[0][0] != '#': # if not a comment
                current_point = np.array([float(row[1]), float(row[2]), float(row[3])])
                points.append(current_point)

    return np.array(points)

def compute_rotation_translation(pts1, pts2):
    # Check if the input arrays have the correct shape
    assert pts1.shape == pts2.shape, "Point sets must have the same shape"
    assert pts1.shape[1] == 3, "Points must be 3D"

    # Number of points
    n = pts1.shape[0]

    # Compute centroids (mean)
    centroid1 = np.mean(pts1, axis=0)
    centroid2 = np.mean(pts2, axis=0)

    # Center the points
    centered_pts1 = pts1 - centroid1
    centered_pts2 = pts2 - centroid2

    # Compute covariance matrix
    H = np.dot(centered_pts1.T, centered_pts2)

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Handle special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    t = centroid2 - np.dot(R, centroid1)

    return R, t

def apply_transformation(pts, R, t):
    # Apply the rotation and translation to the points
    return np.dot(pts, R.T) + t

def compute_error(transformed_pts, target_pts):
    # Compute the Euclidean distance between the transformed points and the target points
    errors = np.linalg.norm(transformed_pts - target_pts, axis=1)
    return errors, np.mean(errors)

def compute_init_transform(atlas_points_set_filename, patient_points_set_filename):
    atlas_points_set = get_points(atlas_points_set_filename)
    patient_points_set = get_points(patient_points_set_filename)

    R, t = compute_rotation_translation(atlas_points_set, patient_points_set)

    transformed_pts = apply_transformation(atlas_points_set, R, t)

    errors, mean_error = compute_error(atlas_points_set, patient_points_set)
    print('error before registration:')
    print(errors)
    print(mean_error)

    print('error after registration:')
    errors, mean_error = compute_error(transformed_pts, patient_points_set)
    print (errors)
    print (mean_error)

    matrix = R.flatten()
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix)
    transform.SetTranslation(t)

    return transform

def apply_transformation_on_image(moving_atlas_image_filename, fixed_image_filename, transformed_image_filename, transform, interpolation_type = sitk.sitkLinear):
    moving_atlas_image = sitk.ReadImage(moving_atlas_image_filename)
    fixed_image = sitk.ReadImage(fixed_image_filename)
    transformed_image = sitk.Resample(
        moving_atlas_image,  # Input image
        fixed_image,  # Reference image for size, spacing, etc.
        transform.GetInverse(),  # Transform to apply
        interpolation_type,  # Interpolation (linear, nearest, etc.)
        0.0,  # Default pixel value for areas outside the image
        moving_atlas_image.GetPixelID()  # Preserve the pixel type of the input image
    )
    sitk.WriteImage(transformed_image, transformed_image_filename)

def command_iteration(method):
    print(f"Iteration: {method.GetOptimizerIteration()}")
    print(f"Metric Value: {method.GetMetricValue()}")

def crop_image(image, mask):
    """
    Crops the input image based on the non-zero region of the mask.

    Args:
        image (sitk.Image): The input image to crop.
        mask (sitk.Image): The binary mask defining the cropping region.

    Returns:
        sitk.Image: The cropped image.
    """
    # Get the bounding box of the mask
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)
    bounding_box = label_shape_filter.GetBoundingBox(1)  # Assumes label '1' is the mask region

    # Extract the bounding box values
    start_x, start_y, start_z, size_x, size_y, size_z = bounding_box

    # Define the ROI
    start = [start_x, start_y, start_z]
    size = [size_x, size_y, size_z]

    # Crop the image
    cropped_image = sitk.RegionOfInterest(image, size=size, index=start)
    return cropped_image

def compute_patch_registration (fixed_patient_T1_image_filename, mask_filename,
                               patch_filename, segmentation_filename, segmentation_output_name):
    # # apply mask on fixed image
    fixed_patient_T1_image = sitk.ReadImage(fixed_patient_T1_image_filename)
    mask_image = sitk.ReadImage(mask_filename)
    fixed_masked_image_itk = crop_image(fixed_patient_T1_image, mask_image) #sitk.Mask(fixed_patient_T1_image, mask_image)

    fixed_masked_image_itk_filename = os.path.join(os.path.dirname(segmentation_filename),
                                      os.path.basename(fixed_patient_T1_image_filename) +
                                                   '_' + os.path.basename(mask_filename) + '.nii.gz')

    sitk.WriteImage(fixed_masked_image_itk, fixed_masked_image_itk_filename)

    moving_image_patch_itk = crop_image(sitk.ReadImage(patch_filename), mask_image)
    moving_segmentation_patch_itk = sitk.ReadImage(segmentation_filename)

    moving_patch_itk_filename = os.path.join(os.path.dirname(patch_filename),
                                                   os.path.basename(patch_filename) + '_masked.nii.gz')

    sitk.WriteImage(moving_image_patch_itk, moving_patch_itk_filename)

    registration_method = sitk.ImageRegistrationMethod()
    transform = sitk.TranslationTransform(fixed_masked_image_itk.GetDimension())
    registration_method.SetInitialTransform(transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetOptimizerAsPowell(numberOfIterations=100, maximumLineIterations=100, stepLength=0.001,
                                             stepTolerance=1e-7, valueTolerance=1e-7)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInterpolator(sitk.sitkLinear)
    fixed_masked_image_itk = sitk.Cast(fixed_masked_image_itk, sitk.sitkFloat32)
    moving_image_patch_itk = sitk.Cast(moving_image_patch_itk, sitk.sitkFloat32)

    phase_1_transform = registration_method.Execute(fixed_masked_image_itk, moving_image_patch_itk)

    transform_filename = os.path.join(os.path.dirname(segmentation_output_name),
                                      os.path.basename(segmentation_output_name) + '_transform.txt')

    patch_phase_1_filename = os.path.join(os.path.dirname(patch_filename),
                                      os.path.basename(patch_filename) + '_phase_1.nii.gz')

    segmetnation_phase_1_filename = os.path.join(os.path.dirname(segmentation_output_name),
                                      os.path.basename(segmentation_output_name) + '_phase_1.nii.gz')

    moving_resampled_segmentation_phase_1 = sitk.Resample(moving_segmentation_patch_itk, fixed_masked_image_itk,
                                                  phase_1_transform, sitk.sitkNearestNeighbor, 0.0, moving_segmentation_patch_itk.GetPixelID())
    moving_resampled_patch_phase_1 = sitk.Resample(moving_image_patch_itk, fixed_masked_image_itk,
                                                  phase_1_transform, sitk.sitkLinear, 0.0,
                                                  moving_segmentation_patch_itk.GetPixelID())

    sitk.WriteImage(moving_resampled_patch_phase_1, patch_phase_1_filename)
    sitk.WriteImage(moving_resampled_segmentation_phase_1, segmetnation_phase_1_filename)
    sitk.WriteTransform(phase_1_transform, transform_filename)

    ###################################

    registration_method_phase_2 = sitk.ImageRegistrationMethod()
    initial_transform_2 = sitk.AffineTransform(fixed_masked_image_itk.GetDimension())
    initial_transform_2.SetCenter(fixed_masked_image_itk.TransformContinuousIndexToPhysicalPoint(
        [size // 2 for size in fixed_masked_image_itk.GetSize()]
    ))


    registration_method_phase_2.SetInitialTransform(copy.deepcopy(initial_transform_2), inPlace=False)
    registration_method_phase_2.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    registration_method_phase_2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration_method_phase_2.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method_phase_2.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=400,
        gradientMagnitudeTolerance=1e-6
    )

    registration_method_phase_2.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method_phase_2.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method_phase_2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method_phase_2.SetInterpolator(sitk.sitkLinear)
    registration_method_phase_2.SetMetricSamplingPercentage(0.2, sitk.sitkWallClock)
    registration_method_phase_2.SetOptimizerScalesFromPhysicalShift()

    moving_resampled_patch_phase_1 = sitk.Cast(moving_resampled_patch_phase_1, sitk.sitkFloat32)

    phase_2_transform = registration_method_phase_2.Execute(fixed_masked_image_itk, moving_resampled_patch_phase_1)

    transform_filename_phase_2 = os.path.join(os.path.dirname(segmentation_output_name),
                                      os.path.basename(segmentation_output_name) + '_transform_phase_2.txt')

    patch_phase_2_filename = os.path.join(os.path.dirname(patch_filename),
                                          os.path.basename(patch_filename) + '_phase_2.nii.gz')

    moving_resampled_segmentation_phase_2 = sitk.Resample(moving_resampled_segmentation_phase_1, fixed_masked_image_itk,
                                                          phase_2_transform, sitk.sitkNearestNeighbor, 0.0,
                                                          moving_segmentation_patch_itk.GetPixelID())
    moving_resampled_patch_phase_2 = sitk.Resample(moving_resampled_patch_phase_1, fixed_masked_image_itk,
                                                   phase_2_transform, sitk.sitkLinear, 0.0,
                                                   moving_segmentation_patch_itk.GetPixelID())

    sitk.WriteImage(moving_resampled_patch_phase_2, patch_phase_2_filename)
    sitk.WriteImage(moving_resampled_segmentation_phase_2, segmentation_output_name)
    sitk.WriteTransform(phase_2_transform, transform_filename_phase_2)


def perform_atlas_based_registration (fixed_patient_folder, fixed_patient_T1_image_filename, patient_points_set_filename, atlas_data_directory):
    # set paths of relevant ATLAS files:

    moving_atlas_image = os.path.join(atlas_data_directory, 'T1_icbm152.nii')
    patch_image_1 = os.path.join(atlas_data_directory, '_side_1_T1_icbm152.nii')
    patch_image_2 = os.path.join(atlas_data_directory, '_side_2_T1_icbm152.nii')
    segmentation_image_1 = os.path.join(atlas_data_directory, '_side_1_male_eye_segmentation_details_T1_coords.nii')
    segmentation_image_2 = os.path.join(atlas_data_directory, '_side_2_male_eye_segmentation_details_T1_coords.nii')
    mask_filename_1 = os.path.join(atlas_data_directory, 'male_eye_mask_eye_1_T1_coords_R.nii')
    mask_filename_2 = os.path.join(atlas_data_directory, 'male_eye_mask_eye_2_T1_coords_L.nii')
    atlas_points_set_filename = os.path.join(atlas_data_directory, 'T1_atlas_init_02.fcsv')

    fixed_patient_T1_image_filename = os.path.join(fixed_patient_folder, fixed_patient_T1_image_filename)

    tmp_folder = os.path.join(fixed_patient_folder, 'tmp')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # compute initial transform based on point based registratoin ###########################################
    init_transformed_image_filename = os.path.join(tmp_folder, 'atlas_init_fit.nii')
    init_transformed_segmentation_1_filename = os.path.join(tmp_folder, 'segmentation_1_init_fit.nii')
    init_transformed_segmentation_2_filename = os.path.join(tmp_folder, 'segmentation_2_init_fit.nii')
    init_transformed_patch_1_filename = os.path.join(tmp_folder, 'patch_1_init_fit.nii')
    init_transformed_patch_2_filename = os.path.join(tmp_folder, 'patch_2_init_fit.nii')
    init_transformed_mask_filename_1 = os.path.join(tmp_folder, 'mask_init_fit_1.nii')
    init_transformed_mask_filename_2 = os.path.join(tmp_folder, 'mask_init_fit_2.nii')
    phase_1_transformed_segmentation_1_filename = os.path.join(tmp_folder, 'segmentation_1_phase_2.nii')
    phase_1_transformed_segmentation_2_filename = os.path.join(tmp_folder, 'segmentation_2_phase_2.nii')

    transform = compute_init_transform(atlas_points_set_filename, patient_points_set_filename)
    apply_transformation_on_image(moving_atlas_image, fixed_patient_T1_image_filename, init_transformed_image_filename,
                                  transform, interpolation_type=sitk.sitkLinear)
    apply_transformation_on_image(segmentation_image_1, fixed_patient_T1_image_filename, init_transformed_segmentation_1_filename,
                                  transform, interpolation_type=sitk.sitkNearestNeighbor)
    apply_transformation_on_image(segmentation_image_2, fixed_patient_T1_image_filename,
                                  init_transformed_segmentation_2_filename,
                                  transform, interpolation_type=sitk.sitkNearestNeighbor)
    apply_transformation_on_image(patch_image_1, fixed_patient_T1_image_filename,
                                  init_transformed_patch_1_filename,
                                  transform, interpolation_type=sitk.sitkLinear)
    apply_transformation_on_image(patch_image_2, fixed_patient_T1_image_filename,
                                  init_transformed_patch_2_filename,
                                  transform, interpolation_type=sitk.sitkLinear)
    apply_transformation_on_image(mask_filename_1, fixed_patient_T1_image_filename,
                                  init_transformed_mask_filename_1,
                                  transform, interpolation_type=sitk.sitkNearestNeighbor)
    apply_transformation_on_image(mask_filename_2, fixed_patient_T1_image_filename,
                                  init_transformed_mask_filename_2,
                                  transform, interpolation_type=sitk.sitkNearestNeighbor)

    compute_patch_registration(fixed_patient_T1_image_filename, init_transformed_mask_filename_1,
                               init_transformed_patch_1_filename, init_transformed_segmentation_1_filename,
                               phase_1_transformed_segmentation_1_filename)

    compute_patch_registration(fixed_patient_T1_image_filename, init_transformed_mask_filename_2,
                               init_transformed_patch_2_filename, init_transformed_segmentation_2_filename,
                               phase_1_transformed_segmentation_2_filename)

    # clean tmp files
    results_folder = os.path.join(fixed_patient_folder, 'results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    shutil.copyfile(phase_1_transformed_segmentation_1_filename, os.path.join(results_folder, "segmentation_side_1.nii"))
    shutil.copyfile(phase_1_transformed_segmentation_2_filename,
                    os.path.join(results_folder, "segmentation_side_2.nii"))
    shutil.rmtree(tmp_folder)


# run IXI data example
fixed_patient_folder = r'/home/ruby/Research/data/02_opthalmology/example' ## TODO: update the path here
atlas_path = r'/home/ruby/Research/data/02_opthalmology/Atlases/T1_atlas' ## TODO: update the path here
output_prefix = "output_registration"

fixed_patient_T1_image_filename = os.path.join(fixed_patient_folder, 'IXI110-Guys-0733-T1.nii.gz')
patient_points_set_filename = os.path.join(fixed_patient_folder, 'IXI110_points.fcsv')
perform_atlas_based_registration (fixed_patient_folder, fixed_patient_T1_image_filename, patient_points_set_filename, atlas_data_directory = atlas_path)


# patients_folder = r'/home/ruby/Research/data/02_opthalmology/ixi/rubyshamir-20250130_083141'
# T1_folder = r'T1/NIfTI'
# relevant_patients_paths = [os.path.join(patients_folder, folder, T1_folder) for folder in os.listdir(patients_folder) if folder.startswith("___")]
# relevant_patients_paths = [os.path.join(patients_folder, folder, T1_folder) for folder in os.listdir(patients_folder) if folder.startswith("_01_")]
# relevant_patients_paths = [os.path.join(patients_folder, folder, T1_folder) for folder in os.listdir(patients_folder) if folder.startswith("_02_")]
# relevant_patients_paths = [os.path.join(patients_folder, folder, T1_folder) for folder in os.listdir(patients_folder) if folder.startswith("_03_")]
# relevant_patients_paths = [os.path.join(patients_folder, folder, T1_folder) for folder in os.listdir(patients_folder) if folder.startswith("_04_")]
# relevant_patients_paths = [os.path.join(patients_folder, folder, T1_folder) for folder in os.listdir(patients_folder) if folder.startswith("_05_")]
#
#
# for relevant_patient_path in relevant_patients_paths:
#     fixed_patient_T1_image_filename = os.path.join(relevant_patient_path, [filename for filename in os.listdir(relevant_patient_path) if filename.endswith('.nii.gz')][0])
#     patient_points_set_filename = os.path.join(relevant_patient_path, [filename for filename in os.listdir(relevant_patient_path) if filename.endswith('.fcsv')][0])
#     perform_atlas_based_registration(relevant_patient_path, fixed_patient_T1_image_filename, patient_points_set_filename, atlas_data_directory = r'/home/ruby/Research/data/02_opthalmology/Atlases/T1_atlas')
#

