import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def adjust_exposure_gamma(image, gamma=0.5, gain=1.0):
    """
    Adjusts the exposure and gamma of an image to isolate bright regions.
    A lower gamma value darkens the image, making bright areas like lasers pop.
    
    Args:
        image (numpy.ndarray): The input image.
        gamma (float): Gamma correction value. < 1 darkens, > 1 lightens.
        gain (float): A multiplier for brightness (not used in this version).
        
    Returns:
        numpy.ndarray: The adjusted image.
    """
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values.
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)

def align_images(im1, im2):
    """
    Aligns im2 to im1 using ORB feature matching.
    This corrects for small shifts or rotations between the two images.
    
    Args:
        im1 (numpy.ndarray): The reference image.
        im2 (numpy.ndarray): The image to be aligned.
        
    Returns:
        tuple: A tuple containing the aligned image and the homography matrix.
    """
    # Convert images to grayscale for feature detection
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # --- 1. Detect ORB features and compute descriptors ---
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.15
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    if descriptors1 is None or descriptors2 is None:
        print("Warning: Could not find features in one or both images for alignment.")
        # Return original image if features can't be found
        return im2, np.identity(3) 

    # --- 2. Match features ---
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors2, descriptors1, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Keep only the best matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # --- 3. Find Homography ---
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.trainIdx].pt
        points2[i, :] = keypoints2[match.queryIdx].pt

    # Find the perspective transformation matrix
    try:
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    except cv2.error as e:
        print(f"Warning: Homography calculation failed: {e}. Returning original image.")
        return im2, np.identity(3) 

    if h is None:
        print("Warning: Homography could not be computed. Using original image.")
        return im2, np.identity(3)

    # --- 4. Warp image ---
    height, width, channels = im1.shape
    im2_aligned = cv2.warpPerspective(im2, h, (width, height))

    return im2_aligned, h

def get_laser_mask(image):
    """
    Isolates the red laser line using advanced filtering and thresholding.
    """
    # --- 1. Programmatically adjust exposure and gamma ---
    adjusted_image = adjust_exposure_gamma(image, gamma=0.5)

    # --- 2. Isolate Red Channel and Threshold ---
    red_channel = adjusted_image[:, :, 2]
    _, thresholded = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
    
    # --- 3. Morphological Cleaning ---
    # First, use OPEN to remove small noise specks from the background
    open_kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, open_kernel, iterations=2)
    
    # Then, use CLOSE with a larger kernel to fill gaps in the main laser line
    close_kernel = np.ones((11, 11), np.uint8)
    final_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    return final_mask, adjusted_image

def clean_skeleton(skeleton):
    """
    Finds all contours in a skeleton image and returns a new image containing only
    the single longest contour, which represents the main profile line.
    """
    # Find all contours in the skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, return a blank image
    if not contours:
        return np.zeros_like(skeleton)
        
    # Find the single longest contour
    longest_contour = max(contours, key=lambda c: cv2.arcLength(c, False))
    
    # Create a blank image and draw only the longest contour on it
    cleaned_skeleton = np.zeros_like(skeleton)
    cv2.drawContours(cleaned_skeleton, [longest_contour], -1, 255, 1)
    
    return cleaned_skeleton

def get_top_segment(contour):
    """
    Isolates the top, mostly horizontal segment of an L-shaped contour.
    This segment is used as a trusted "anchor" for alignment.
    """
    if contour is None or len(contour) < 3:
        return None

    # Find the corner point, which is assumed to be the point with the minimum y-value.
    corner_idx = np.argmin(contour[:, 0, 1])

    # Find the two endpoints of the contour
    endpoint1 = tuple(contour[0][0])
    endpoint2 = tuple(contour[-1][0])

    # The endpoint belonging to the horizontal segment will have a y-value closer
    # to the corner's y-value.
    dist1_y = abs(endpoint1[1] - contour[corner_idx][0][1])
    dist2_y = abs(endpoint2[1] - contour[corner_idx][0][1])

    if dist1_y < dist2_y:
        # Endpoint 1 and the corner define the top segment
        start_idx, end_idx = 0, corner_idx
    else:
        # Endpoint 2 and the corner define the top segment
        start_idx, end_idx = corner_idx, len(contour) - 1
    
    # Ensure start_idx is always less than end_idx for slicing
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return contour[start_idx:end_idx+1]

def analyze_dent(perfect_image_path, dented_image_path):
    """
    Compares a dented image against a perfect reference image, focusing on a
    red laser line to detect and visualize defects.
    """
    try:
        img_perfect = cv2.imread(perfect_image_path)
        img_dented = cv2.imread(dented_image_path)
        if img_perfect is None or img_dented is None:
            print("Error: Could not load one or both images. Check file paths.")
            return
    except Exception as e:
        print(f"An error occurred while loading images: {e}")
        return

    print("Aligning images...")
    img_dented_aligned, _ = align_images(img_perfect, img_dented)
    print("Alignment complete.")

    # Isolate laser and get the exposure-adjusted images for visualization
    mask_perfect, adjusted_perfect = get_laser_mask(img_perfect)
    mask_dented, adjusted_dented = get_laser_mask(img_dented_aligned)

    # Focused Metric Calculation
    combined_mask = cv2.bitwise_or(mask_perfect, mask_dented)
    gray_perfect = cv2.cvtColor(img_perfect, cv2.COLOR_BGR2GRAY)
    gray_dented = cv2.cvtColor(img_dented_aligned, cv2.COLOR_BGR2GRAY)
    roi_perfect = cv2.bitwise_and(gray_perfect, gray_perfect, mask=combined_mask)
    roi_dented = cv2.bitwise_and(gray_dented, gray_dented, mask=combined_mask)
    
    if np.any(roi_dented) and np.any(roi_perfect):
        psnr_value = psnr(roi_perfect, roi_dented, data_range=roi_dented.max() - roi_dented.min())
        ssim_value, _ = ssim(roi_perfect, roi_dented, full=True, data_range=roi_dented.max() - roi_dented.min())
        print(f"\n--- FOCUSED Image Quality Metrics (Laser ROI only) ---")
        print(f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
        print("-----------------------------------------------------\n")
    else:
        print("\n--- Could not calculate focused metrics: Laser not detected ---\n")

    # Create and Clean Skeletonized Profiles
    skeleton_perfect = (skeletonize(mask_perfect / 255) * 255).astype(np.uint8)
    skeleton_dented = (skeletonize(mask_dented / 255) * 255).astype(np.uint8)
    cleaned_skeleton_perfect = clean_skeleton(skeleton_perfect)
    cleaned_skeleton_dented = clean_skeleton(skeleton_dented)

    # --- Refined Alignment by matching the top straight segment ---
    final_aligned_skeleton_dented = cleaned_skeleton_dented.copy()

    contours_perfect, _ = cv2.findContours(cleaned_skeleton_perfect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_dented, _ = cv2.findContours(cleaned_skeleton_dented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_perfect and contours_dented:
        contour_p = contours_perfect[0]
        contour_d = contours_dented[0]

        # Isolate the top segments of both profiles to use as alignment anchors
        top_segment_p = get_top_segment(contour_p)
        top_segment_d = get_top_segment(contour_d)

        if top_segment_p is not None and top_segment_d is not None and len(top_segment_p) > 0 and len(top_segment_d) > 0:
            
            # --- Align by position (Translation) only ---
            M_p = cv2.moments(top_segment_p)
            M_d = cv2.moments(top_segment_d)

            if M_p["m00"] != 0 and M_d["m00"] != 0:
                # Calculate the center of the anchor segments
                cX_p = int(M_p["m10"] / M_p["m00"])
                cY_p = int(M_p["m01"] / M_p["m00"])
                cX_d = int(M_d["m10"] / M_d["m00"])
                cY_d = int(M_d["m01"] / M_d["m00"])
                
                # Calculate the translation required
                dx = cX_p - cX_d
                dy = cY_p - cY_d
                
                # Create the translation matrix
                M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # Apply the translation to the entire dented skeleton
                h, w = cleaned_skeleton_dented.shape
                final_aligned_skeleton_dented = cv2.warpAffine(cleaned_skeleton_dented, M_trans, (w, h))
                print(f"Aligning based on top segment: Applied translation of ({dx}, {dy}) pixels.")
            else:
                print("Warning: Could not calculate moments for fine-tuned alignment.")
        else:
            print("Warning: Could not isolate top segments for fine-tuned alignment.")
    else:
        print("Warning: Could not find contours in one or both skeletons for alignment.")

    # --- Create Final Visualization ---
    # Thicken the lines for better visibility
    dilate_kernel = np.ones((3, 3), np.uint8)
    thick_skeleton_perfect = cv2.dilate(cleaned_skeleton_perfect, dilate_kernel)
    thick_skeleton_dented = cv2.dilate(final_aligned_skeleton_dented, dilate_kernel)

    # Create the comparison image
    profile_vis = np.ones_like(img_perfect) * 255
    perfect_coords = np.where(thick_skeleton_perfect > 0)
    dented_coords = np.where(thick_skeleton_dented > 0)
    profile_vis[perfect_coords] = [255, 0, 0]  # Blue
    profile_vis[dented_coords] = [0, 0, 255]   # Red
    
    cv2.imwrite("profile_comparison.jpg", profile_vis)
    print("Saved 'profile_comparison.jpg'.")

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(adjusted_perfect, cv2.COLOR_BGR2RGB))
    plt.title("Perfect Image (Exposure Adjusted)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(adjusted_dented, cv2.COLOR_BGR2RGB))
    plt.title("Dented Image (Exposure Adjusted)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(profile_vis, cv2.COLOR_BGR2RGB))
    plt.title("Clean Profile Comparison\n(Blue=Perfect, Red=Dented)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    perfect_image_file = "Perfect_Image.jpg"
    dented_image_file = "Dented_Image.jpg"
    
    analyze_dent(perfect_image_file, dented_image_file)

