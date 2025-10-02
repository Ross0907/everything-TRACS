import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import time

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
    # The OPEN operation was removed as it was likely deleting the faint corner connection.
    # A single, very aggressive CLOSE operation is now used to both remove noise and
    # strongly connect all parts of the laser line.
    close_kernel = np.ones((25, 25), np.uint8) # Increased kernel size for stronger connection
    final_mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    
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
        return np.zeros_like(skeleton), None
            
    # Find the single longest contour
    longest_contour = max(contours, key=lambda c: cv2.arcLength(c, False))
    
    # Create a blank image and draw only the longest contour on it
    cleaned_skeleton = np.zeros_like(skeleton)
    cv2.drawContours(cleaned_skeleton, [longest_contour], -1, 255, 1)
    
    return cleaned_skeleton, longest_contour


def segment_profile(contour):
    """
    Intelligently segments a profile contour into top and bottom anchor sections.
    This version finds the corner of the 'L' shape for robust segmentation.
    """
    if contour is None or len(contour) < 3:
        return None, None

    # Find the corner point (highest point on the screen, smallest y-value)
    corner_idx = np.argmin(contour[:, 0, 1])
    corner_pt = tuple(contour[corner_idx][0])

    # Find the two endpoints of the contour
    endpoint1_idx = 0
    endpoint2_idx = len(contour) - 1
    
    # Find the distance from each endpoint to the corner
    dist1 = np.linalg.norm(np.array(corner_pt) - contour[endpoint1_idx][0])
    dist2 = np.linalg.norm(np.array(corner_pt) - contour[endpoint2_idx][0])

    # The longer segment is the main vertical line, the shorter one is the top horizontal
    if dist1 > dist2:
        # Vertical segment is from corner to endpoint1
        # Top segment is from corner to endpoint2
        top_segment_indices = (corner_idx, endpoint2_idx)
        bottom_endpoint_idx = endpoint1_idx
    else:
        # Vertical segment is from corner to endpoint2
        # Top segment is from corner to endpoint1
        top_segment_indices = (corner_idx, endpoint1_idx)
        bottom_endpoint_idx = endpoint2_idx

    # Ensure indices are ordered for slicing
    start_idx, end_idx = min(top_segment_indices), max(top_segment_indices)
    top_anchor = contour[start_idx:end_idx+1]
    
    # --- Define bottom anchor ---
    # Take a percentage of points leading up to the bottom endpoint
    num_points_for_anchor = max(10, int(len(contour) * 0.1)) # Use 10%
    if bottom_endpoint_idx == 0: # If the bottom is at the start of the array
        bottom_anchor = contour[:num_points_for_anchor]
    else: # If the bottom is at the end of the array
        bottom_anchor = contour[-num_points_for_anchor:]

    if len(top_anchor) < 2 or len(bottom_anchor) < 2:
        return None, None

    return top_anchor, bottom_anchor


def analyze_dent(perfect_image_path, dented_image_path):
    """
    Compares a dented image against a perfect reference image, focusing on a
    red laser line to detect and visualize defects.
    """
    start_time = time.time()
    try:
        img_perfect = cv2.imread(perfect_image_path)
        img_dented = cv2.imread(dented_image_path)
        if img_perfect is None or img_dented is None:
            print("Error: Could not load one or both images. Check file paths.")
            return
    except Exception as e:
        print(f"An error occurred while loading images: {e}")
        return

    # NOTE: The initial global alignment has been removed as it was unstable.
    # We will rely on the more robust piecewise alignment later.
    img_dented_aligned = img_dented

    mask_perfect, adjusted_perfect = get_laser_mask(img_perfect)
    mask_dented, adjusted_dented = get_laser_mask(img_dented_aligned)

    skeleton_perfect = (skeletonize(mask_perfect / 255) * 255).astype(np.uint8)
    skeleton_dented = (skeletonize(mask_dented / 255) * 255).astype(np.uint8)
    
    # Get the cleaned skeleton image AND the main contour from it
    cleaned_skeleton_perfect, contour_p = clean_skeleton(skeleton_perfect)
    cleaned_skeleton_dented, contour_d = clean_skeleton(skeleton_dented)

    final_warped_contour_d = contour_d # Initialize with original

    if contour_p is not None and contour_d is not None:
        # Get top and bottom anchor segments for alignment
        top_p, bottom_p = segment_profile(contour_p)
        top_d, bottom_d = segment_profile(contour_d)

        if all(seg is not None for seg in [top_p, bottom_p, top_d, bottom_d]):
            # Calculate translation for the top anchor
            M_p_top = cv2.moments(top_p)
            M_d_top = cv2.moments(top_d)
            if M_p_top["m00"] != 0 and M_d_top["m00"] != 0:
                cX_p_top = M_p_top["m10"] / M_p_top["m00"]
                cY_p_top = M_p_top["m01"] / M_p_top["m00"]
                cX_d_top = M_d_top["m10"] / M_d_top["m00"]
                cY_d_top = M_d_top["m01"] / M_d_top["m00"]
                dx_top = cX_p_top - cX_d_top
                dy_top = cY_p_top - cY_d_top
            else:
                dx_top, dy_top = 0, 0

            # Calculate translation for the bottom anchor
            M_p_bottom = cv2.moments(bottom_p)
            M_d_bottom = cv2.moments(bottom_d)
            if M_p_bottom["m00"] != 0 and M_d_bottom["m00"] != 0:
                cX_p_bottom = M_p_bottom["m10"] / M_p_bottom["m00"]
                cY_p_bottom = M_p_bottom["m01"] / M_p_bottom["m00"]
                cX_d_bottom = M_d_bottom["m10"] / M_d_bottom["m00"]
                cY_d_bottom = M_d_bottom["m01"] / M_d_bottom["m00"]
                dx_bottom = cX_p_bottom - cX_d_bottom
                dy_bottom = cY_p_bottom - cY_d_bottom
            else:
                dx_bottom, dy_bottom = 0, 0
                
            print(f"Top anchor translation: ({dx_top:.2f}, {dy_top:.2f})")
            print(f"Bottom anchor translation: ({dx_bottom:.2f}, {dy_bottom:.2f})")

            # --- Apply Interpolated Warp (Rubber Sheeting) ---
            warped_points = []
            num_points = len(contour_d)
            if num_points > 1:
                for i, point in enumerate(contour_d):
                    x, y = point[0]
                    # Calculate the weight. Weight is 1 at the top, 0 at the bottom.
                    weight = (num_points - 1 - i) / (num_points - 1)

                    # Interpolate the translation
                    dx = (weight * dx_top) + ((1 - weight) * dx_bottom)
                    dy = (weight * dy_top) + ((1 - weight) * dy_bottom)
                    
                    warped_points.append([x + dx, y + dy])
                
                final_warped_contour_d = np.array(warped_points, dtype=np.int32).reshape(-1, 1, 2)
                print("Applied piecewise warp to the dented profile.")
        else:
             print("Warning: Could not create anchor segments for piecewise alignment.")

    # --- Create Final Visualization ---
    final_aligned_skeleton_dented = np.zeros_like(cleaned_skeleton_dented)
    if final_warped_contour_d is not None:
        cv2.drawContours(final_aligned_skeleton_dented, [final_warped_contour_d], -1, 255, 1)

    dilate_kernel = np.ones((3, 3), np.uint8)
    thick_skeleton_perfect = cv2.dilate(cleaned_skeleton_perfect, dilate_kernel)
    thick_skeleton_dented = cv2.dilate(final_aligned_skeleton_dented, dilate_kernel)

    profile_vis = np.ones_like(img_perfect) * 255
    perfect_coords = np.where(thick_skeleton_perfect > 0)
    dented_coords = np.where(thick_skeleton_dented > 0)
    profile_vis[perfect_coords] = [255, 0, 0]  # Blue
    profile_vis[dented_coords] = [0, 0, 255]   # Red
    
    cv2.imwrite("profile_comparison.jpg", profile_vis)
    print("Saved 'profile_comparison.jpg'.")
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nTotal processing time: {processing_time:.2f} seconds")


    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Analysis complete in {processing_time:.2f} seconds", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(adjusted_perfect, cv2.COLOR_BGR2RGB))
    plt.title("Perfect Image (Exposure Adjusted)")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(adjusted_dented, cv2.COLOR_BGR2RGB))
    plt.title("Dented Image")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(profile_vis, cv2.COLOR_BGR2RGB))
    plt.title("Clean Profile Comparison\n(Blue=Perfect, Red=Dented)")
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

if __name__ == '__main__':
    perfect_image_file = "Perfect_Image.jpg"
    dented_image_file = "Dented_image.jpg"
    
    analyze_dent(perfect_image_file, dented_image_file)

