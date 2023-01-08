import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Iterate through folder or select specific image
# path = os.path.join("./img_data/", "new_wet_samples")
path = os.path.join("./img_data/", "old_wet_samples_real_humans")
images = next(os.walk(path), (None, None, []))[2]
NUM_ALL_IMAGES = len(images)

print(images)


def threshold(img):
    """
    Threshold grayscale image using the following three methods:
        1. Binary thresholding with threshold = 127
        2. Otsu method
        3. Gaussian filtering + Otsu method

    Input: Grayscale image
    Output: Three binary images, respectively processed by the three methods
        described above
    """
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th1, th2, th3


def morph(img):
    """
    Perform morphology to remove isolated black dots surrounding vegetable and
        close white areas within the vegetable

    Input: Binary image
    Output: Binary image that has been closed and then opened.  Sorry for the
        bad documentation lol
    """
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((100, 100), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


# Plot the images
def plot_image_routine2(images, img_no=1, show=False):
    num_images = len(images)
    for i, (title, img) in enumerate(images):
        plt.subplot(NUM_ALL_IMAGES, num_images, (img_no * num_images) + (i + 1))
        plt.imshow(img, cmap="gray")
        # plt.title(title)
        plt.axis("off")
    if show:
        plt.show()


# Loop through images and process
for img_no, img_file in enumerate(sorted(images)):
    # Load image and convert to gray
    print(f"Processing {img_file}")
    img_path = os.path.join(path, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        print("Could not open or find the images!")
        exit(0)
    print(f"Image Shape: {img.shape}")

    # Normalize image to min and max intensity of 0 and 255
    print(f"Original min: {np.min(img)} max: {np.max(img)}")
    norm = np.zeros_like(img)
    norm = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)
    print(f"Norm min: {np.min(norm)} max: {np.max(norm)}")

    # Get separate channels
    # zeros = np.zeros_like(norm.shape[:2])
    R, G, B = cv2.split(norm)
    zeros = np.zeros(norm.shape[:2], dtype="uint8")
    red = cv2.merge([R, zeros, zeros])
    green = cv2.merge([zeros, G, zeros])
    # blue = cv2.merge([zeros, zeros, B])
    # gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

    # Threshold the red image
    global_thr, otsu_thr, gaussian_thr = threshold(R)
    # morphed_gaussian_thr = morph(gaussian_thr)

    # Erode the image
    kernel = np.ones((40, 40), np.uint8)
    gaussian_thr = cv2.erode(gaussian_thr, kernel, iterations=1)

    # Zero background where we want to overlay
    masked_img = np.copy(img)
    mask = gaussian_thr == 0
    masked_img[mask] = 0
    masked_green = np.copy(green)
    masked_green[mask] = 0

    # Add object to zeroed out space
    # img_copy += img * (img_copy > 0)

    # Plot images
    def plot_image_routine(title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    transformed_images = [
        ("Original", img),
        ("Masked original", masked_img),
        ("Green", green),
        ("Masked green", masked_green),
    ]
    # for img in images:
    #     plot_image_routine(*img)

    # cv2.imshow("Greyscale masked green", masked_green[:, :, 1])
    # cv2.waitKey(0)

    _, _, gaussian_masked_green = threshold(masked_green[:, :, 1])
    # plot_image_routine("Gaussian masked green", gaussian_masked_green)

    # Use morphology to remove isolated black spots and remove white openings in vegetable
    # morphed_masked_img = morph(masked_img)

    transformed_images.append(("Gaussian masked green", gaussian_masked_green))
    is_last_image = img_no == len(images) - 1
    plot_image_routine2(transformed_images, img_no, show=is_last_image)

    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.title("Original")
    # plt.subplot(2, 2, 2)
    # plt.imshow(norm, cmap="gray")
    # plt.title("Normalized")
    # plt.subplot(2, 2, 3)
    # plt.imshow(red, cmap="gray")
    # plt.title("Red")
    # plt.subplot(2, 2, 4)
    # plt.imshow(masked_img, cmap="gray")
    # plt.title("Masked")
    # plt.show()
