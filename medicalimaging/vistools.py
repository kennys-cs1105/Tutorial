# 可视化工具

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from medicalimaging import CTtools


def show_img(img):
    if img.ndim == 3:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img)
    plt.show()


def gen_contrast_colors(n, order="BGR"):
    """# Example: Generate 5 BGR colors with high contrast
    print(gen_contrast_colors(5))

    # Example: Generate 5 RGB colors with high contrast
    print(gen_contrast_colors(5, 'RGB'))

    Args:
        n (_type_): _description_
        order (str, optional): _description_. Defaults to 'BGR'.
    """

    def rand_comp():
        # Generate a random color component value
        return random.randint(128, 255)  # Ensure high brightness

    def comp_color(c):
        # Compute the complementary color component
        return 255 - c

    def format_clr(r, g, b, ord):
        # Format the color in the specified order
        clr = (r, g, b)
        if ord == "BGR":
            return (clr[2], clr[1], clr[0])
        elif ord == "RGB":
            return clr
        else:
            raise ValueError("Invalid order. Use 'BGR' or 'RGB'.")

    colors = []
    for _ in range(n):
        # Generate random color and its complementary color
        r, g, b = rand_comp(), rand_comp(), rand_comp()
        while max([r, g, b]) - min([r, g, b]) < 50:  # Avoid grayscale
            r, g, b = rand_comp(), rand_comp(), rand_comp()

        comp_r, comp_g, comp_b = comp_color(r), comp_color(g), comp_color(b)
        # Choose one of the two colors randomly to ensure contrast
        chosen_clr = (
            format_clr(r, g, b, order)
            if random.choice([True, False])
            else format_clr(comp_r, comp_g, comp_b, order)
        )
        colors.append(chosen_clr)
    return colors


def display_ct_with_lesion_contours(
    apply_limits=False,
    class_num=None,
    ct_array=None,
    ct_path=None,
    lesion_array=None,
    lesion_path=None,
    preset="lung",
    window_center=None,
    window_width=None,
    save=False,
    pdf=False,
    serial_number=None,
    save_folder="output",
):
    if ct_array is None:
        # 读取CT NIfTI图像
        ct_nii = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_nii)
        ct_array = CTtools.apply_windowing(
            ct_array,
            window_center=window_center,
            window_width=window_width,
            apply_limits=apply_limits,
            preset=preset,
        )
    if lesion_array is None:
        # 读取病灶NIfTI图像
        lesion_nii = sitk.ReadImage(lesion_path)
        lesion_array = sitk.GetArrayFromImage(lesion_nii)
    # Check if the dimensions match
    if ct_array.shape != lesion_array.shape:
        raise ValueError("CT and lesion images must have the same dimensions.")

    if class_num is None:
        class_num = np.unique(lesion_array[lesion_array != 0]).shape[
            0
        ]  # 除去背景0。直接移除0，剩下稀疏的标注，运行效率更高。
    # Generate colors for each class
    colors = gen_contrast_colors(class_num)

    # Create a legend for the colors
    plt.figure(figsize=(2, class_num))
    plt.title("Legend")
    for i, color in enumerate(colors):
        plt.barh(i, 1, color=np.array(color) / 255.0)
        plt.text(0.5, i, f"Class {i+1}", ha="center", va="center", color="white")
    plt.yticks([])
    if not save:
        plt.show()

    # Create a PDF object if needed
    if save and pdf:
        import matplotlib.backends.backend_pdf

        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"{save_folder}/{serial_number}.pdf"
        )

    if save:
        os.makedirs(save_folder, exist_ok=True)
    # Process each slice
    for slice_index in range(lesion_array.shape[0]):
        ct_slice = ct_array[slice_index].copy()
        lesion_slice = lesion_array[slice_index].copy()

        # Skip slices without lesions
        if not np.any(lesion_slice):
            continue

        # Draw contours for each class
        ct_slice_rgb = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)
        for class_id in range(1, class_num + 1):
            mask = lesion_slice == class_id
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(ct_slice_rgb, contours, -1, colors[class_id - 1], 2)

        # Display the original and contoured slices
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(ct_slice, cmap="gray")
        ax[0].set_title(f"Original CT Slice - Slice {slice_index}")
        ax[1].imshow(ct_slice_rgb)
        ax[1].set_title(f"CT Slice with Lesion Contours - Slice {slice_index}")
        if not save:
            plt.show()

        # Save the figure if needed
        if save:
            if pdf:
                pdf.savefig(fig)
            else:
                plt.savefig(f"{save_folder}/{serial_number}.{slice_index}.png")
        plt.close(fig)  # Close the figure
        
    # Close the PDF object if needed
    if save and pdf:
        pdf.close()
