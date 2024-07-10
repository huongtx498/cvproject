import cv2
import numpy as np
from typing import List


class Contour:
    def __init__(self, contour_data):
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour_data)
        return

    def get_precedence(self, row_and_column_classifiers):
        y_values_of_rows, horizontal_one_x = row_and_column_classifiers
        x, y, w, h = self.x, self.y, self.w, self.h

        for row_y in y_values_of_rows:
            if row_y - h / 3 < y < row_y + h / 3:
                row_num = y_values_of_rows.index(row_y)
                break
            
        if x < horizontal_one_x or horizontal_one_x * 2 < x < horizontal_one_x * 3:
            self.x = x + 40 
            self.width = w - 40

        column = int(x >= horizontal_one_x * 2)
        return column * 10000000 + row_num * 10000 + x


def box_extraction(img: np.ndarray, num_row: int=None, num_col: int=4):
    # Adaptive mean: White => Black
    extraction_template = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    extraction_template = 255 - extraction_template

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[0] // 45)) # [1x35]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 55, 1)) # [35x1]

    vertical_template = cv2.erode(extraction_template, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(vertical_template, vertical_kernel, iterations=3)

    horizontal_template = cv2.erode(extraction_template, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_template, horizontal_kernel, iterations=3)

    alpha = 0.5  
    beta = 1.0 - alpha
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    template_sum = cv2.addWeighted(vertical_lines, alpha, horizontal_lines, beta, 0.0)
    template_sum = cv2.erode(~template_sum, kernel_3x3, iterations=2)
    (thresh, template_sum) = cv2.threshold(template_sum, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    extraction_template = cv2.morphologyEx(template_sum, cv2.MORPH_OPEN, vertical_kernel)
    extraction_template = cv2.morphologyEx(extraction_template, cv2.MORPH_OPEN, horizontal_kernel)

    contours, hierarchy = cv2.findContours(extraction_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [Contour(contour) for contour in contours]

    contours = filter_contours(contours, img)

    y_values_of_rows = get_row_y_values(contours)
    horizontal_one_x = round(extraction_template.shape[1] / num_col)
    row_and_column_classifiers = (y_values_of_rows, horizontal_one_x)

    # order column number
    contours.sort(key=lambda contour: Contour.get_precedence(contour, row_and_column_classifiers))

    extracted_boxes = []
    for contour in contours:
        x, y, w, h = contour.x, contour.y, contour.w, contour.h
        extracted_boxes.append(img[y:y + h, x:x + w])
    return extracted_boxes


def filter_contours(contours: List[Contour], img: np.ndarray, num_col: int=4, delta: int=100):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = contour.x, contour.y, contour.w, contour.h
        if (img.shape[1]/num_col - delta) < w:
            filtered_contours.append(contour)
    return filtered_contours


def get_row_y_values(contours):
    y_values = []

    for contour in contours:
        x, y, w, h = contour.x, contour.y, contour.w, contour.h
        new_row = True

        for row_y in y_values:
            same_row = y - h / 3 < row_y < y + h / 3
            if same_row:
                new_row = False
        if new_row:
            y_values.append(y)

    y_values.sort(key=lambda x: x)
    return y_values


import click
@click.command()
@click.option('--image', '-i', type=str, help='input file path', 
              default='./data/001_0.png', show_default='./data/001_0.png', prompt=True)
@click.option('--num_col', '-c', type=int, help='number of columns', 
              default=5, show_default=5, prompt=True)
@click.option('--num_row', '-r', type=int, help='number of rows', 
              default=35, show_default=35, prompt=True)
def main(image: str, num_col: int, num_row: int) -> None:
    import os
    from pathlib import Path
    
    from PIL import Image
    import matplotlib.pyplot as plt
    
    os.makedirs('./output', exist_ok=True)

    im = Image.open(image)
    numpy = np.array(im)[:, :, 0]

    resized = cv2.resize(numpy, (1100, 1700))
    cut_images = box_extraction(resized)
    num_boxes = len(cut_images)

    rows = (num_boxes + num_col - 1) // num_col

    fig, axes = plt.subplots(rows, num_col, figsize=(15, 15))
    axes = axes.flatten()

    for i, box in enumerate(cut_images):
        axes[i].imshow(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.savefig(os.path.join('./output', Path(image).stem + '.png'), bbox_inches='tight')


if __name__ == '__main__':
    main()