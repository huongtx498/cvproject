import cv2
import numpy as np
from typing import List


def fourier_transform(image):
    # Chuyển ảnh từ miền không gian sang miền tần số bằng FFT
    f = np.fft.fft2(image)
    center_shift = np.fft.fftshift(f)

    # Tính độ lớn của phổ Fourier (Fourier Spectrum)
    magnitude_spectrum = 30 * np.log(np.abs(center_shift))
    return center_shift, magnitude_spectrum


def inverse_FFT(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    # inverse furier
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
    return img_back


def computeIdealFiltering(D, Do, mode=0):
    """Computes Ideal Filtering based on the cut off frequency (Do).
    If mode=0, it compute Lowpass Filtering otherwise Highpass filtering    
    """
    H = np.zeros_like(D)
    if mode==0:
        H = (D<=Do).astype(int)
    else:
        H = (D>Do).astype(int)
    return H

def constructDuv(N):
    """Constructs the frequency matrix, D(u,v), of size NxN"""
    u = np.arange(N)
    v = np.arange(N)

    idx = np.where(u>N/2)[0]
    u[idx] = u[idx] - N
    idy = np.where(v>N/2)[0]
    v[idy] = v[idx] - N

    [V,U]= np.meshgrid(v,u)
    D = np.sqrt(U**2 + V**2)
    
    return D

def computeIdealFilters(F, D, Do):
    #Compute Ideal Lowpass Filter (ILPF)
    H_low = computeIdealFiltering(D, Do, 0)
    filtered_high = H_low * F

    #Compute Ideal Highpass Filter (IHPF)
    H_high = computeIdealFiltering(D, Do, 1)

    #Compute the filtered image (result in space domain)
    filtered_low = H_high * F 
    return filtered_low, filtered_high


import click
@click.command()
@click.option('--image', '-i', type=str, help='input file path', 
              default='./data/rice1.png', show_default='./data/rice1.png', prompt=True)
def main(image: str) -> None:
    import os
    from pathlib import Path

    os.makedirs('./output', exist_ok=True)

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    center_shift, magnitude_spectrum = fourier_transform(blur)

    rows, cols = blur.shape
    crow, ccol = rows // 2, cols // 2

    center_shift[crow - 2:crow + 2, 0:ccol - 2] = 1
    center_shift[crow - 2:crow + 2, ccol + 2:] = 1

    denoised_image = inverse_FFT(center_shift)
    blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)

    # can bang histogram cuc bo
    # clipLimit càng nhỏ thì mức độ nhiễu càng ít tuy nhiên hiệu quả cân bằng sáng lại càng thấp.
    h,w = blurred.shape[:]
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(int(w/50), int(h/50)))
    cl = clahe.apply(blurred)
    cl = cv2.GaussianBlur(cl, (5, 5), 0)

    thres1 = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, -12)

    kernel_morphological = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    thres_er = cv2.erode(thres1, kernel_morphological, iterations=1)
    thres_mor_1 = cv2.dilate(thres_er, kernel_morphological, iterations=1)

    contours, hierarchy = cv2.findContours(thres_mor_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = np.copy(img)
    count = 0
    
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True) 
        boundRect = cv2.boundingRect(contours_poly)
        
        cv2.rectangle(output_image,
                      (int(boundRect[0]), int(boundRect[1])),
                      (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])),
                      (0, 255, 0),
                      1)
        cv2.drawContours(output_image, contours, i, (0, 0, 255), 1)
        cv2.putText(output_image, str(count), (int(boundRect[0]), int(boundRect[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        count += 1
        
    print("\n>>> Number:\t", count)
    cv2.imwrite(os.path.join('./output', Path(image).stem + '.png'), output_image)

if __name__ == '__main__':
    main()