import numpy as np


def center_crop_numpy(H, size):
    Nh = H.shape[0]
    Nw = H.shape[1]

    return H[(Nh - size) // 2 : (Nh + size) // 2, (Nw - size) // 2 : (Nw + size) // 2]


def normxcorr2(template, image):
    # Compute mean of template and image
    template_mean = np.mean(template)
    image_mean = np.mean(image)

    # Subtract mean from template and image
    template_norm = template - template_mean
    image_norm = image - image_mean

    # Compute normalized cross-correlation
    numerator = np.sum(template_norm * image_norm)
    denominator = np.sqrt(np.sum(template_norm**2) * np.sum(image_norm**2))
    correlation = numerator / denominator

    return correlation


def imcrop_numpy(arr, rect, add=0):
    """
    주어진 NumPy 배열 arr에서 rect에 정의된 사각형 부분을 잘라서 반환합니다.

    Parameters:
    - arr (numpy.ndarray): 원본 이미지 데이터를 나타내는 NumPy 배열
    - rect (list): 자를 사각형의 정보를 담고 있는 리스트 [left, upper, width, height]

    Returns:
    - cropped_arr (numpy.ndarray): 잘린 부분을 포함하는 NumPy 배열
    """
    left, upper, width, height = rect

    left = left - add // 2
    upper = upper - add // 2
    width = width + add
    height = height + add

    # 사각형을 벗어나지 않도록 좌표 조정
    right = left + width
    lower = upper + height

    # NumPy 배열에서 사각형 부분 자르기
    cropped_arr = arr[upper:lower, left:right]

    return cropped_arr
