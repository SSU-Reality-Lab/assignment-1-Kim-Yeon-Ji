import sys
import cv2
import numpy as np
import os

def gaussian_blur_kernel_2d(sigma, height, width):
    '''주어진 sigma와 (height x width) 차원에 해당하는 가우시안 블러 커널을
    반환합니다. width와 height는 서로 다를 수 있습니다.

    입력(Input):
        sigma:  가우시안 블러의 반경(정도)을 제어하는 파라미터.
                본 과제에서는 높이와 너비 방향으로 대칭인 원형 가우시안(등방성)을 가정합니다.
        width:  커널의 너비.
        height: 커널의 높이.

    출력(Output):
        (height x width) 크기의 커널을 반환합니다. 이 커널로 이미지를 컨볼브하면
        가우시안 블러가 적용된 결과가 나옵니다.
    '''
    # 블러 반경을 제어하는 시그마가 적용된 크기의 커널이 반환되어야 함.   
    # 이미지는 행열 2D grid, 필터는 radius로 정의 됨.
    # 원형 필터 중심에서 멀어질수록 가우시안 필터 값이 덜 적용되어야 함. 
    
    img = np.zeros((height, width), dtype=np.float32) # 이미지 
    center_x = width // 2 # 커널 중심 좌표 계산
    center_y = height // 2
    for i in range(height): # 커널 각 픽셀에 대해 가우시안 필터 적용
        for j in range(width): 
            x = j - center_x 
            y = i - center_y
            img[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    return img / np.sum(img) # 커널 합이 1이 되게 정규화   


def cross_correlation_2d(img, kernel):
    '''주어진 커널(크기 m x n )을 사용하여 입력 이미지와의
    2D 상관(cross-correlation)을 계산합니다. 출력은 입력 이미지와 동일한 크기를
    가져야 하며, 이미지 경계 밖의 픽셀은 0이라고 가정합니다. 입력이 RGB 이미지인
    경우, 각 채널에 대해 커널을 별도로 적용해야 합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).
    '''
    
    '''출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    height, width = img.shape[:2] # 입력 이미지
    kernel_height, kernel_width = kernel.shape # 커널 높이와 너비

    padding_height = kernel_height // 2 # 패딩
    padding_width = kernel_width // 2

    # 입력 이미지 채널에 따라 패딩 적용
    # 그레이스케일
    if len(img.shape) == 2: 
        padding = np.pad(img, ((padding_height, padding_height), (padding_width, padding_width)), 'constant', constant_values=0) # 경계 밖은 0으로 처리 
        output_img = np.zeros((height, width), dtype=np.float32) # 출력 이미지 초기화 

        # 이미지 슬라이싱 및 처리
        for i in range(height):
            for j in range(width):
                output_img[i,j] = np.sum(padding[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    # RGB
    elif len(img.shape) == 3: 
        padding = np.pad(img, ((padding_height, padding_height), (padding_width, padding_width), (0,0)), 'constant', constant_values=0) # 경계 밖은 0으로 처리 
        output_img = np.zeros((height, width, 3), dtype=np.float32) # 출력 이미지 초기화

        # 이미지 슬라이싱 및 처리
        for RGB in range(3):
            for i in range(height):
                for j in range(width):
                    output_img[i,j,RGB] = np.sum(padding[i:i+kernel_height, j:j+kernel_width, RGB ] * kernel)
    return output_img


def convolve_2d(img, kernel):
    '''cross_correlation_2d()를 사용하여 2D 컨볼루션을 수행합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    # 커널 180도 회전
    filpped_kernel = np.flip(kernel, axis=(0,1))

    # 회전된 커널로 convolution 수행
    convolve_img = cross_correlation_2d(img, filpped_kernel)
    return convolve_img

def low_pass(img, sigma, size):
    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 저역통과(low-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 저역통과 필터는 이미지의
    고주파(세밀한 디테일) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''

    # 가우시안 커널 생성 
    kernel = np.zeros((size, size), dtype=np.float32) # 커널 초기화 
    center = size // 2 # 커널 중심 좌표 계산

    # 커널 각 픽셀에 대해 가우시안 필터 적용
    for i in range(size): 
        for j in range(size): 
            x = j - center 
            y = i - center
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
   
    # 커널 정규화
    kernel_sum = np.sum(kernel)
    kernel = kernel / kernel_sum

    # 커널 사용해 컨볼루션 함수 호출
    low_filtered_img = convolve_2d(img, kernel)

    return low_filtered_img 


def high_pass(img, sigma, size):
    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 고역통과(high-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 고역통과 필터는 이미지의
    저주파(거친 형태) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''

    # 가우시안 커널 생성 
    kernel = np.zeros((size, size), dtype=np.float32) # 커널 초기화
    center = size // 2 # 커널 중심 좌표 계산

    # 커널 각 픽셀에 대해 가우시안 필터 적용
    for i in range(size): 
        for j in range(size): 
            x = j - center 
            y = i - center
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
   
    # 커널 정규화
    kernel_sum = np.sum(kernel)
    kernel = kernel / kernel_sum

    # 커널 사용해 컨볼루션 함수 호출
    low_filtered_img = convolve_2d(img, kernel)

    return img - low_filtered_img # 원본 이미지에서 저역 성분 빼서 고역 성분만 남김



def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
