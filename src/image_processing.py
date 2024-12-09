import cv2
import numpy as np

def preprocess_image(image_path):
    """Попередня обробка зображення."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Згладжування для зменшення шуму
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def normalize_brightness(image):
    """Нормалізує яскравість зображення до діапазону [0, 255]."""
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm_image

def gaussian_blur(image, kernel_size=5):
    """Розмиття зображення для видалення шумів."""
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Застосовує білатеральну фільтрацію для видалення шуму."""
    filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered

def sobel_edges(image):
    """Виділення країв методом Собеля."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.uint8(sobel)

def canny_edges(image, threshold1=50, threshold2=150):
    """Виділення країв методом Кані."""
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges


def align_images(base_image, target_image):
    """Вирівнює target_image під base_image."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(base_image, None)
    kp2, des2 = orb.detectAndCompute(target_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(target_image, M, (base_image.shape[1], base_image.shape[0]))
    return aligned

def threshold_objects(image, threshold=127):
    """Виділяє об'єкти на зображенні за допомогою порогу."""
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def find_contours(image):
    """Пошук контурів на зображенні."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_mask(image, threshold=127):
    """Створює маску для виділення об'єктів."""
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return mask

def annotate_image(image, contours):
    """Анотує зображення, додаючи прямокутники навколо об'єктів."""
    annotated = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated