import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_moving_objects(image1_path, image2_path, threshold=30):
    """
    Виявлення рухомих об'єктів між двома зображеннями.

    Args:
        image1_path (str): Шлях до першого зображення.
        image2_path (str): Шлях до другого зображення.
        threshold (int): Поріг для відмінностей.

    Returns:
        tuple: Відфільтроване зображення відмінностей і список контурів об'єктів.
    """
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, img1.shape[::-1])

    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    clean_diff = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return clean_diff, contours


def display_detected_objects(image_path, contours):
    """
    Відображення знайдених об'єктів на зображенні.

    Args:
        image_path (str): Шлях до зображення.
        contours (list): Список контурів.
    """
    img = cv2.imread(image_path)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Moving Objects")
    plt.axis("off")
    plt.show()


def save_detected_objects(contours, output_path="data/processed/objects.csv"):
    """
    Зберігає координати знайдених об'єктів у CSV.

    Args:
        contours (list): Список контурів.
        output_path (str): Шлях до файлу.
    """
    with open(output_path, "w") as f:
        f.write("x,y,width,height\n")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            f.write(f"{x},{y},{w},{h}\n")