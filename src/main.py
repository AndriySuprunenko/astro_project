from src.sdss_api import fetch_sdss_image
from src.image_processing import (
    normalize_brightness,
    gaussian_blur,
    canny_edges,
    find_contours,
    annotate_image
)
import cv2
import matplotlib.pyplot as plt


def main():
    # Встанови координати (Right Ascension і Declination)
    ra, dec = 180.0, 0.0

    # Завантаження зображення з SDSS
    image_path = fetch_sdss_image(ra, dec)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Обробка зображення
    print("Нормалізація яскравості...")
    normalized_image = normalize_brightness(image)

    print("Розмиття зображення...")
    blurred_image = gaussian_blur(normalized_image)

    print("Виявлення країв...")
    edges = canny_edges(blurred_image)

    print("Пошук контурів...")
    contours = find_contours(edges)

    print("Анотація зображення...")
    annotated_image = annotate_image(image, contours)

    # Візуалізація результатів
    visualize_results(image, normalized_image, blurred_image, edges, annotated_image)


def visualize_results(original, normalized, blurred, edges, annotated):
    """Функція для візуалізації результатів на кожному етапі."""
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Оригінал")

    axs[1].imshow(normalized, cmap='gray')
    axs[1].set_title("Нормалізоване")

    axs[2].imshow(blurred, cmap='gray')
    axs[2].set_title("Розмите")

    axs[3].imshow(edges, cmap='gray')
    axs[3].set_title("Краї")

    axs[4].imshow(annotated, cmap='gray')
    axs[4].set_title("Анотоване")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()