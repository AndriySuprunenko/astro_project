import argparse
from src.sdss_api import fetch_sdss_image
from src.image_processing import (
    normalize_brightness,
    gaussian_blur,
    canny_edges,
    find_contours,
    annotate_image
)
from src.object_detection import detect_moving_objects, display_detected_objects, save_detected_objects
import cv2
import os
import matplotlib.pyplot as plt


def process_single_image(ra, dec, scale=1.0):
    """Обробка одного зображення."""
    print("Завантаження зображення з SDSS...")
    image_path = fetch_sdss_image(ra, dec, scale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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


def process_moving_objects(ra1, dec1, ra2, dec2):
    """Обробка для виявлення рухомих об'єктів."""
    print("Завантаження зображень з SDSS...")
    image1_path = fetch_sdss_image(ra1, dec1)
    image2_path = fetch_sdss_image(ra2, dec2)

    print("Виявлення рухомих об'єктів...")
    diff_image, contours = detect_moving_objects(image1_path, image2_path)

    # Створення папки, якщо її немає
    os.makedirs("data/processed", exist_ok=True)

    print("Збереження результатів...")
    diff_output_path = "data/processed/detected_diff.jpg"
    cv2.imwrite(diff_output_path, diff_image)
    save_detected_objects(contours, "data/processed/objects.csv")

    print("Відображення результатів...")
    display_detected_objects(image1_path, contours)


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
    parser = argparse.ArgumentParser(description="Обробка астрономічних зображень")
    parser.add_argument("--mode", choices=["single", "motion"], required=True, help="Режим роботи")
    parser.add_argument("--ra1", type=float, required=True, help="Right Ascension (перший кадр або одиночний кадр)")
    parser.add_argument("--dec1", type=float, required=True, help="Declination (перший кадр або одиночний кадр)")
    parser.add_argument("--ra2", type=float, help="Right Ascension (другий кадр, якщо обрано motion)")
    parser.add_argument("--dec2", type=float, help="Declination (другий кадр, якщо обрано motion)")

    args = parser.parse_args()

    if args.mode == "single":
        process_single_image(args.ra1, args.dec1)
    elif args.mode == "motion":
        if args.ra2 is None or args.dec2 is None:
            raise ValueError("Для режиму motion потрібно вказати ra2 та dec2.")
        process_moving_objects(args.ra1, args.dec1, args.ra2, args.dec2)