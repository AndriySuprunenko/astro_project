import argparse
import os
import logging
from nasa_api import fetch_apod
from sdss_api import fetch_sdss_image
from image_processing import (
    normalize_brightness,
    gaussian_blur,
    canny_edges,
    find_contours,
    annotate_image,
)
from object_detection import (
    detect_moving_objects,
    display_detected_objects,
    save_detected_objects,
)
import cv2
import matplotlib.pyplot as plt


def ensure_data_folder():
    os.makedirs("data/processed", exist_ok=True)


def process_single_image(ra, dec, scale=1.0):
    """Обробка одного зображення."""
    logging.info("Завантаження зображення з SDSS...")
    image_path = fetch_sdss_image(ra, dec, scale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    logging.info("Нормалізація яскравості...")
    normalized_image = normalize_brightness(image)

    logging.info("Розмиття зображення...")
    blurred_image = gaussian_blur(normalized_image)

    logging.info("Виявлення країв...")
    edges = canny_edges(blurred_image)

    logging.info("Пошук контурів...")
    contours = find_contours(edges)

    logging.info("Анотація зображення...")
    annotated_image = annotate_image(image, contours)

    # Візуалізація результатів
    visualize_results(image, normalized_image, blurred_image, edges, annotated_image)


def process_moving_objects(ra1, dec1, ra2, dec2):
    logging.info("Завантаження зображень з SDSS...")
    scale = 1.0  # Встановіть значення масштабу

    # Завантаження зображень
    image1_path = fetch_sdss_image(ra1, dec1, scale)
    image2_path = fetch_sdss_image(ra2, dec2, scale)

    # Виявлення рухомих об'єктів
    diff_image, contours = detect_moving_objects(image1_path, image2_path)

    # Збереження результатів
    diff_output_path = "data/processed/detected_diff.jpg"
    cv2.imwrite(diff_output_path, diff_image)
    save_detected_objects(contours, "data/processed/objects.csv")

    # Відображення результатів
    display_detected_objects(image1_path, contours)


def visualize_results(original, normalized, blurred, edges, annotated):
    """Функція для візуалізації результатів на кожному етапі."""
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Оригінал")

    axs[1].imshow(normalized, cmap="gray")
    axs[1].set_title("Нормалізоване")

    axs[2].imshow(blurred, cmap="gray")
    axs[2].set_title("Розмите")

    axs[3].imshow(edges, cmap="gray")
    axs[3].set_title("Краї")

    axs[4].imshow(annotated, cmap="gray")
    axs[4].set_title("Анотоване")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def process_apod(date=None):
    logging.info("Отримання зображення NASA APOD...")
    image_path, metadata = fetch_apod(date)

    if image_path:
        logging.info(f"Зображення збережено за шляхом: {image_path}")
    else:
        logging.info("Зображення відсутнє, але метадані отримані.")

    logging.info("Метадані:")
    for key, value in metadata.items():
        logging.info(f"{key}: {value}")


if __name__ == "__main__":
    # Налаштування логування
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ensure_data_folder()

    parser = argparse.ArgumentParser(description="Обробка астрономічних зображень")
    parser.add_argument(
        "--mode", choices=["single", "motion", "apod"], required=True, help="Режим роботи"
    )
    parser.add_argument(
        "--ra1",
        type=float,
        help="Right Ascension (перший кадр або одиночний кадр)",
    )
    parser.add_argument(
        "--dec1",
        type=float,
        help="Declination (перший кадр або одиночний кадр)",
    )
    parser.add_argument(
        "--ra2", type=float, help="Right Ascension (другий кадр, якщо обрано motion)"
    )
    parser.add_argument(
        "--dec2", type=float, help="Declination (другий кадр, якщо обрано motion)"
    )
    parser.add_argument("--date", type=str, help="Дата у форматі YYYY-MM-DD для завантаження APOD.")

    args = parser.parse_args()

    if args.mode == "single":
        if args.ra1 is None or args.dec1 is None:
            raise ValueError("Для режиму single потрібно вказати ra1 та dec1.")
        process_single_image(args.ra1, args.dec1)
    elif args.mode == "apod":
        process_apod(args.date)
    elif args.mode == "motion":
        ra2 = args.ra2 if args.ra2 is not None else args.ra1 + 0.001
        dec2 = args.dec2 if args.dec2 is not None else args.dec1
        if args.ra1 is None or args.dec1 is None:
            raise ValueError("Для режиму motion потрібно вказати ra1 та dec1.")
        process_moving_objects(args.ra1, args.dec1, ra2, dec2)
