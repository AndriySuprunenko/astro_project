import requests
import os

# Вкажіть ваш ключ NASA API
NASA_API_KEY = "OMIVj1eCgWcPry5lODVp75JoDCbEEhxBhOdty5jY"


def fetch_apod(date=None, save_path="data/nasa"):
    """
    Завантажує зображення та метадані з NASA APOD API.

    :param date: Дата у форматі 'YYYY-MM-DD'. Якщо None, бере поточний день.
    :param save_path: Шлях для збереження зображення.
    :return: Шлях до завантаженого зображення і метадані.
    """
    url = "https://api.nasa.gov/planetary/apod"
    params = {"api_key": NASA_API_KEY}
    if date:
        params["date"] = date

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Створити папку для збереження, якщо її немає
    os.makedirs(save_path, exist_ok=True)

    # Завантажити зображення
    if "url" in data and data["media_type"] == "image":
        image_url = data["url"]
        image_name = os.path.join(save_path, f"apod_{date or 'today'}.jpg")
        with open(image_name, "wb") as file:
            image_data = requests.get(image_url)
            file.write(image_data.content)
        return image_name, data

    return None, data