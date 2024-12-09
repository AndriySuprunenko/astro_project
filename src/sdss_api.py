import requests
import os
from config import RAW_DATA_PATH

def fetch_sdss_image(ra, dec, scale=0.2, width=512, height=512):
    """Завантажує зображення із SDSS."""
    url = f"https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
    params = {
        "ra": ra,
        "dec": dec,
        "scale": scale,
        "width": width,
        "height": height
    }

    # Перевірка існування папки і створення, якщо її немає
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

    response = requests.get(url, params=params)
    if response.status_code == 200:
        filepath = os.path.join(RAW_DATA_PATH, f"sdss_{ra}_{dec}.jpg")
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Зображення збережено: {filepath}")
        return filepath
    else:
        print(f"Помилка {response.status_code}: {response.text}")
        return None