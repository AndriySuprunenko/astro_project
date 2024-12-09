# Astro Project

Astro Project is a Python-based tool for processing astronomical images, specifically using data from the Sloan Digital Sky Survey (SDSS). It includes functionalities such as image fetching, noise reduction, edge detection, and object annotation.

## Project Structure
```
astro_project/
├── .venv/                   # Virtual environment
├── src/
│   ├── main.py              # Entry point of the application
│   ├── sdss_api.py          # Handles API interactions with SDSS
│   ├── image_processing.py  # Functions for image processing
├── data/
│   ├── raw/                 # Raw images fetched from SDSS
│   ├── processed/           # Processed images
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## Features
- Fetch astronomical images using SDSS API.
- Preprocess images (normalization, noise reduction).
- Detect and annotate objects using edge detection and contour analysis.
- Visualize intermediate and final results.

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment tools (e.g., `venv` or `virtualenv`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/AndriySuprunenko/astro_project.git
   cd astro_project
   ```
2. Set up a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Project
1. Ensure the virtual environment is activated.
2. Execute the `main.py` script:
   ```bash
   python src/main.py
   ```

### Workflow
- The program fetches an astronomical image from SDSS based on specified coordinates (RA and Dec).
- The image undergoes processing (e.g., normalization, noise reduction, edge detection).
- Annotated results are displayed and saved in the `data/processed/` folder.

### Example Output
1. **Original Image**: Raw image fetched from SDSS.
2. **Normalized Image**: Adjusted brightness levels.
3. **Blurred Image**: Noise reduction using Gaussian blur.
4. **Edge Detection**: Object boundaries highlighted.
5. **Annotated Image**: Objects annotated with bounding boxes.

## Configuration
- To fetch a specific image, modify the RA and Dec values in `main.py`:
  ```python
  ra, dec = 180.0, 0.0  # Replace with desired coordinates
  ```

## Dependencies
- `matplotlib`: For visualization.
- `opencv-python`: For image processing.
- `requests`: For interacting with SDSS API.

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- Support for additional data sources (e.g., ESA Gaia).
- Advanced object classification using machine learning.
- Improved image alignment for multi-frame analysis.