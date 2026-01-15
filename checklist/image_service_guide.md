# Image Service (image_service) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `image_service` module provides functionality for translating text directly within images ("In-Place Image Translation"). It detects text using OCR, translates it, and overlays the translated text back onto the original image.

1.  **Image Upload & Validation (`router.py`)**:
    -   Accepts image uploads (JPG, PNG, WEBP).
    -   Validates content type and extension.
    -   Converts input into a standard NumPy array format for processing.

2.  **OCR Processing (`ocr_service.py`)**:
    -   **Engine**: Uses **DocTR** (Document Text Recognition), a deep learning-based OCR library.
    -   **Optimization**: Automatically resizes large images (`MAX_IMAGE_DIMENSION = 2048`) to prevent OOM errors and improve speed.
    -   **Output**: Returns precise bounding boxes and detected text strings. It converts DocTR's relative coordinates to absolute pixel coordinates.

3.  **Translation (`translation_service.py`)**:
    -   **Engine**: Google Gemini via LangChain.
    -   **Strategy**: Batches all detected text blocks into a single LLM prompt to ensure context and reduce API calls.
    -   **Constraint**: Enforces strict line-by-line output to match the input list length.

4.  **Image Reconstruction (`image_processing.py`)**:
    -   **Logic**: Iterates through original bounding boxes.
    -   **Drawing**:
        -   Draws a semi-transparent white background over the original text.
        -   Dynamically calculates font size to fit the translated text within the box.
        -   Centers the text within the bounding box.
    -   **Font**: Uses `NotoSansTC-Regular.ttf` for proper Traditional Chinese rendering.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `image_service/router.py` | Main API entry point (`/translate_image`). Orchestrates the pipeline. |
| `image_service/ocr_service.py` | Wrapper for the DocTR OCR engine. Handles resizing and coordinate normalization. |
| `image_service/translation_service.py` | Handles batch translation of text strings using Gemini. |
| `image_service/image_processing.py` | Contains graphics logic (Pillow) for drawing text and boxes on images. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
**Translate Image:**
`POST /imagemd/translate_image`
-   **Form Data**: `file=@/path/to/image.jpg`
-   **Auth**: Requires JWT token.
-   **Response**: Returns the modified image (JPEG).

### Standalone Testing
To test OCR or image processing logic:

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# No dedicated script provided by default, but you can import the service in a python shell:
# >>> from image_service.ocr_service import perform_ocr
# >>> ...
```

## 4. Dependencies

### Internal Modules
-   `core`: Authentication, LLM factory.
-   `fonts`: Contains the required font file (`NotoSansTC-Regular.ttf`).

### External Libraries
-   `fastapi`: API framework.
-   `python-doctr`: OCR engine (requires TensorFlow or PyTorch).
-   `Pillow`: Image manipulation.
-   `numpy`: Array handling for images.
-   `langchain`: Translation chain.
