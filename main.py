import os
import glob
import base64
import json
import cv2
import pandas as pd
import google.generativeai as genai
from typing import Dict, Any, Optional

# Constants for file paths and parameters
INPUT_DIR = "input"
OUTPUT_FILE = "output_report.xlsx"
MODEL_NAME = "gemini-2.5-flash"

# System prompt configured as per the requirement
SYSTEM_PROMPT = """
제공된 이미지는 '주금납입능력 확인 등 관련 확약서'입니다. 다음 항목을 정확히 추출하여 JSON으로 반환하십시오. 수기로 작성된 부분도 포함해야 합니다.

{
  "document_type": "문서 종류 (예: 주금납입능력 확약서)",
  "company_name": "회사명 (예: (주)세븐나인파트너스)",
  "address": "주소 (수기 작성된 부분 포함)",
  "payment_ability_amount": "주금납입능력 금액 (숫자와 원 단위까지, 예: 3,000,771,186원)",
  "is_representative_signed": "대표이사 서명 또는 직인 날인 여부 (true/false)",
  "missing_fields": ["추출하지 못한 빈 항목의 이름 목록"]
}
"""

def setup_gemini_api() -> None:
    """
    Initialize Gemini API configuration.
    Assumes GEMINI_API_KEY is available in environment variables.
    """
    api_key = "AIzaSyDES9LkuilFJN4hxqsHVRWXUSoeTzJECRA"
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing. Please set it appropriately.")
    genai.configure(api_key=api_key)

def preprocess_image(image_path: str) -> Optional[str]:
    """
    Load an image, apply grayscale, denoising, and contrast enhancement.
    Returns the processed image as a base64 encoded string.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None

    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Denoising to handle low-quality monitor captures
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 3: Contrast Enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)

    # Convert processed image to base64
    success, buffer = cv2.imencode(".jpg", enhanced_image)
    if not success:
        print(f"Error encoding image to JPEG: {image_path}")
        return None

    base64_string = base64.b64encode(buffer).decode("utf-8")
    return base64_string

def extract_document_features(base64_img: str) -> Dict[str, Any]:
    """
    Call Gemini API with base64 encoded image to extract information.
    """
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"}
    )

    image_part = {
        "mime_type": "image/jpeg",
        "data": base64_img
    }

    try:
        response = model.generate_content(
            contents=[image_part, "Please extract the required details from the attached image strictly matching the specified JSON format. Ensure highly accurate OCR interpretation."],
            request_options={"timeout": 60}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"API Request failed: {e}")
        return {
            "document_type": "API Error",
            "missing_fields": ["All (API Error)"]
        }

def analyze_and_export_documents() -> None:
    """
    Iterate over images in input directory, process them, and save to an Excel file.
    """
    try:
        setup_gemini_api()
    except ValueError as e:
        print(e)
        return

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Input directory '{INPUT_DIR}' created. Please add document images and run the script again.")
        return

    # Look for common image extensions
    search_patterns = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []
    for pattern in search_patterns:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, pattern.upper())))

    if not image_paths:
        print(f"No valid image files found in '{INPUT_DIR}' directory.")
        return

    report_data = []
    print(f"Starting Document RPA Extraction Pipeline... Found {len(image_paths)} images.")
    
    for idx, path in enumerate(image_paths, start=1):
        filename = os.path.basename(path)
        print(f"[{idx}/{len(image_paths)}] Processing file: {filename}...")

        base64_data = preprocess_image(path)
        if not base64_data:
            print(f"[{idx}/{len(image_paths)}] Skipped file (Pre-processing failed): {filename}")
            continue

        extracted_info = extract_document_features(base64_data)

        # Business Logic: Determine if manual inspection is needed
        missing_fields = extracted_info.get("missing_fields", [])
        is_signed = extracted_info.get("is_representative_signed", False)

        needs_inspection = "N(정상)"
        if (len(missing_fields) > 0) or (not is_signed):
            needs_inspection = "Y(수동확인)"

        # Prepare payload row based on extracted JSON keys
        report_data.append({
            "파일명": filename,
            "검수_필요여부": needs_inspection,
            "문서_종류": extracted_info.get("document_type", "미기입"),
            "주금납입능력_금액": extracted_info.get("payment_ability_amount", "미기입"),
            "회사명": extracted_info.get("company_name", "미기입"),
            "주소": extracted_info.get("address", "미기입"),
            "직인_날인여부": "날인" if is_signed else "누락",
            "비고(누락항목)": ", ".join(missing_fields) if missing_fields else ""
        })

    # Export compiled result mapping mapping to excel
    if report_data:
        df = pd.DataFrame(report_data)
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"Pipeline successfully completed. Validation report saved at: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No valid data exported.")

if __name__ == "__main__":
    analyze_and_export_documents()
