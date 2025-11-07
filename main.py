#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
from PIL import Image


# Image Path specifications
IMG = "/home/krushang/radhey/personal/college/Doc-Scanner-Project/test/test1.png"
OUTPUT_DIR = "./out"


# helper function to see image output
def show(window_name, img, wait=True, max_width=1000, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = img.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)

    key = cv2.waitKey(0 if wait else 1) & 0xFF
    if key == 27:
        cv2.destroyAllWindows()
        raise SystemExit("Exited visualization early by user.")

    cv2.destroyWindow(window_name)


# preproceding pip line
def preprocess_for_handwriting(image):
    print("Pre Processing Pipline")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Grayscale")
    show("1. Grayscale", gray)

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    print("CLAHE Contrast Enhancement")
    show("2. Contrast Enhanced", contrast_enhanced)

    # Denoise while preserving edges (critical for handwriting)
    denoised = cv2.fastNlMeansDenoising(
        contrast_enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    print("Denoising")
    show("3. Denoised", denoised)

    # Apply bilateral filter to smooth while keeping edges
    bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
    print("Bilateral Filter")
    show("4. Bilateral Filter", bilateral)

    # Adaptive thresholding - works better for varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    print("Adaptive Thresholding")
    show("5. Adaptive Threshold", adaptive_thresh)

    # Remove small noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    morph_cleaned = cv2.morphologyEx(
        adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1
    )
    morph_cleaned = cv2.morphologyEx(
        morph_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    print("Morphological Cleanups")
    show("6. Morphological Cleanup", morph_cleaned)

    # Invert if needed (text should be black on white)
    if np.mean(morph_cleaned) > 127:
        morph_cleaned = cv2.bitwise_not(morph_cleaned)

    # Upscale for better OCR (handwriting needs higher resolution)
    h, w = morph_cleaned.shape
    scale_factor = 2
    if h < 1500:
        scale_factor = 3

    upscaled = cv2.resize(
        morph_cleaned,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    print("Upscaling")
    show("7. Upscaled (Final)", upscaled)

    return image, bilateral, upscaled


def preprocessing(img_path, out_dir="./out"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading IMAGE: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    # For handwriting, always use specialized preprocessing
    return preprocess_for_handwriting(image)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def manual_corner_selection(image):
    points = []
    clone = image.copy()

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(clone, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(
                clone,
                str(len(points) - 1),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            if len(points) > 1:
                cv2.line(clone, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(clone, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow("Select 4 Corners", clone)

    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", click_event)
    cv2.imshow("Select 4 Corners", clone)

    print("Click 4 corners in order: top-left, top-right, bottom-right, bottom-left")
    print("Press any key when done...")

    cv2.waitKey(0)
    cv2.destroyWindow("Select 4 Corners")

    if len(points) == 4:
        return np.array(points, dtype=np.float32)
    return None


def contour_detection(image, processed):
    h, w = image.shape[:2]

    print("\nContour Detection Options:")
    print("1. Auto-detect document edges")
    print("2. Use whole image")
    print("3. Manual corner selection")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "2":
        print("Using whole image...")
        doc_contour = (
            np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            .reshape(4, 1, 2)
            .astype(np.int32)
        )
        return doc_contour

    elif choice == "3":
        doc_contour = manual_corner_selection(image)
        if doc_contour is not None:
            doc_contour = order_points(doc_contour.reshape(4, 2))
            doc_contour = doc_contour.reshape(4, 1, 2).astype(np.int32)
            return doc_contour
        else:
            print("Manual selection failed. Using whole image.")
            doc_contour = (
                np.array(
                    [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
                )
                .reshape(4, 1, 2)
                .astype(np.int32)
            )
            return doc_contour

    # Auto-detect (option 1)
    edges = cv2.Canny(processed, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                doc_contour = order_points(approx.reshape(4, 2))
                doc_contour = doc_contour.reshape(4, 1, 2).astype(np.int32)

                # Show detected contour
                contour_img = image.copy()
                cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 3)
                show("Auto-detected Contour", contour_img)

                return doc_contour

    print("Auto-detection failed. Using whole image.")
    doc_contour = (
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        .reshape(4, 1, 2)
        .astype(np.int32)
    )
    return doc_contour


def perspective_transform(image, contour):
    rect = order_points(contour.reshape(4, 2))
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    show("Warped Document", warped)
    return warped


def select_language():
    print("\nSelect language for OCR:")
    print("1. English")
    print("2. Hindi")
    print("3. Gujarati")

    choice = input("Enter choice (1-3): ").strip()

    language_map = {
        "1": ("eng", "English"),
        "2": ("hin", "Hindi"),
        "3": ("guj", "Gujarati"),
    }

    if choice in language_map:
        return language_map[choice]
    else:
        print("Invalid choice. Using English by default.")
        return ("eng", "English")


def ai_spell_correction(text, lang_code):
    if "eng" not in lang_code:
        print("Spell checking only available for English text")
        return text

    spell = SpellChecker()
    lines = text.split("\n")
    corrected_lines = []
    corrections_made = 0

    print("\nPerforming AI Spell Correction...")

    for line in lines:
        if not line.strip():
            corrected_lines.append(line)
            continue

        words = line.split()
        corrected_words = []

        for word in words:
            # Keep punctuation
            prefix = ""
            suffix = ""
            clean_word = word

            # Extract leading/trailing punctuation
            while clean_word and not clean_word[0].isalnum():
                prefix += clean_word[0]
                clean_word = clean_word[1:]

            while clean_word and not clean_word[-1].isalnum():
                suffix = clean_word[-1] + suffix
                clean_word = clean_word[:-1]

            if clean_word and clean_word.isalpha() and len(clean_word) > 2:
                correction = spell.correction(clean_word.lower())
                if correction and correction != clean_word.lower():
                    # Preserve original capitalization
                    if clean_word[0].isupper():
                        correction = correction.capitalize()
                    corrected_words.append(prefix + correction + suffix)
                    corrections_made += 1
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_lines.append(" ".join(corrected_words))

    print(f"Corrections made: {corrections_made}")
    return "\n".join(corrected_lines)


def text_extraction(
    image, lang_code="eng", lang_name="English", output_file="extracted_text.txt"
):
    print(f"\nPerforming OCR ({lang_name})...")

    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Multiple OCR configurations optimized for handwriting
    ocr_configs = [
        ("--oem 1 --psm 6", "Uniform text block"),
        ("--oem 1 --psm 4", "Single column text"),
        ("--oem 1 --psm 3", "Fully automatic"),
        ("--oem 1 --psm 11", "Sparse text"),
    ]

    results = []

    for config, description in ocr_configs:
        try:
            print(f"  Trying: {description}...")

            # Get text with confidence
            data = pytesseract.image_to_data(
                gray,
                config=f"{config} -l {lang_code}",
                output_type=pytesseract.Output.DICT,
            )

            # Calculate average confidence
            confidences = [
                int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Extract text
            text = pytesseract.image_to_string(gray, config=f"{config} -l {lang_code}")
            text = text.replace("\x0c", "").strip()

            # Calculate text quality score
            word_count = len([w for w in text.split() if len(w) > 2])
            quality_score = avg_confidence * (1 + word_count / 100)

            results.append(
                {
                    "text": text,
                    "confidence": avg_confidence,
                    "quality": quality_score,
                    "method": description,
                }
            )

            print(
                f"    Confidence: {avg_confidence:.1f}%, Words: {word_count}, Score: {quality_score:.1f}"
            )

        except Exception as e:
            print(f"    Failed: {e}")
            continue

    # Select best result
    if not results:
        print("All OCR attempts failed!")
        return "", 0

    best_result = max(results, key=lambda x: x["quality"])
    best_text = best_result["text"]
    best_confidence = best_result["confidence"]

    print(f"\nBest method: {best_result['method']}")
    print(f"Final confidence: {best_confidence:.2f}%")

    # Apply spell correction for English
    if "eng" in lang_code:
        corrected_text = ai_spell_correction(best_text, lang_code)
    else:
        corrected_text = best_text

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HANDWRITTEN TEXT EXTRACTION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Language: {lang_name}\n")
        f.write(f"OCR Method: {best_result['method']}\n")
        f.write(f"Confidence: {best_confidence:.2f}%\n")
        f.write(f"Quality Score: {best_result['quality']:.2f}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("EXTRACTED TEXT:\n")
        f.write("-" * 60 + "\n\n")
        f.write(corrected_text)

        # Also save original if spell-corrected
        if corrected_text != best_text:
            f.write("\n\n" + "-" * 60 + "\n")
            f.write("ORIGINAL (BEFORE SPELL CHECK):\n")
            f.write("-" * 60 + "\n\n")
            f.write(best_text)

    print(f"\n{'=' * 60}")
    print("EXTRACTED TEXT:")
    print("=" * 60)
    print(corrected_text)
    print("=" * 60)
    print(f"\nResults saved to: {output_file}")

    return corrected_text, best_confidence


def save_processed_image(image, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")
    return output_path


def enhance_image_quality(image):
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def convert_to_high_quality_jpg(image, output_filename):
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    enhanced = enhance_image_quality(image)
    pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    pil_image.save(output_path, "JPEG", quality=95, optimize=True, dpi=(300, 300))
    print(f"High-quality JPG saved: {output_path}")
    return output_path


def convert_image_to_pdf(image, output_filename):
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    enhanced = enhance_image_quality(image)
    pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    pil_image.save(output_path, "PDF", resolution=300.0, quality=95)
    print(f"PDF saved: {output_path}")
    return output_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HANDWRITTEN TEXT EXTRACTION SYSTEM")
    print("=" * 60)

    # Step 1: Preprocessing
    image, processed, final_preprocessed = preprocessing(IMG)
    save_processed_image(final_preprocessed, "01_preprocessed.png")

    # Step 2: Contour detection (optional)
    contour = contour_detection(image, processed)

    # Step 3: Perspective transform
    if contour is not None:
        h, w = image.shape[:2]
        full_image_contour = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        ).reshape(4, 2)

        contour_reshaped = contour.reshape(4, 2)

        # Check if contour is essentially the whole image
        if not np.allclose(contour_reshaped, full_image_contour, atol=20):
            warped = perspective_transform(image, contour)
            warped_preprocessed = perspective_transform(final_preprocessed, contour)
            final_image = warped
            final_for_ocr = warped_preprocessed
        else:
            final_image = image
            final_for_ocr = final_preprocessed
    else:
        final_image = image
        final_for_ocr = final_preprocessed

    save_processed_image(final_for_ocr, "02_final_ocr_ready.png")
    show("Final Image for OCR", final_for_ocr)

    # Step 4: Language selection and OCR
    lang_code, lang_name = select_language()
    text, confidence = text_extraction(
        final_for_ocr,
        lang_code,
        lang_name,
        os.path.join(OUTPUT_DIR, "extracted_text.txt"),
    )

    # Step 5: Save output formats
    print("\n" + "=" * 60)
    print("GENERATING OUTPUT FILES")
    print("=" * 60)

    jpg_path = convert_to_high_quality_jpg(final_image, "scanned_document.jpg")
    pdf_path = convert_image_to_pdf(final_image, "scanned_document.pdf")

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  ✓ Preprocessed image: 01_preprocessed.png")
    print("  ✓ OCR-ready image: 02_final_ocr_ready.png")
    print("  ✓ High-quality JPG: scanned_document.jpg")
    print("  ✓ PDF: scanned_document.pdf")
    print("  ✓ Extracted text: extracted_text.txt")
    print(f"\nOCR Quality: {confidence:.1f}% confidence")
    print(f"Text length: {len(text)} characters")


if __name__ == "__main__":
    main()
