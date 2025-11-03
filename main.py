#!/usr/bin/env python3

# libs

import os
import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
import re

IMG = "/home/krushang/radhey/personal/college/Doc-Scanner-Project/test/test5.png"
OUTPUT_DIR = "./out"


# helper function and pre-processing
def show(window_name, img, wait=True):
    cv2.imshow(window_name, img)
    key = cv2.waitKey(0 if wait else 1) & 0xFF
    if key == 27:  # escape key
        cv2.destroyAllWindows()
        raise SystemExit("Exited visualization early by user.")
    cv2.destroyWindow(window_name)


def detect_document_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    is_digital = False

    if laplacian_var > 100 and std_brightness < 50 and edge_density < 0.1:
        is_digital = True
        print("Document type detected: DIGITAL/SCREENSHOT")
    else:
        print("Document type detected: SCANNED/PHOTO")

    return is_digital


def preprocess_digital_document(image):
    print("Applying digital document preprocessing...")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show("Original Gray", gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show("Otsu Binarization", binary)

    kernel = np.ones((1, 1), np.uint8)
    denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    show("Morphological Cleanup", denoised)

    h, w = denoised.shape
    scale_factor = 2
    if h < 1000:
        scale_factor = 3

    upscaled = cv2.resize(
        denoised, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
    )
    show("Upscaled", upscaled)

    return image, upscaled, upscaled


def preprocess_scanned_document(image):
    print("Applying scanned document preprocessing...")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    corrected = cv2.merge([l, a, b])
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
    show("Illumination Corrected", corrected)

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    show("GrayScale", gray)

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    show("Denoised", denoised)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    show("CLAHE Enhanced", enhanced)

    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    show("Sharpened", sharpened)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)
    show("Morphological Operations", opened)

    binarized = cv2.adaptiveThreshold(
        opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    show("Binarized", binarized)

    return corrected, opened, binarized


def preprocessing(img_path, out_dir="./out"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading IMAGE: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    is_digital = detect_document_type(image)

    if is_digital:
        return preprocess_digital_document(image)
    else:
        return preprocess_scanned_document(image)


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


def verify_contour(image, contour):
    contour_img = image.copy()
    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)

    for i, point in enumerate(contour):
        cv2.circle(contour_img, tuple(point[0]), 10, (0, 0, 255), -1)
        cv2.putText(
            contour_img,
            str(i),
            tuple(point[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    cv2.imshow("Detected Contour - Press any key", contour_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Detected Contour - Press any key")

    print("\nIs the detected contour correct?")
    print("1. Yes, use this contour")
    print("2. No, let me select corners manually")
    print("3. No, use whole image")

    choice = input("Enter choice (1-3): ").strip()
    return choice


def contour_detection(image, processed):
    edges1 = cv2.Canny(processed, 30, 100)
    edges2 = cv2.Canny(processed, 50, 150)
    edges3 = cv2.Canny(processed, 70, 200)

    edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    print("Detecting Multi-scale Canny Edges")
    show("Edges (Canny Multi-scale)", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found!")

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * 0.05

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    doc_contour = None

    for contour in contours[:15]:
        peri = cv2.arcLength(contour, True)

        for epsilon_factor in [
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
            0.035,
            0.04,
            0.045,
            0.05,
            0.06,
            0.07,
            0.08,
        ]:
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                rect_area = cv2.contourArea(cv2.convexHull(approx))

                if area / rect_area > 0.75:
                    doc_contour = approx
                    break

        if doc_contour is not None:
            break

    if doc_contour is None:
        print("No 4-point contour found, trying convex hull approach...")
        if contours:
            largest_contour = contours[0]
            hull = cv2.convexHull(largest_contour)
            peri = cv2.arcLength(hull, True)

            for epsilon_factor in [
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
            ]:
                approx = cv2.approxPolyDP(hull, epsilon_factor * peri, True)

                if len(approx) == 4:
                    doc_contour = approx
                    break
                elif len(approx) > 4:
                    points = approx.reshape(-1, 2)

                    top_left = points[np.argmin(points.sum(axis=1))]
                    bottom_right = points[np.argmax(points.sum(axis=1))]
                    top_right = points[np.argmin(np.diff(points, axis=1))]
                    bottom_left = points[np.argmax(np.diff(points, axis=1))]

                    doc_contour = np.array(
                        [[top_left], [top_right], [bottom_right], [bottom_left]],
                        dtype=np.int32,
                    )
                    break

    if doc_contour is None:
        print("\nCould not automatically detect document corners.")
        print("Choose an option:")
        print("1. Use whole image")
        print("2. Manually select corners")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            h, w = image.shape[:2]
            doc_contour = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
            )
            print("Using whole image as document area")
        elif choice == "2":
            doc_contour = manual_corner_selection(image)
            if doc_contour is None:
                print("Manual selection failed. Using whole image.")
                h, w = image.shape[:2]
                doc_contour = np.array(
                    [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
                )
        else:
            print("Invalid choice. Using whole image.")
            h, w = image.shape[:2]
            doc_contour = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
            )
    else:
        doc_contour = doc_contour.reshape(4, 2)
        doc_contour = order_points(doc_contour)
        doc_contour = doc_contour.reshape(4, 1, 2).astype(np.int32)

        user_choice = verify_contour(image, doc_contour)

        if user_choice == "2":
            print("\nManual corner selection mode...")
            manual_contour = manual_corner_selection(image)
            if manual_contour is not None:
                doc_contour = manual_contour
                doc_contour = doc_contour.reshape(4, 2)
                doc_contour = order_points(doc_contour)
                doc_contour = doc_contour.reshape(4, 1, 2).astype(np.int32)
            else:
                print("Manual selection failed. Using automatically detected contour.")
        elif user_choice == "3":
            print("\nUsing whole image...")
            h, w = image.shape[:2]
            doc_contour = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
            )
            doc_contour = doc_contour.reshape(4, 2)
            doc_contour = order_points(doc_contour)
            doc_contour = doc_contour.reshape(4, 1, 2).astype(np.int32)

        return doc_contour

    doc_contour = doc_contour.reshape(4, 2)
    doc_contour = order_points(doc_contour)
    doc_contour = doc_contour.reshape(4, 1, 2).astype(np.int32)

    contour_img = image.copy()
    cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 3)

    for i, point in enumerate(doc_contour):
        cv2.circle(contour_img, tuple(point[0]), 10, (0, 0, 255), -1)
        cv2.putText(
            contour_img,
            str(i),
            tuple(point[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    print("Contour Detection Done!")
    show("Contour Detection", contour_img)

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

    print("Perspective Transform Applied")
    show("Warped Document", warped)

    return warped


def select_language():
    print("\nSelect language for OCR:")
    print("1. English only")
    print("2. Hindi only")
    print("3. Gujarati only")
    print("4. English + Hindi")
    print("5. English + Gujarati")
    print("6. Hindi + Gujarati")
    print("7. English + Hindi + Gujarati")

    choice = input("Enter choice (1-7): ").strip()

    language_map = {
        "1": ("eng", "English"),
        "2": ("hin", "Hindi"),
        "3": ("guj", "Gujarati"),
        "4": ("eng+hin", "English + Hindi"),
        "5": ("eng+guj", "English + Gujarati"),
        "6": ("hin+guj", "Hindi + Gujarati"),
        "7": ("eng+hin+guj", "English + Hindi + Gujarati"),
    }

    if choice in language_map:
        return language_map[choice]
    else:
        print("Invalid choice. Using English only by default.")
        return ("eng", "English")


def ai_spell_correction(text, lang_code):
    if "eng" not in lang_code:
        print("Spell checking only available for English text")
        return text

    spell = SpellChecker()

    lines = text.split("\n")
    corrected_lines = []

    print("\nPerforming AI-based Spell Correction...")
    corrections_made = 0

    for line in lines:
        words = line.split()
        corrected_words = []

        for word in words:
            clean_word = re.sub(r"[^\w\s]", "", word)

            if clean_word and clean_word.isalpha() and len(clean_word) > 2:
                correction = spell.correction(clean_word)
                if correction and correction != clean_word.lower():
                    corrected_words.append(correction)
                    corrections_made += 1
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_lines.append(" ".join(corrected_words))

    print(f"Spell correction complete. {corrections_made} corrections made.")
    return "\n".join(corrected_lines)


def text_extraction(image, lang_code, lang_name, output_file="extracted_text.txt"):
    print(f"\nPerforming OCR with {lang_name}...")

    custom_config = f"--oem 1 --psm 6 -l {lang_code}"

    text = pytesseract.image_to_string(image, config=custom_config)
    text = text.replace("\x0c", "").strip()

    print("\nTrying alternative OCR configurations for better accuracy...")

    configs_to_try = [
        "--oem 1 --psm 3",
        "--oem 1 --psm 4",
        "--oem 1 --psm 6",
        "--oem 3 --psm 6",
    ]

    best_text = text
    best_confidence = 0

    for config in configs_to_try:
        try:
            temp_config = f"{config} -l {lang_code}"
            data = pytesseract.image_to_data(
                image, config=temp_config, output_type=pytesseract.Output.DICT
            )
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]

            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_text = pytesseract.image_to_string(image, config=temp_config)
                    best_text = best_text.replace("\x0c", "").strip()
        except:
            continue

    print(f"Best OCR confidence: {best_confidence:.2f}%")

    corrected_text = ai_spell_correction(best_text, lang_code)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"AI-Powered Document Scanner - OCR Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Language: {lang_name}\n")
        f.write(f"OCR Confidence: {best_confidence:.2f}%\n")
        f.write(f"{'-' * 50}\n")
        f.write("EXTRACTED TEXT:\n")
        f.write(f"{'-' * 50}\n")
        f.write(corrected_text)

    print(f"\nExtracted Text ({lang_name}):\n{'-' * 50}")
    print(corrected_text)
    print(f"{'-' * 50}")
    print(f"Text saved to: {output_file}")

    return corrected_text


def save_processed_image(image, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("AI-POWERED DOCUMENT SCANNER WITH OCR")
    print("=" * 60)

    image, processed, binarized = preprocessing(IMG)
    save_processed_image(binarized, "01_preprocessed.png")

    print("\nSkip contour detection for digital documents? (y/n): ", end="")
    skip_contour = input().strip().lower()

    if skip_contour == "y":
        print("Using whole image...")
        warped_bin = binarized
    else:
        contour = contour_detection(image, processed)
        warped = perspective_transform(image, contour)
        save_processed_image(warped, "02_warped.png")

        warped_gray = (
            cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            if len(warped.shape) == 3
            else warped
        )

        warped_bin = cv2.threshold(
            warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    show("Final Document for OCR", warped_bin)
    save_processed_image(warped_bin, "03_final_ocr_ready.png")

    lang_code, lang_name = select_language()
    text = text_extraction(warped_bin, lang_code, lang_name)

    print("\n" + "=" * 60)
    print("AI-POWERED DOCUMENT PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
