import cv2
import pytesseract
import numpy as np
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = \
r"C:\Program Files\Tesseract-OCR\tesseract.exe"



image = cv2.imread("sample.jpg")

if image is None:
    print("sample.jpg not found")
    exit()

output_image = image.copy()


# IMAGE PREPROCESSING


gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# =====================================================
# OCR TEXT EXTRACTION
# =====================================================

text = pytesseract.image_to_string(
    thresh
)

print("\n" + "=" * 50)
print("EXTRACTED TEXT")
print("=" * 50)
print(text)


# OCR WORD DATA


ocr_data = pytesseract.image_to_data(
    thresh,
    output_type=pytesseract.Output.DICT
)

words = []
confidences = []

for i in range(len(ocr_data["text"])):

    word = ocr_data["text"][i]

    try:
        conf = float(
            ocr_data["conf"][i]
        )
    except:
        conf = 0

    if conf > 30 and word.strip() != "":

        words.append(word)
        confidences.append(conf)

        x = ocr_data["left"][i]
        y = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]

        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_image,
            word,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


# OCR STATISTICS


if len(confidences) > 0:
    avg_ocr_conf = sum(confidences) / len(confidences)
else:
    avg_ocr_conf = 0

print("\n" + "=" * 50)
print("OCR STATISTICS")
print("=" * 50)

print(f"Total Words : {len(words)}")
print(f"Average OCR Confidence : {avg_ocr_conf:.2f}%")


# OCR CONFIDENCE TABLE


df = pd.DataFrame({
    "Word": words,
    "Confidence": confidences
})

print("\nWORD CONFIDENCE TABLE\n")
print(df)


# SAVE OCR TEXT


with open(
    "extracted_text.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(text)


# MOBILENET SSD OBJECT DETECTION


CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

print("\n" + "=" * 50)
print("OBJECT DETECTION")
print("=" * 50)

net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    0.007843,
    (300, 300),
    127.5
)

net.setInput(blob)

detections = net.forward()

object_count = 0

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:

        idx = int(
            detections[0, 0, i, 1]
        )

        box = detections[0, 0, i, 3:7] * \
              np.array([w, h, w, h])

        (startX,
         startY,
         endX,
         endY) = box.astype("int")

        label = \
            f"{CLASSES[idx]}: {confidence*100:.2f}%"

        object_count += 1

        print(label)

        cv2.rectangle(
            output_image,
            (startX, startY),
            (endX, endY),
            (255, 0, 0),
            2
        )

        y = startY - 10

        if y < 10:
            y = startY + 20

        cv2.putText(
            output_image,
            label,
            (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

# FINAL STATS

print("\n" + "=" * 50)
print("FINAL REPORT")
print("=" * 50)

print(f"Words Detected      : {len(words)}")
print(f"OCR Confidence      : {avg_ocr_conf:.2f}%")
print(f"Objects Detected    : {object_count}")

# SAVE OUTPUT

cv2.imwrite(
    "output.jpg",
    output_image
)

print("\noutput.jpg saved")
print("extracted_text.txt saved")
# DISPLAY

cv2.imshow(
    "Final Output",
    output_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
