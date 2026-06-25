import cv2
import pytesseract
import os

# For Windows users only
# Uncomment and change the path if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_IMAGE = "input_images/sample.jpg"

if not os.path.exists(INPUT_IMAGE):
    print("Image not found!")
    exit()

image = cv2.imread(INPUT_IMAGE)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# OCR
text = pytesseract.image_to_string(gray)

print("\n===== EXTRACTED TEXT =====\n")
print(text)

# Save extracted text
os.makedirs("output", exist_ok=True)

with open("output/extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(text)

# Draw OCR boxes
boxes = pytesseract.image_to_data(
    gray,
    output_type=pytesseract.Output.DICT
)

n_boxes = len(boxes['text'])

for i in range(n_boxes):

    if int(boxes['conf'][i]) > 30:

        x = boxes['left'][i]
        y = boxes['top'][i]
        w = boxes['width'][i]
        h = boxes['height'][i]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

# Save processed image
cv2.imwrite("output/processed_image.jpg", image)

print("\nFiles Saved Successfully!")
print("1. output/extracted_text.txt")
print("2. output/processed_image.jpg")

cv2.imshow("OCR Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()