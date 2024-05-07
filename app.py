import cv2
import imutils
import numpy as np
import os
import pytesseract
import streamlit as st
from PIL import Image

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Page configuration
st.set_page_config(
    page_title="Plate Detection",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Function to show download success
@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def download_success():
    st.balloons()
    st.success('✅ Download Successful !!')

# Load images
top_image = Image.open('static/banner_top.png')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')

# Sidebar and main page layout
upload_path = "uploads/"
download_path = "downloads/"

st.image(main_image, use_column_width='auto')
st.title('Automatic Number Plate Recognition 🚘🚙')
st.sidebar.image(top_image, use_column_width='auto')
st.sidebar.header('Input 🛠')

# Image upload and processing
st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
uploaded_file = st.sidebar.file_uploader("Upload Image of car's number plate 🚓", type=["png", "jpg", "bmp", "jpeg"])
st.sidebar.image(bottom_image, use_column_width='auto')

if uploaded_file is not None:
    # Save uploaded image
    with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())
    with st.spinner(f"Working... 💫"):
        # Load and display original image
        uploaded_image = os.path.join(upload_path, uploaded_file.name)
        img = Image.open(uploaded_image)
        st.progress(100)
        st.success('✅ Upload Successful !!')
        st.image(img, caption='Original Image', use_column_width=True)

        # Load image for processing
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 400))

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(gray, 30, 200)

        # Contour detection
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        # License plate detection
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        # License plate extraction and OCR
        if screenCnt is not None:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

            # Display processed images and extracted text
            st.image(img, caption='Contour Image', use_column_width=True)
            st.image(Cropped, caption='Cropped Image', use_column_width=True)

            text = pytesseract.image_to_string(Cropped, config='--psm 11')
            st.write("Detected license plate Number is:", text)

            img = cv2.resize(img, (500, 300))
            Cropped = cv2.resize(Cropped, (400, 200))

            # Download output image
            if st.button("Download Output Image 📷"):
                with open(os.path.join(download_path, uploaded_file.name), "wb") as file:
                    file.write(uploaded_file.read())
                download_success()
        else:
            st.write("No contour detected")

else:
    st.warning('⚠ Please upload your Image 😯')

# Footer
st.markdown("<br><hr><center>Made with ❤️ by <strong>Kelompok 6 Computer vision</strong></a></center><hr>", unsafe_allow_html=True)
