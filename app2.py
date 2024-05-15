import cv2
import imutils
import numpy as np
import os
import pytesseract
import pycountry
import streamlit as st
from PIL import Image

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load images
main_image = Image.open('static/main_banner.png')
plat_image = Image.open('static/plat.png')

# Paths upload & download
upload_path = "uploads/"
download_path = "downloads/"

# Page configuration
st.set_page_config(
    page_title="Plate Detection",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Variabel negara
countries = [country.name for country in pycountry.countries]

# CSS for custom styles
st.markdown("""
    <style>
    .stButton button.st-emotion-cache-19rxjzo.ef3psqc12 {
        width: 100%;
    }
    .justified-text {
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

# Landing page
def landing_page():
    st.image(main_image, use_column_width='auto')
    st.title('Recognition Number Plate 🚘🚙')
    st.write("### Welcome to Recognition Number Plate 🚘🚙")
    st.markdown(
        """
        <p class="justified-text">
        Kemajuan dalam teknologi komputer, terutama dalam bidang computer vision dan optical character recognition (OCR), telah menginspirasi pengembangan sistem deteksi dan pengenalan plat nomor kendaraan.
        Sebelumnya, proses pengenalan plat nomor dilakukan secara manual oleh petugas, yang memakan waktu dan tenaga.
        Tujuan utama pengembangan sistem ini adalah untuk memberikan solusi yang lebih efisien dan akurat dalam mengenali plat nomor kendaraan.
        Penggunaan teknologi computer vision dan OCR memungkinkan deteksi otomatis area plat nomor dari gambar kendaraan yang diunggah serta identifikasi teks pada plat nomor dengan cepat dan akurat.
        Aplikasi sistem ini mencakup pengawasan lalu lintas untuk tujuan keamanan, penegakan hukum, manajemen parkir, dan penggunaan sumber daya manusia yang lebih efisien.
        </p>
        <br>
        """, unsafe_allow_html=True)
    if st.button("Mulai", key="mulai_button"):
        st.session_state.get_started_pressed = True
        st.experimental_rerun()

# Main page
def main_page():
    st.image(main_image, use_column_width='auto')
    st.sidebar.title('Recognition Number Plate 🚘🚙')
    st.sidebar.image(plat_image, use_column_width='auto')
    st.sidebar.info("🚘🚙 Recognition Number Plate adalah solusi cerdas untuk mengenali plat nomor kendaraan secara instan. Dengan teknologi mutakhir, aplikasi ini memungkinkan pengguna untuk dengan mudah memproses gambar plat nomor kendaraan dan mendapatkan informasi yang dibutuhkan tanpa repot. Dari pengawasan lalu lintas hingga manajemen parkir, aplikasi ini meningkatkan efisiensi tanpa kompromi. Dengan kemampuan mengenali berbagai jenis plat nomor, Recognition Number Plate adalah alat esensial bagi mereka yang mengutamakan kecepatan dan ketepatan dalam mengelola informasi kendaraan.")
    if st.sidebar.button("Menuju Landing Page"):
        # Set the session state variable to False and rerun the script
        st.session_state.get_started_pressed = False
        st.experimental_rerun()
    st.title('Recognition Number Plate 🚘🚙')
    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    negara = st.selectbox("Pilih Negara", countries)
    uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png", "jpg", "bmp", "jpeg"])

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        with st.spinner(f"Working... 💫"):
            uploaded_image = os.path.join(upload_path, uploaded_file.name)
            img = Image.open(uploaded_image)
            st.progress(100)
            st.success('✅ Upload Successful !!')
            st.image(img, width=300, caption='Original Image', use_column_width=True)

            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (600, 400))

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.bilateralFilter(gray, 19, 15, 15)
            edged = cv2.Canny(gray, 200, 600)

            contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None

            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 4:
                    screenCnt = approx
                    break

            if screenCnt is not None:
                cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
                new_image = cv2.bitwise_and(img, img, mask=mask)

                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

                col1, col2, col3 = st.columns([1, 0.5, 1])
                with col1:
                    st.image(img, caption='Contour Image', width=300)
                with col2:
                    st.empty()
                with col3:
                    st.image(Cropped, caption='Cropped Image', width=300)

                text = pytesseract.image_to_string(Cropped, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789QWERTYUIOPASDFGHJKLZXCVBNM')
                st.write("Detected license plate Number is:", text)
                st.write("Plat dari negara :", negara)
            else:
                st.write("No contour detected")
    else:
        st.warning('⚠ Please upload your Image 😯')

if 'get_started_pressed' not in st.session_state:
    st.session_state.get_started_pressed = False

if not st.session_state.get_started_pressed:
    landing_page()
else:
    main_page()

st.markdown("<br><hr><center>Made with ❤️ by <strong>Kelompok 6 Computer vision</strong></a></center><hr>", unsafe_allow_html=True)
