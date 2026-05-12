import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import io
from skimage.feature import canny
from skimage.transform import resize, probabilistic_hough_line, rotate
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
from scipy import ndimage
import easyocr

# Optimalizace: Načtení modelu OCR proběhne jen jednou při startu
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.set_page_config(page_title="OCR Detektor Karet", layout="centered")
st.title("🎴 OCR Detektor a Čtečka Karet")
st.write("Nahrajte fotku nebo použijte fotoaparát k extrakci textu z karet.")

# --- VSTUP OD UŽIVATELE ---
option = st.radio("Vyberte zdroj obrázku:", ("Fotoaparát", "Nahrát soubor"))

if option == "Fotoaparát":
    img_file = st.camera_input("Vyfoťte karty")
else:
    img_file = st.file_uploader("Vyberte obrázek (jpg, png)", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Načtení obrázku
    image = io.imread(img_file, as_gray=True)
    
    with st.spinner('Zpracovávám obrázek... (může to chvíli trvat)'):
        # Škálování
        scale_factor = 3200 / image.shape[1]
        new_shape = (int(image.shape[0] * scale_factor), 3200)
        image_res = resize(image, new_shape, anti_aliasing=True)

        # Detekce karet
        edges = canny(image_res, sigma=3.0)
        thick_edges = dilation(edges, square(5))
        filled_cards = ndimage.binary_fill_holes(thick_edges)
        labeled_image = label(filled_cards)
        regions = regionprops(labeled_image)

        extracted_cards = []
        for region in regions:
            if region.area > 5000:
                min_row, min_col, max_row, max_col = region.bbox
                card_crop = image_res[min_row:max_row, min_col:max_col]
                extracted_cards.append(card_crop)

        st.success(f"Nalezeno karet: {len(extracted_cards)}")

        results_data = []

        # Zpracování každé karty
        for i, card_img in enumerate(extracted_cards):
            # Skew Correction
            card_scale_factor = 3200 / card_img.shape[1]
            card_new_shape = (int(card_img.shape[0] * card_scale_factor), 3200)
            card_img = resize(card_img, card_new_shape, anti_aliasing=True)

            edges_for_skew = canny(card_img, sigma=3.0)
            lines_for_skew = probabilistic_hough_line(edges_for_skew, threshold=10, line_length=500, line_gap=100)

            angles = []
            for p0, p1 in lines_for_skew:
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                angle = math.degrees(math.atan2(dy, dx))
                if -45 < angle < 45: angles.append(angle)
                elif angle > 135: angles.append(angle - 180)
                elif angle < -135: angles.append(angle + 180)

            skew_angle = np.median(angles) if angles else 0
            straight_img = rotate(card_img, skew_angle, resize=True, mode='edge')

            # Detekce boxu pro text
            straight_edges = canny(straight_img, sigma=5.0)
            straight_lines = probabilistic_hough_line(straight_edges, threshold=10, line_length=500, line_gap=100)

            horizontal_ys, vertical_xs = [], []
            for p0, p1 in straight_lines:
                if abs(p1[1] - p0[1]) < 50: horizontal_ys.extend([p0[1], p1[1]])
                elif abs(p1[0] - p0[0]) < 50: vertical_xs.extend([p0[0], p1[0]])

            if horizontal_ys and vertical_xs:
                top_y, bottom_y = min(horizontal_ys), max(horizontal_ys)
                left_x, right_x = min(vertical_xs), max(vertical_xs)
                h, w = bottom_y - top_y, right_x - left_x
                name_box_crop = straight_img[int(top_y+(h*0.035)):int(top_y+(h*0.3)), int(left_x+(w*0.055)):int(right_x-(w*0.25))]
            else:
                name_box_crop = straight_img[0:int(straight_img.shape[0]*0.15), :]

            # OCR
            image_to_process = (name_box_crop * 255).astype(np.uint8) if name_box_crop.dtype != np.uint8 else name_box_crop
            ocr_results = reader.readtext(image_to_process)
            
            if not ocr_results:
                card_name = "[Text nenalezen]"
            else:
                ocr_results = sorted(ocr_results, key=lambda x: x[0][0][0])
                card_name = " ".join([text for (_, text, _) in ocr_results])

            results_data.append((name_box_crop, card_name))

        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        for i, (crop, name) in enumerate(results_data):
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(crop, caption=f"Výřez karty {i+1}", use_column_width=True)
                with col2:
                    st.subheader(f"Karta {i+1}")
                    st.code(name)
                st.divider()