import streamlit as st
import numpy as np
import math
from skimage import io
from skimage.feature import canny
from skimage.transform import resize, probabilistic_hough_line, rotate
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
from scipy import ndimage
import easyocr

# 1. OPTIMALIZACE: Načítáme model s omezením paměti
@st.cache_resource
def load_reader():
    # gpu=False je na Streamlit Cloud jistota, aby to nehledalo CUDA
    return easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="OCR Karet", layout="centered")
st.title("🎴 Úsporná čtečka karet")

# Načtení modelu hned na začátku
reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])
camera_file = st.sidebar.camera_input("Nebo použij kameru")

final_file = camera_file if camera_file else img_file

if final_file is not None:
    # Načtení v nižším rozlišení pro úsporu RAM
    raw_img = io.imread(final_file, as_gray=True)
    
    # 2. ZMĚNA: Snížení rozlišení z 3200 na 1200 (Klíčové pro stabilitu!)
    target_width = 1200
    scale = target_width / raw_img.shape[1]
    new_shape = (int(raw_img.shape[0] * scale), target_width)
    image = resize(raw_img, new_shape, anti_aliasing=True)
    
    st.info("Zpracovávám... prosím čekejte.")

    # Detekce karet (tvoje logika zůstává, jen na menším obrázku)
    edges = canny(image, sigma=2.0)
    filled_cards = ndimage.binary_fill_holes(dilation(edges, square(3)))
    labeled_image = label(filled_cards)
    regions = regionprops(labeled_image)

    extracted_count = 0
    
    for region in regions:
        if region.area > 2000: # Upravený limit pro menší rozlišení
            extracted_count += 1
            min_row, min_col, max_row, max_col = region.bbox
            card_crop = image[min_row:max_row, min_col:max_col]

            # Skew correction (zjednodušeno pro rychlost)
            card_img_res = resize(card_crop, (int(card_crop.shape[0]*(800/card_crop.shape[1])), 800))
            
            # OCR přímo na výřezu
            # Převod na uint8 je nutný pro EasyOCR
            img_uint8 = (card_img_res * 255).astype(np.uint8)
            
            # Spuštění OCR
            with st.spinner(f'Čtu kartu č. {extracted_count}...'):
                result = reader.readtext(img_uint8, detail=0)
                text = " ".join(result) if result else "Text nenalezen"
            
            # Zobrazení výsledku
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(card_img_res, caption=f"Karta {extracted_count}")
            with col2:
                st.success(f"Nalezený text: **{text}**")
            st.divider()

    if extracted_count == 0:
        st.warning("Nenalezena žádná karta. Zkuste fotit z větší dálky nebo na kontrastním pozadí.")
