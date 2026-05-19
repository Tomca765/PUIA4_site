import streamlit as st
import numpy as np
from skimage import io
from skimage.feature import canny
from skimage.transform import resize
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
from scipy import ndimage
import easyocr

# 1. OPTIMALIZACE: Načítáme model s omezením paměti
@st.cache_resource
def load_reader():
    # gpu=False je na Streamlit Cloud jistota, aby to nehledalo CUDA
    return easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="OCR Karet (Vysoké Rozlišení)", layout="centered")
st.title("🎴 Čtečka karet ve vysoké kvalitě")

# Načtení modelu hned na začátku
reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])
camera_file = st.sidebar.camera_input("Nebo použij kameru")

final_file = camera_file if camera_file else img_file

if final_file is not None:
    # Načtení v grayscale pro úsporu RAM při zachování ostrosti hran
    raw_img = io.imread(final_file, as_gray=True)
    
    # --- ÚPRAVA 1: Zvyšujeme cílové rozlišení zpracování ---
    # Původně bylo 1200. Zkusíme 2400, což je kompromis pro kvalitu vs RAM na Streamlit Cloudu.
    # U moderních fotek z mobilu (často 4000+ px) je stále nutné zmenšit, jinak app spadne.
    target_width = 2400 
    
    if raw_img.shape[1] > target_width:
        scale = target_width / raw_img.shape[1]
        new_shape = (int(raw_img.shape[0] * scale), target_width)
        image = resize(raw_img, new_shape, anti_aliasing=True)
        st.info(f"Obrázek zmenšen na šířku {target_width}px pro stabilitu (bylo {raw_img.shape[1]}px). Zpracovávám...")
    else:
        # Pokud je menší než target_width, ponecháme originál pro max. kvalitu
        image = raw_img
        st.info(f"Zpracovávám obrázek v plném rozlišení ({raw_img.shape[1]}x{raw_img.shape[0]})...")

    st.info("Zpracovávám... prosím čekejte.")

    # Detekce karet (Tvoje logika zůstává)
    edges = canny(image, sigma=2.0)
    filled_cards = ndimage.binary_fill_holes(dilation(edges, square(3)))
    labeled_image = label(filled_cards)
    regions = regionprops(labeled_image)

    extracted_count = 0
    
    for region in regions:
        # --- ÚPRAVA 2: Upravený limit plochy detekce ---
        # Protože je rozlišení obrázku 2x větší na šířku, plocha karty v pixelech je 4x větší.
        # Upraveno threshold z 2000 na 8000.
        if region.area > 8000: 
            extracted_count += 1
            min_row, min_col, max_row, max_col = region.bbox
            card_crop = image[min_row:max_row, min_col:max_col]

            # --- ÚPRAVA 3: ODSTRANĚNÍ ZMENŠENÍ VÝŘEZU ---
            # Původní kód zmenšoval výřez na fixních 800px, což u vysokého rozlišení ničilo kvalitu.
            # card_img_res = resize(card_crop, (int(card_crop.shape[0]*(800/card_crop.shape[1])), 800))
            # Ponecháme card_crop v plné kvalitě detekovaného rozlišení.

            # Převod na uint8 je nutný pro EasyOCR
            # Protože skimage.resize vrací float, musíme převést zpět na standardní formát obrázku.
            if card_crop.dtype != np.uint8:
                img_uint8 = (card_crop * 255).astype(np.uint8)
            else:
                img_uint8 = card_crop
            
            # Spuštění OCR
            with st.spinner(f'Čtu kartu č. {extracted_count}...'):
                result = reader.readtext(img_uint8, detail=0)
                text = " ".join(result) if result else "Text nenalezen"
            
            # Zobrazení výsledku
            col1, col2 = st.columns([1, 1])
            with col1:
                # Zobrazíme výřez v plné detekované kvalitě (Streamlit si ho automaticky na škáluje pro zobrazení v prohlížeči)
                st.image(card_crop, caption=f"Karta {extracted_count} (Vysoké rozlišení)")
            with col2:
                st.success(f"Nalezený text: **{text}**")
            st.divider()

    if extracted_count == 0:
        st.warning("Nenalezena žádná karta. Zkuste fotit z větší dálky nebo na kontrastním pozadí.")
