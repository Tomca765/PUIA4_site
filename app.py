import streamlit as st
import numpy as np
import math
from skimage import io
from skimage.feature import canny
from skimage.transform import resize
from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
from scipy import ndimage
import easyocr
import requests
import urllib.parse

# 1. OPTIMALIZACE: Načítáme model s omezením paměti
@st.cache_resource
def load_reader():
    # gpu=False je na Streamlit Cloud jistota, aby to nehledalo CUDA
    return easyocr.Reader(['en'], gpu=False)

# Funkce pro hledání na Scryfallu
def fetch_scryfall_card(ocr_result):
    if not ocr_result:
        return None, "Žádný text k vyhledání."

    # Předpokládáme, že první přečtený řádek je jméno karty
    candidate_name = ocr_result[0]
    
    # Scryfall API pro fuzzy vyhledávání (odolné vůči drobným překlepům z OCR)
    url = f"https://api.scryfall.com/cards/named?fuzzy={urllib.parse.quote(candidate_name)}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Získání URL obrázku (ošetření karet s více tvářemi/oboustranných)
            image_url = data.get('image_uris', {}).get('normal')
            if not image_url and 'card_faces' in data:
                image_url = data['card_faces'][0].get('image_uris', {}).get('normal')
                
            return data.get('name'), image_url
        else:
            return None, f"Karta nenalezena pro dotaz: '{candidate_name}'"
    except Exception as e:
        return None, f"Chyba API: {e}"

st.set_page_config(page_title="OCR Karet", layout="centered")
st.title("🎴 Úsporná čtečka karet se Scryfallem")

# Načtení modelu hned na začátku
reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])
camera_file = st.sidebar.camera_input("Nebo použij kameru")

final_file = camera_file if camera_file else img_file

if final_file is not None:
    # Načtení v původním plném rozlišení (necháme si ho pro OCR)
    raw_img = io.imread(final_file, as_gray=True)
    
    # Zmenšení pouze pro účely detekce hran (Klíčové pro stabilitu RAM!)
    target_width = 1200
    scale = target_width / raw_img.shape[1]
    new_shape = (int(raw_img.shape[0] * scale), target_width)
    image = resize(raw_img, new_shape, anti_aliasing=True)
    
    st.info("Zpracovávám obrázek a hledám karty... prosím čekejte.")

    # Detekce karet (pracuje na zmenšeném obrázku)
    edges = canny(image, sigma=2.0)
    filled_cards = ndimage.binary_fill_holes(dilation(edges, square(3)))
    labeled_image = label(filled_cards)
    regions = regionprops(labeled_image)

    extracted_count = 0
    
    for region in regions:
        if region.area > 2000: # Limit pro rozlišení detekčního obrázku
            extracted_count += 1
            min_row, min_col, max_row, max_col = region.bbox
            
            # ZMĚNA: Přepočet detekovaných souřadnic zpět na původní plné rozlišení
            min_row_raw = max(0, int(min_row / scale))
            min_col_raw = max(0, int(min_col / scale))
            max_row_raw = min(raw_img.shape[0], int(max_row / scale))
            max_col_raw = min(raw_img.shape[1], int(max_col / scale))
            
            # ZMĚNA: Výřez provádíme přímo z originálu, abychom neztratili detaily textu
            card_crop_high_res = raw_img[min_row_raw:max_row_raw, min_col_raw:max_col_raw]
            
            # Převod na uint8 z plného rozlišení (odstranili jsme vynucených 800px šířky)
            img_uint8 = (card_crop_high_res * 255).astype(np.uint8)
            
            # Spuštění OCR na ultra-ostré kartě
            with st.spinner(f'Čtu kartu č. {extracted_count} a hledám na Scryfallu...'):
                result = reader.readtext(img_uint8, detail=0)
                text = " ".join(result) if result else "Text nenalezen"
                
                # Dotaz na Scryfall
                scryfall_name, scryfall_img = fetch_scryfall_card(result)
            
            # Zobrazení výsledku ve 3 sloupcích pro lepší přehled
            col1, col2, col3 = st.columns([1.2, 1, 1.2])
            
            with col1:
                # Streamlit automaticky upraví zobrazení pro UI, ale data zůstávají detailní
                st.image(img_uint8, caption=f"Výřez (Karta {extracted_count})")
                
            with col2:
                st.write("**OCR Text:**")
                st.caption(text)
                if scryfall_name:
                    st.success(f"**Shoda:** {scryfall_name}")
                else:
                    st.error(scryfall_img) # Zobrazí chybovou hlášku, pokud se nenašlo
                    
            with col3:
                if scryfall_img and scryfall_img.startswith("http"):
                    st.image(scryfall_img, caption="Scryfall Databáze")
                else:
                    st.warning("Náhled není k dispozici")
                    
            st.divider()

    if extracted_count == 0:
        st.warning("Nenalezena žádná karta. Zkuste fotit z větší dálky nebo na kontrastním pozadí.")
