import streamlit as st
import numpy as np
import math
from skimage import io
from skimage.feature import canny
from skimage.transform import resize, rotate
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

# Příprava prázdného kontejneru pro seznam karet na začátku stránky
seznam_container = st.container()

# Načtení modelu hned na začátku
reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])

final_file = img_file

if final_file is not None:
    # Načtení v nižším rozlišení pro úsporu RAM
    raw_img = io.imread(final_file, as_gray=True)
    
    # Snížení rozlišení pro stabilitu na Streamlit Cloud
    target_width = 1200
    scale = target_width / raw_img.shape[1]
    new_shape = (int(raw_img.shape[0] * scale), target_width)
    image = resize(raw_img, new_shape, anti_aliasing=True)
    
    st.info("Zpracovávám obrázek a hledám karty... prosím čekejte.")

    # Detekce karet
    edges = canny(image, sigma=2.0)
    filled_cards = ndimage.binary_fill_holes(dilation(edges, square(3)))
    labeled_image = label(filled_cards)
    regions = regionprops(labeled_image)

    extracted_count = 0
    found_cards = []
    
    for region in regions:
        if region.area > 2000: # Limit pro menší rozlišení
            extracted_count += 1
            min_row, min_col, max_row, max_col = region.bbox
            card_crop = image[min_row:max_row, min_col:max_col]

            # --- OPRÁVENÉ: Automatické narovnání náklonu (Deskewing) ---
            angle = np.rad2deg(region.orientation)
            
            # Normalizace úhlu: Chceme srovnat jen jemný náklon.
            # Pokud skimage detekuje úhel větší než 45°, přepočítáme ho, aby karta neuskakovala o 90°.
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            
            # OPRAVA SMĚRU: Používáme mínus úhel (-angle), abychom rotaci kompenzovali zpět do nuly
            card_crop = rotate(card_crop, -angle, resize=True, mode='edge')

            # --- POHYBOVÁNO SEM: Kontrola orientace až PO narovnání drobného náklonu ---
            if card_crop.shape[1] > card_crop.shape[0]:
                card_crop = np.rot90(card_crop, k=1)

            # Skew correction a finální resize na šířku 800px pro OCR
            card_img_res = resize(card_crop, (int(card_crop.shape[0]*(800/card_crop.shape[1])), 800))
            
            # Převod na uint8 pro EasyOCR
            img_uint8 = (card_img_res * 255).astype(np.uint8)
            
            # Spuštění OCR a vyhledávání
            with st.spinner(f'Čtu kartu č. {extracted_count} a hledám na Scryfallu...'):
                result = reader.readtext(img_uint8, detail=0)
                scryfall_name, scryfall_img = fetch_scryfall_card(result)
                
                # Pokud Scryfall nic nenašel, karta je možná vzhůru nohama (otočená o 180°)
                if not scryfall_name and result:
                    img_uint8 = np.rot90(img_uint8, k=2)
                    card_img_res = np.rot90(card_img_res, k=2)
                    
                    result = reader.readtext(img_uint8, detail=0)
                    scryfall_name, scryfall_img = fetch_scryfall_card(result)
                
                # Uložení unikátního jména karty
                if scryfall_name and scryfall_name not in found_cards:
                    found_cards.append(scryfall_name)
                
                text = " ".join(result) if result else "Text nenalezen"
            
            # Zobrazení výsledku ve 3 sloupcích
            col1, col2, col3 = st.columns([1.2, 1, 1.2])
            
            with col1:
                st.image(card_img_res, caption=f"Výřez (Karta {extracted_count})")
                
            with col2:
                st.write("**OCR Text:**")
                st.caption(text)
                if scryfall_name:
                    st.success(f"**Shoda:** {scryfall_name}")
                else:
                    st.error(scryfall_img)
                    
            with col3:
                if scryfall_img and scryfall_img.startswith("http"):
                    st.image(scryfall_img, caption="Scryfall Databáze")
                else:
                    st.warning("Náhled není k dispozici")
                    
            st.divider()

    # Zpětné vykreslení seznamu karet na úplný začátek stránky
    if found_cards:
        with seznam_container:
            st.subheader("📋 Seznam získaných karet:")
            for card in found_cards:
                st.markdown(f"**• {card}**")
            st.divider()

    if extracted_count == 0:
        st.warning("Nenalezena žádná karta. Zkuste fotit z větší dálky nebo na kontrastním pozadí.")
