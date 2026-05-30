import streamlit as st
import numpy as np
import math
import cv2  # PŘIDÁNO: Import OpenCV pro perspective transform
from skimage import io
from skimage.feature import canny
from skimage.morphology import dilation, square
from scipy import ndimage
import easyocr
import requests
import urllib.parse

# 1. OPTIMALIZACE: Načítáme model s omezením paměti
@st.cache_resource
def load_reader():
    # gpu=False je na Streamlit Cloud jistota
    return easyocr.Reader(['en'], gpu=False)

# Pomocná funkce pro perspective transform (dokonalý výřez)
def rectify_image(image, pts_rect):
    # Určíme rozměry výřezu (800x600 je dobrý poměr pro karty nastojato)
    width, height = 600, 800
    
    # Cílové souřadnice pro perspective transform
    # Cílem je dostat kartu do rovného obdélníku
    # Pořadí bodů musí odpovídat pořadí detekovaných rohů
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]], dtype="float32")
    
    # Seřadíme body: horní-levý, horní-pravý, dolní-pravý, dolní-levý
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_rect.sum(axis=1)
    rect[0] = pts_rect[np.argmin(s)] # horní-levý (nejmenší suma)
    rect[2] = pts_rect[np.argmax(s)] # dolní-pravý (největší suma)
    diff = np.diff(pts_rect, axis=1)
    rect[1] = pts_rect[np.argmin(diff)] # horní-pravý (nejmenší rozdíl)
    rect[3] = pts_rect[np.argmax(diff)] # dolní-levý (největší rozdíl)

    # Výpočet matice transformace a aplikace
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

# Funkce pro hledání na Scryfallu
def fetch_scryfall_card(ocr_result):
    if not ocr_result:
        return None, None

    # Předpokládáme, že první rozumný řádek je jméno karty
    candidate_name = None
    for line in ocr_result:
        # Odfiltrujeme příliš krátké šumy (třeba jen číslice, jako jsi měl ty)
        if len(line.strip()) > 3:
            candidate_name = line.strip()
            break
            
    if not candidate_name:
        return None, None
    
    url = f"https://api.scryfall.com/cards/named?fuzzy={urllib.parse.quote(candidate_name)}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            image_url = data.get('image_uris', {}).get('normal')
            if not image_url and 'card_faces' in data:
                image_url = data['card_faces'][0].get('image_uris', {}).get('normal')
            return data.get('name'), image_url
        else:
            return None, None
    except Exception as e:
        return None, None

st.set_page_config(page_title="Dokonalá čtečka karet", layout="centered")
st.title("🎴 Dokonalá čtečka karet se Scryfallem")

# Příprava prázdného kontejneru pro seznam karet na začátku stránky
seznam_container = st.container()

# Načtení modelu hned na začátku
reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Načtení v nižším rozlišení pro úsporu RAM
    raw_img = io.imread(img_file, as_gray=True)
    
    # Snížení rozlišení pro stabilitu (stále důležité pro Streamlit Cloud)
    target_width = 1200
    scale = target_width / raw_img.shape[1]
    new_shape = (int(raw_img.shape[0] * scale), target_width)
    image = cv2.resize(raw_img, (target_width, int(raw_img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    
    st.info("Zpracovávám obrázek a hledám karty... prosím čekejte.")

    # Detekce karet (kontury z OpenCV jsou pro perspective transform lepší)
    gray = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    # Hledání kontur
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # Seřadit podle velikosti

    extracted_count = 0
    found_cards = []
    
    for c in cnts:
        # Přibližná kontura
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Pokud má kontura 4 body, považujeme to za kartu
        if len(approx) == 4 and cv2.contourArea(c) > 20000: # Limit pro velikost, aby to nechytalo smítka
            extracted_count += 1
            
            # --- ZÍSKÁNÍ DOKONALÉHO VÝŘEZU ---
            rectified_card = rectify_image(image, approx.reshape(4, 2))
            
            # Skew correction (finální resize na šířku 800px)
            card_img_res = cv2.resize(rectified_card, (int(rectified_card.shape[0]*(800/rectified_card.shape[1])), 800), interpolation=cv2.INTER_CUBIC)
            img_uint8 = (card_img_res * 255).astype(np.uint8)
            
            # --- BRUTE FORCE: ZKOUŠÍME VŠECHNY 4 ORIENTACE ---
            # Seznam otočených obrázků (0°, 90°, 180°, 270°)
            rotated_images = [img_uint8]
            rotated_captions = [card_img_res] # Pro zobrazení v aplikaci
            for k in range(1, 4):
                rotated_images.append(np.rot90(img_uint8, k=k))
                rotated_captions.append(np.rot90(card_img_res, k=k))

            best_scryfall_name = None
            best_scryfall_img = None
            best_ocr_text = "Nenalezen validní název"
            correct_img_uint8 = img_uint8 # Výchozí

            with st.spinner(f'Čtu kartu č. {extracted_count} ze všech stran...'):
                for idx, rotated_img in enumerate(rotated_images):
                    result = reader.readtext(rotated_img, detail=0)
                    if result:
                        scryfall_name, scryfall_img = fetch_scryfall_card(result)
                        
                        # Poku Scryfall něco vrátil, orientace je správná!
                        if scryfall_name and scryfall_img:
                            best_scryfall_name = scryfall_name
                            best_scryfall_img = scryfall_img
                            best_ocr_text = " ".join(result)
                            correct_img_uint8 = rotated_captions[idx]
                            
                            # Zastavíme se, našli jsme to!
                            break
                            
            # Uložení unikátního jména do seznamu
            if best_scryfall_name and best_scryfall_name not in found_cards:
                found_cards.append(best_scryfall_name)
            
            # Zobrazení výsledku ve 3 sloupcích
            col1, col2, col3 = st.columns([1.2, 1, 1.2])
            
            with col1:
                # Zobrazíme správně otočený náhled
                st.image(correct_img_uint8, caption=f"Dokonalý výřez (Karta {extracted_count})")
                
            with col2:
                st.write("**OCR Text (nejlepší pokus):**")
                st.caption(best_ocr_text)
                if best_scryfall_name:
                    st.success(f"**Shoda:** {best_scryfall_name}")
                else:
                    st.error("Karta nenalezena v databázi.")
                    
            with col3:
                if best_scryfall_img:
                    st.image(best_scryfall_img, caption="Scryfall Databáze")
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
        st.warning("Nenalezena žádná karta. Zkuste fotit na kontrastním pozadí.")
