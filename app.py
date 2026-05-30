import streamlit as st
import numpy as np
import math
import cv2
from skimage import io
import easyocr
import requests
import urllib.parse

# 1. OPTIMALIZACE: Načítáme model s omezením paměti
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

# VYLEPŠENÁ FUNKCE: Zachovává perfektní poměr stran bez deformace
def rectify_image(image, pts_rect):
    # Seřadíme 4 body tak, aby šly spolehlivě za sebou: tl, tr, br, bl
    x_sorted = pts_rect[np.argsort(pts_rect[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    tl = left_most[np.argmin(left_most[:, 1]), :]
    bl = left_most[np.argmax(left_most[:, 1]), :]
    tr = right_most[np.argmin(right_most[:, 1]), :]
    br = right_most[np.argmax(right_most[:, 1]), :]
    
    rect = np.array([tl, tr, br, bl], dtype="float32")
    
    # Výpočet reálných rozměrů stran v obrázku pomocí euklidovské vzdálenosti
    width_a = np.linalg.norm(tr - tl)
    width_b = np.linalg.norm(br - bl)
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.linalg.norm(bl - tl)
    height_b = np.linalg.norm(br - tr)
    max_height = max(int(height_a), int(height_b))
    
    # Poměr stran MTG karty je ideálně cca 1 : 1.39 (např. 600 x 834)
    # Dynamicky určíme rozměr podle toho, jak karta na fotce reálně leží
    if max_width > max_height:
        # Karta leží na boku -> cílový výřez bude schválně na šířku
        dst_w, dst_h = 834, 600
    else:
        # Karta stojí -> cílový výřez bude standardně na výšku
        dst_w, dst_h = 600, 834
        
    dst_pts = np.array([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

# Funkce pro hledání na Scryfallu
def fetch_scryfall_card(ocr_result):
    if not ocr_result:
        return None, None

    candidate_name = None
    for line in ocr_result:
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

seznam_container = st.container()
reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    raw_img = io.imread(img_file, as_gray=True)
    
    target_width = 1200
    image = cv2.resize(raw_img, (target_width, int(raw_img.shape[0] * (target_width / raw_img.shape[1]))), interpolation=cv2.INTER_AREA)
    
    st.info("Zpracovávám obrázek and hledám karty... prosím čekejte.")

    # Detekce hran
    gray = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 20, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    extracted_count = 0
    found_cards = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        
        if area > 8000:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="float32")
            
            extracted_count += 1
            
            # 1. KROK: Získáme narovnaný výřez, který ale NENÍ deformovaný
            rectified_card = rectify_image(image, box)
            rectified_card = np.clip(rectified_card, 0.0, 1.0)
            
            # 2. KROK: Pokud nám vylezl výřez naležato, otočíme ho o 90°, aby stál vertikálně
            if rectified_card.shape[1] > rectified_card.shape[0]:
                rectified_card = np.rot90(rectified_card, k=1)
            
            # Nyní máme garantovaný rozměr 600x834 bez deformace textu
            card_img_res = rectified_card
            img_uint8 = (card_img_res * 255).astype(np.uint8)
            
            # Příprava 4 rotací pro jistotu (0°, 90°, 180°, 270°)
            rotated_images = [img_uint8]
            rotated_captions = [card_img_res]
            for k in range(1, 4):
                rotated_images.append(np.rot90(img_uint8, k=k))
                rotated_captions.append(np.rot90(card_img_res, k=k))

            best_scryfall_name = None
            best_scryfall_img = None
            best_ocr_text = "Nenalezen validní název"
            correct_img_res = card_img_res

            with st.spinner(f'Čtu kartu č. {extracted_count} ze všech stran bez deformace...'):
                for idx, rotated_img in enumerate(rotated_images):
                    result = reader.readtext(rotated_img, detail=0)
                    if result:
                        scryfall_name, scryfall_img = fetch_scryfall_card(result)
                        
                        if scryfall_name and scryfall_img:
                            best_scryfall_name = scryfall_name
                            best_scryfall_img = scryfall_img
                            best_ocr_text = " ".join(result)
                            correct_img_res = rotated_captions[idx]
                            break
                            
            if best_scryfall_name and best_scryfall_name not in found_cards:
                found_cards.append(best_scryfall_name)
            
            col1, col2, col3 = st.columns([1.2, 1, 1.2])
            
            with col1:
                st.image(correct_img_res, caption=f"Krásný výřez (Karta {extracted_count})", clamp=True)
                
            with col2:
                st.write("**OCR Text:**")
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

    if found_cards:
        with seznam_container:
            st.subheader("📋 Seznam získaných karet:")
            for card in found_cards:
                st.markdown(f"**• {card}**")
            st.divider()

    if extracted_count == 0:
        st.warning("Nenalezena žádná karta. Zkuste fotit na kontrastnějším pozadí.")
