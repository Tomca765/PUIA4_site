import streamlit as st
import numpy as np
import math
import cv2
from skimage import io
import easyocr
import requests
import urllib.parse
import difflib
import gc  # PŘIDÁNO: Garbage Collector pro okamžité uvolňování RAM

# 1. OPTIMALIZACE: Načítáme model s omezením paměti
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

# Pomocná funkce pro výpočet textové shody
def get_similarity(str1, str2):
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

# Funkce pro perspective transform (dokonalý výřez)
def rectify_image(image, pts_rect):
    x_sorted = pts_rect[np.argsort(pts_rect[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    tl = left_most[np.argmin(left_most[:, 1]), :]
    bl = left_most[np.argmax(left_most[:, 1]), :]
    tr = right_most[np.argmin(right_most[:, 1]), :]
    br = right_most[np.argmax(right_most[:, 1]), :]
    
    rect = np.array([tl, tr, br, bl], dtype="float32")
    
    width_a = np.linalg.norm(tr - tl)
    width_b = np.linalg.norm(br - bl)
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.linalg.norm(bl - tl)
    height_b = np.linalg.norm(br - tr)
    max_height = max(int(height_a), int(height_b))
    
    if max_width > max_height:
        dst_w, dst_h = 834, 600
    else:
        dst_w, dst_h = 600, 834
        
    dst_pts = np.array([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

# API pro Scryfall
def query_scryfall(candidate_name):
    if not candidate_name or len(candidate_name.strip()) <= 3:
        return None, None
        
    url = f"https://api.scryfall.com/cards/named?fuzzy={urllib.parse.quote(candidate_name.strip())}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            image_url = data.get('image_uris', {}).get('normal')
            if not image_url and 'card_faces' in data:
                image_url = data['card_faces'][0].get('image_uris', {}).get('normal')
            return data.get('name'), image_url
    except:
        pass
    return None, None

st.set_page_config(page_title="Dokonalá čtečka karet", layout="centered")
st.title("🎴 Dokonalá čtečka karet se Scryfallem")

# --- INICIALIZACE TRVALÉ PAMĚTI ---
if "master_card_list" not in st.session_state:
    st.session_state.master_card_list = []

# Tlačítko pro vymazání seznamu v sidebaru
if st.sidebar.button("🗑️ Vymazat celou paměť karet"):
    st.session_state.master_card_list = []
    st.rerun()

# Příprava prázdného kontejneru pro seznam karet na úplném začátku stránky
seznam_container = st.container()

# Vykreslení aktuálního stavu permanentního seznamu navrchu
if st.session_state.master_card_list:
    with seznam_container:
        st.subheader(f"📋 Celkový seznam naskenovaných karet ({len(st.session_state.master_card_list)}):")
        for card in st.session_state.master_card_list:
            st.markdown(f"**• {card}**")
        st.divider()

reader = load_reader()

# ZMĚNA: Přidán parametr accept_multiple_files=True
img_files = st.sidebar.file_uploader("Nahraj fotky nebo vyfoť (i více najednou)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if img_files:
    # Procházíme nahrané soubory jeden po druhém
    for file_idx, img_file in enumerate(img_files):
        
        # Každou fotku zabalíme do přehledného expanderu
        with st.expander(f"📸 Zpracování souboru: {img_file.name}", expanded=True):
            raw_img = io.imread(img_file, as_gray=True)
            
            target_width = 1200
            image = cv2.resize(raw_img, (target_width, int(raw_img.shape[0] * (target_width / raw_img.shape[1]))), interpolation=cv2.INTER_AREA)
            
            # Detekce hran
            gray = (image * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 20, 100)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edged = cv2.dilate(edged, kernel, iterations=1)
            
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            extracted_count = 0
            
            for c in cnts:
                area = cv2.contourArea(c)
                
                if area > 8000:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.array(box, dtype="float32")
                    
                    extracted_count += 1
                    
                    rectified_card = rectify_image(image, box)
                    rectified_card = np.clip(rectified_card, 0.0, 1.0)
                    
                    if rectified_card.shape[1] > rectified_card.shape[0]:
                        rectified_card = np.rot90(rectified_card, k=1)
                    
                    card_img_res = rectified_card
                    img_uint8 = (card_img_res * 255).astype(np.uint8)
                    
                    # 4 rotace
                    rotated_images = [img_uint8]
                    rotated_captions = [card_img_res]
                    for k in range(1, 4):
                        rotated_images.append(np.rot90(img_uint8, k=k))
                        rotated_captions.append(np.rot90(card_img_res, k=k))

                    best_score = -1.0
                    best_scryfall_name = None
                    best_scryfall_img = None
                    best_ocr_text = "Nenalezen validní název"
                    correct_img_res = card_img_res

                    with st.spinner(f'Čtu kartu č. {extracted_count} ze všech stran...'):
                        for idx, rotated_img in enumerate(rotated_images):
                            result = reader.readtext(rotated_img, detail=0)
                            if result:
                                for candidate in result[:3]:
                                    if len(candidate.strip()) > 3:
                                        scryfall_name, scryfall_img = query_scryfall(candidate)
                                        
                                        if scryfall_name:
                                            score = get_similarity(candidate, scryfall_name)
                                            if score > best_score:
                                                best_score = score
                                                best_scryfall_name = scryfall_name
                                                best_scryfall_img = scryfall_img
                                                best_ocr_text = " | ".join(result)
                                                correct_img_res = rotated_captions[idx]

                    if best_score < 0.2:
                        best_scryfall_name = None

                    # UKLÁDÁNÍ DO TRVALÉ PAMĚTI: Pokud kartu známe a ještě v ní není, šoupneme ji tam
                    if best_scryfall_name and best_scryfall_name not in st.session_state.master_card_list:
                        st.session_state.master_card_list.append(best_scryfall_name)
                    
                    # Výpis detailů na obrazovku pod expander
                    col1, col2, col3 = st.columns([1.2, 1, 1.2])
                    
                    with col1:
                        st.image(correct_img_res, caption=f"Výřez (Karta {extracted_count})", clamp=True)
                        
                    with col2:
                        st.write("**OCR Text:**")
                        st.caption(best_ocr_text)
                        if best_scryfall_name:
                            st.success(f"**Shoda ({int(best_score*100)}%):** {best_scryfall_name}")
                        else:
                            st.error("Karta nenalezena.")
                            
                    with col3:
                        if best_scryfall_img:
                            st.image(best_scryfall_img, caption="Scryfall")
                        else:
                            st.warning("Žádný náhled")
                            
                    st.divider()

            if extracted_count == 0:
                st.warning("V tomto souboru nebyla rozpoznána žádná karta.")

        # --- KLÍČOVÉ ČIŠTĚNÍ PAMĚTI ---
        # Po dokončení práce s daným souborem smažeme obrovská obrazová pole z RAM
        del raw_img, image, gray, blurred, edged, cnts
        gc.collect()  # Vynutíme okamžité vyčištění paměti Pythonem
        
    # Na konci celého cyklu bleskově překreslíme horní kontejner, aby se okamžitě aktualizovaly nově přidané karty
    st.rerun()
