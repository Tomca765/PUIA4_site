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

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

def fetch_scryfall_card(ocr_result):
    if not ocr_result:
        return None, "Žádný text k vyhledání."

    candidate_name = ocr_result[0]
    url = f"https://api.scryfall.com/cards/named?fuzzy={urllib.parse.quote(candidate_name)}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # QUALITY UPGRADE 1: Změněno z 'normal' na 'large' pro maximální ostrost ze Scryfallu
            image_url = data.get('image_uris', {}).get('large')
            if not image_url and 'card_faces' in data:
                image_url = data['card_faces'][0].get('image_uris', {}).get('large')
                
            return data.get('name'), image_url
        else:
            return None, f"Karta nenalezena pro dotaz: '{candidate_name}'"
    except Exception as e:
        return None, f"Chyba API: {e}"

st.set_page_config(page_title="OCR Karet", layout="centered")
st.title("🎴 Úsporná čtečka karet se Scryfallem")

reader = load_reader()

img_file = st.sidebar.file_uploader("Nahraj fotku nebo vyfoť", type=['jpg', 'jpeg', 'png'])
camera_file = st.sidebar.camera_input("Nebo použij kameru")

final_file = camera_file if camera_file else img_file

if final_file is not None:
    # 1. Načteme originál v plné kvalitě
    raw_img = io.imread(final_file, as_gray=True)
    
    # 2. Zmenšíme ho JEN pro detekci hran (šetří RAM na Streamlit serveru)
    target_width = 1200
    scale = target_width / raw_img.shape[1]
    new_shape = (int(raw_img.shape[0] * scale), target_width)
    image = resize(raw_img, new_shape, anti_aliasing=True)
    
    st.info("Zpracovávám obrázek a hledám karty... prosím čekejte.")

    # Detekce objektů (stále běží na rychlém, malém obrázku)
    edges = canny(image, sigma=2.0)
    filled_cards = ndimage.binary_fill_holes(dilation(edges, square(3)))
    labeled_image = label(filled_cards)
    regions = regionprops(labeled_image)

    extracted_count = 0
    
    for region in regions:
        if region.area > 2000: 
            extracted_count += 1
            min_row, min_col, max_row, max_col = region.bbox

            # QUALITY UPGRADE 2: Přepočítáme souřadnice výřezu zpět na původní originální fotku
            orig_min_row = int(min_row / scale)
            orig_max_row = int(max_row / scale)
            orig_min_col = int(min_col / scale)
            orig_max_col = int(max_col / scale)
            
            # Výřez bereme z originálu (raw_img), ne ze zmenšeniny (image)
            card_crop = raw_img[orig_min_row:orig_max_row, orig_min_col:orig_max_col]

            # Zmenšíme čistě z vysokého rozlišení dolů na 800px (dolů = ostré, nahoru = rozmazané)
            card_img_res = resize(card_crop, (int(card_crop.shape[0]*(800/card_crop.shape[1])), 800), anti_aliasing=True)
            
            img_uint8 = (card_img_res * 255).astype(np.uint8)
            
            with st.spinner(f'Čtu kartu č. {extracted_count} a hledám na Scryfallu...'):
                result = reader.readtext(img_uint8, detail=0)
                text = " ".join(result) if result else "Text nenalezen"
                scryfall_name, scryfall_img = fetch_scryfall_card(result)
            
            col1, col2, col3 = st.columns([1.2, 1, 1.2])
            
            with col1:
                st.image(card_img_res, caption=f"Ostrý Výřez (Karta {extracted_count})")
                
            with col2:
                st.write("**OCR Text:**")
                st.caption(text)
                if scryfall_name:
                    st.success(f"**Shoda:** {scryfall_name}")
                else:
                    st.error(scryfall_img)
                    
            with col3:
                if scryfall_img and scryfall_img.startswith("http"):
                    st.image(scryfall_img, caption="Scryfall (Vysoká kvalita)")
                else:
                    st.warning("Náhled není k dispozici")
                    
            st.divider()

    if extracted_count == 0:
        st.warning("Nenalezena žádná karta. Zkuste fotit z větší dálky nebo na kontrastním pozadí.")
