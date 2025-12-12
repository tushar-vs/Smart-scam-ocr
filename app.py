
import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="SmartScan AI", layout="wide", page_icon="🧾")

# We use @st.cache_resource to load the OCR model once and keep it in memory.
# This prevents the app from reloading the heavy model every time you click a button.
@st.cache_resource
def load_model():
    
    return easyocr.Reader(['en'], gpu=False) 

reader = load_model()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def preprocess_image(uploaded_file):
    """
    Reads the file uploaded by the user and converts it into a format
    that OpenCV and EasyOCR can read (NumPy array).
    """
    image = Image.open(uploaded_file)
    image = np.array(image) # Convert PIL image to NumPy array
    return image

def extract_financial_data(text_list):
    """
    Takes a list of raw strings found in the image and uses Regex (Regular Expressions)
    to guess which string is the Date and which is the Total Amount.
    """
    data = {
        "Vendor": "Unknown",
        "Date": None,
        "Total_Amount": None
    }
    
    # 1. Vendor Logic: The first line is usually the store name (e.g., "Walmart")
    if len(text_list) > 0:
        data["Vendor"] = text_list[0]

    # 2. Date Logic: Look for patterns like DD/MM/YYYY or YYYY-MM-DD
    # \d{2} means 2 digits, [/-] means a slash or dash separator
    date_pattern = r'(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})'
    
    # 3. Price Logic: Look for numbers formatted like currency (10.99, 1,000.00)
    # This regex looks for digits, optional commas, and a decimal point with 2 digits
    price_pattern = r'[$€£]?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2}))' 

    amounts = []

    for line in text_list:
        # --- Find Date ---
        if not data["Date"]:
            date_match = re.search(date_pattern, line)
            if date_match:
                data["Date"] = date_match.group(1)

        # --- Find Money ---
        # We prioritize lines that have keywords like "Total" or "Amount"
        clean_line = line.lower()
        if any(keyword in clean_line for keyword in ["total", "amount", "grand", "due", "balance"]):
             price_match = re.search(price_pattern, line)
             if price_match:
                 # Remove commas (1,000 -> 1000) and convert to float
                 value = float(price_match.group(1).replace(',', ''))
                 amounts.append(value)
        
        # Fallback: Capture all money-looking things just in case
        else:
             price_match = re.search(price_pattern, line)
             if price_match:
                 # We assume the largest number on the receipt might be the total
                 # if we don't find the word "Total"
                 pass 

    # If we found multiple amounts, usually the largest one is the Grand Total
    if amounts:
        data["Total_Amount"] = max(amounts)
    
    return data

# ==========================================
# 3. MAIN APP INTERFACE
# ==========================================

st.title("🧾 SmartScan: AI Invoice Parser")
st.markdown("### Upload a receipt or invoice to extract data automatically.")

# Create two columns for a clean layout
col1, col2 = st.columns(2)

# File Uploader Widget
uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # --- Step 1: Load Image ---
    image_np = preprocess_image(uploaded_file)
    
    with col1:
        st.subheader("Original Receipt")
        st.image(image_np, use_container_width=True)

    # --- Step 2: The "Magic" Button ---
    if st.button("Extract Data 🚀", type="primary"):
        with st.spinner("Scanning document... (This may take a moment)"):
            
            # A. Run EasyOCR
            # readtext returns a list of tuples: (bounding_box, text, confidence)
            results = reader.readtext(image_np) 
            
            # Extract just the text strings for our data analysis
            raw_text_list = [res[1] for res in results]
            
            # B. Draw Bounding Boxes (Visualization)
            # We copy the image so we don't draw over the original
            annotated_img = image_np.copy()
            for (bbox, text, prob) in results:
                # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                # Draw a green rectangle
                cv2.rectangle(annotated_img, top_left, bottom_right, (0, 255, 0), 2)

            with col2:
                st.subheader("AI Analysis")
                st.image(annotated_img, caption="Detected Text Zones", use_container_width=True)
                
            # C. Extract Structured Information
            structured_data = extract_financial_data(raw_text_list)
            
            # D. Show Results Table
            st.divider()
            st.subheader("📂 Extracted Results")
            
            # Convert dictionary to Pandas DataFrame for a nice table
            df = pd.DataFrame([structured_data])
            st.dataframe(df, use_container_width=True)
            
            # E. Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="invoice_data.csv",
                mime="text/csv"
            )
            
            # F. Debugging (Show raw text)
            with st.expander("See Raw Extracted Text"):
                st.write(raw_text_list)
