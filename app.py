## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab 1 -> Week 1 - Streamlit Application (converted from Gradio)

import streamlit as st
from mosaic_pipeline import create_mosaic
from PIL import Image
import numpy as np

st.title("Interactive Image Mosaic Generator")
st.write("Upload an image, choose a grid size, and reconstruct it with predefined image tiles.")

# ---- INPUT IMAGE ----
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---- GRID SIZE ----
grid_size = st.selectbox("Grid size", [16, 32, 64], index=1)

# ---- TILE SET ----
tile_set = st.selectbox("Tile set", ["Solid", "Diagonal", "Checkerboard"], index=1)

# ---- RUN PIPELINE ----
if uploaded:
    # Convert uploaded file to numpy array
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    # Run your existing pipeline
    preprocessed, segmented, mosaic = create_mosaic(img_np, grid_size, tile_set)

    # Display results
    st.subheader("Preprocessed image")
    st.image(preprocessed)

    st.subheader("Segmented image")
    st.image(segmented)

    st.subheader("Mosaic reconstruction")
    st.image(mosaic)
