import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

# --- CONFIGURACIÓN Y RUTAS ---
DEVICE = "cpu"
# Asegúrate de que esta URL sea la correcta de tu Hugging Face
HF_URL = "https://huggingface.co/LASIELL/RAIDEN/resolve/main/raiden_modelo.pth?download=true"
MODEL_CACHE = "/tmp/raiden_modelo.pth"
MODEL_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos_entrenados", "raiden_modelo.pth")

CATEGORIAS = ["Atelectasia","Cardiomegalia","Efusión","Infiltración","Masa","Nódulo","Neumonía","Neumotórax","Consolidación","Edema","Enfisema","Fibrosis","Eng.Pleural","Hernia"]
ZONA = {"Atelectasia":"Pulmón lóbulo inferior","Cardiomegalia":"Corazón / Mediastino","Efusión":"Espacio pleural","Infiltración":"Parénquima pulmonar","Masa":"Pulmón / Mediastino","Nódulo":"Parénquima pulmonar","Neumonía":"Pulmón consolidación","Neumotórax":"Espacio pleural","Consolidación":"Parénquima","Edema":"Pulmón bilateral","Enfisema":"Pulmón hiperinsuflación","Fibrosis":"Intersticio","Eng.Pleural":"Pleura","Hernia":"Diafragma"}
HALLAZGO = {"Atelectasia":"Opacidad laminar con pérdida de volumen","Cardiomegalia":"Silueta cardíaca aumentada","Efusión":"Opacificación del seno costofrénico","Infiltración":"Opacidades heterogéneas","Masa":"Opacidad redondeada > 3 cm","Nódulo":"Opacidad redondeada < 3 cm","Neumonía":"Consolidación con broncograma aéreo","Neumotórax":"Línea pleural visible sin trama vascular","Consolidación":"Opacidad homogénea","Edema":"Opacidades bilaterales perihiliares","Enfisema":"Hiperinsuflación, aplanamiento diafragmático","Fibrosis":"Opacidades reticulares basales","Eng.Pleural":"Opacidad pleural periférica","Hernia":"Estructura abdominal sobre el diafragma"}

# --- UMBRALES MÉDICOS CORREGIDOS (Más sensibilidad) ---
UMBRAL_ALTO = 0.35    # Rojo si supera 35%
UMBRAL_MEDIO = 0.12   # Naranja si supera 12% (Para no perder sospechas)

st.set_page_config(page_title="RUBÉN — Diagnóstico RX", page_icon="🩻", layout="wide")

# --- ESTILOS VISUALES ---
st.markdown("""
<style>
:root{--bg:#05080f;--card:#0f1726;--cyan:#00d4ff;--green:#00e676;--red:#ff3d5a;--yellow:#ffc400;--text:#c8d8f0;}
.stApp{background:var(--bg) !important;color:var(--text) !important;}
.normal{background:rgba(0,230,118,0.1);border:2px solid var(--green);border-radius:12px;padding:20px;margin-bottom:20px;}
.anormal{background:rgba(255,61,90,0.1);border:2px solid var(--red);border-radius:12px;padding:20px;margin-bottom:20px;}
.sugestivo{background:rgba(255,196,0,0.1);border:2px solid var(--yellow);border-radius:12px;padding:20px;margin-bottom:20px;}
.seccion{background:var(--card);border-radius:10px;padding:15px;margin:10px 0;border:1px solid #1a2540;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    # Estructura base de TorchXRayVision
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # Lógica de búsqueda del archivo
    path = MODEL_LOCAL if os.path.exists(MODEL_LOCAL) else (MODEL_CACHE if os.path.exists(MODEL_CACHE) else None)
    
    if not path:
        try:
            urllib.request.urlretrieve(HF_URL, MODEL_CACHE)
            path = MODEL_CACHE
        except: pass

    if path:
        try:
            # Cargamos los pesos (state_dict)
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            
            # --- ARREGLO DE LLAVES (Evita el RuntimeError) ---
            # Quitamos prefijos como "model." que a veces añade el entrenamiento
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("model.", "") 
                new_state_dict[name] = v
            
            modelo.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            st.error(f"Error técnico al cargar pesos: {e}")
            
    modelo.eval()
    return modelo

# --- INTERFAZ DE USUARIO ---
st.title("🩻 Proyecto RAIDEN (Diagnóstico Médico)")
modelo = cargar_modelo()

archivo = st.file_uploader("Subir Radiografía de Tórax", type=["jpg","png","jpeg"])

if archivo