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
HF_URL = "https://huggingface.co/LASIELL/RAIDEN/resolve/main/raiden_modelo.pth?download=true"
MODEL_CACHE = "/tmp/raiden_modelo.pth"
MODEL_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos_entrenados", "raiden_modelo.pth")

CATEGORIAS = ["Atelectasia","Cardiomegalia","Efusión","Infiltración","Masa","Nódulo","Neumonía","Neumotórax","Consolidación","Edema","Enfisema","Fibrosis","Eng.Pleural","Hernia"]
ZONA = {"Atelectasia":"Pulmón lóbulo inferior","Cardiomegalia":"Corazón / Mediastino","Efusión":"Espacio pleural","Infiltración":"Parénquima pulmonar","Masa":"Pulmón / Mediastino","Nódulo":"Parénquima pulmonar","Neumonía":"Pulmón consolidación","Neumotórax":"Espacio pleural","Consolidación":"Parénquima","Edema":"Pulmón bilateral","Enfisema":"Pulmón hiperinsuflación","Fibrosis":"Intersticio","Eng.Pleural":"Pleura","Hernia":"Diafragma"}
HALLAZGO = {"Atelectasia":"Opacidad laminar con pérdida de volumen","Cardiomegalia":"Silueta cardíaca aumentada","Efusión":"Opacificación del seno costofrénico","Infiltración":"Opacidades heterogéneas","Masa":"Opacidad redondeada > 3 cm","Nódulo":"Opacidad redondeada < 3 cm","Neumonía":"Consolidación con broncograma aéreo","Neumotórax":"Línea pleural visible sin trama vascular","Consolidación":"Opacidad homogénea","Edema":"Opacidades bilaterales perihiliares","Enfisema":"Hiperinsuflación, aplanamiento diafragmático","Fibrosis":"Opacidades reticulares basales","Eng.Pleural":"Opacidad pleural periférica","Hernia":"Estructura abdominal sobre el diafragma"}

# --- UMBRALES MÉDICOS DE SEGURIDAD ---
UMBRAL_ALTO = 0.35    
UMBRAL_MEDIO = 0.12   

st.set_page_config(page_title="RUBÉN — Diagnóstico RX", page_icon="🩻", layout="wide")

# --- ESTILOS VISUALES ---
st.markdown("""
<style>
:root{--bg:#05080f;--card:#0f1726;--cyan:#00d4ff;--green:#00e676;--red:#ff3d5a;--yellow:#ffc400;--text:#c8d8f0;}
.stApp{background:var(--bg) !important;color:var(--text) !important;}
.normal{background:rgba(0,230,118,0.1);border:2px solid var(--green);border-radius:12px;padding:20px;margin-bottom:2