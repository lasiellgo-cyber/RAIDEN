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

UMBRAL_ALTO = 0.35    
UMBRAL_MEDIO = 0.12   

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
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    path = MODEL_LOCAL if os.path.exists(MODEL_LOCAL) else (MODEL_CACHE if os.path.exists(MODEL_CACHE) else None)
    if not path:
        try:
            urllib.request.urlretrieve(HF_URL, MODEL_CACHE)
            path = MODEL_CACHE
        except:
            pass
    if path:
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            modelo.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            st.error(f"Error al cargar pesos: {e}")
    modelo.eval()
    return modelo

# --- INTERFAZ ---
st.title("🩻 Proyecto RAIDEN (Diagnóstico Médico)")
modelo = cargar_modelo()

archivo = st.file_uploader("Subir Radiografía de Tórax", type=["jpg","png","jpeg"])

if archivo:
    col1, col2 = st.columns([1, 1.2])
    img_pil = Image.open(archivo).convert("L")
    with col1:
        st.image(img_pil, use_container_width=True, caption="Imagen para análisis")
    with col2:
        img_arr = np.array(img_pil)
        img_arr = xrv.datasets.normalize(img_arr, 255)
        t = torch.from_numpy(img_arr[None, None, :, :]).float()
        t = torch.nn.functional.interpolate(t, size=(224, 224))
        # Normalización sincronizada con el entrenamiento
        t = (t * 2048.0) - 1024.0
        
        with torch.no_grad():
            preds = torch.sigmoid(modelo(t)).numpy()[0][:14]
        
        positivos = sorted([(CATEGORIAS[i], preds[i]) for i in range(14) if preds[i] >= UMBRAL_ALTO], key=lambda x: -x[1])
        sugestivos = sorted([(CATEGORIAS[i], preds[i]) for i in range(14) if UMBRAL_MEDIO <= preds[i] < UMBRAL_ALTO], key=lambda x: -x[1])
        
        if positivos:
            st.markdown(f'<div class="anormal"><h3>⚠ ESTADO: ANORMAL</h3>Se han detectado {len(positivos)} hallazgos.</div>', unsafe_allow_html=True)
        elif sugestivos:
            st.markdown(f'<div class="sugestivo"><h3>🔍 HALLAZGOS SUGESTIVOS</h3>Existen sospechas que requieren revisión clínica.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal"><h3>✓ EXPLORACIÓN NORMAL</h3>No se observan hallazgos significativos.</div>', unsafe_allow_html=True)
            
        todos = positivos + sugestivos
        if todos:
            st.subheader("Informe del Asistente")
            for cat, prob in todos:
                color = "#ff3d5a" if prob >= UMBRAL_ALTO else "#ffc400"
                st.markdown(f'<div class="seccion"><b style="color:{color}">{cat.upper()} ({prob*100:.1f}%)</b><br><small><b>Zona:</b> {ZONA[cat]}</small><br><i>{HALLAZGO[cat]}</i></div>', unsafe_allow_html=True)