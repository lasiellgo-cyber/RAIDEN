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
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

CATEGORIAS = ["Atelectasia","Cardiomegalia","Efusión","Infiltración","Masa","Nódulo","Neumonía","Neumotórax","Consolidación","Edema","Enfisema","Fibrosis","Eng.Pleural","Hernia"]
ZONA = {"Atelectasia":"Pulmón lóbulo inferior","Cardiomegalia":"Corazón / Mediastino","Efusión":"Espacio pleural","Infiltración":"Parénquima pulmonar","Masa":"Pulmón / Mediastino","Nódulo":"Parénquima pulmonar","Neumonía":"Pulmón consolidación","Neumotórax":"Espacio pleural","Consolidación":"Parénquima","Edema":"Pulmón bilateral","Enfisema":"Pulmón hiperinsuflación","Fibrosis":"Intersticio","Eng.Pleural":"Pleura","Hernia":"Diafragma"}
HALLAZGO = {"Atelectasia":"Opacidad laminar con pérdida de volumen","Cardiomegalia":"Silueta cardíaca aumentada","Efusión":"Opacificación del seno costofrénico","Infiltración":"Opacidades heterogéneas","Masa":"Opacidad redondeada > 3 cm","Nódulo":"Opacidad redondeada < 3 cm","Neumonía":"Consolidación con broncograma aéreo","Neumotórax":"Línea pleural visible sin trama vascular","Consolidación":"Opacidad homogénea","Edema":"Opacidades bilaterales perihiliares","Enfisema":"Hiperinsuflación, aplanamiento diafragmático","Fibrosis":"Opacidades reticulares basales","Eng.Pleural":"Opacidad pleural periférica","Hernia":"Estructura abdominal sobre el diafragma"}

UMBRAL_ALTO = 0.45
UMBRAL_MEDIO = 0.20 # Bajamos un poco para no perder sospechas

st.set_page_config(page_title="RUBÉN — Diagnóstico RX", page_icon="🩻", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
:root{--bg:#05080f;--card:#0f1726;--cyan:#00d4ff;--green:#00e676;--red:#ff3d5a;--yellow:#ffc400;--text:#c8d8f0;}
.stApp{background:var(--bg) !important;color:var(--text) !important;}
.normal{background:rgba(0,230,118,0.1);border:2px solid var(--green);border-radius:12px;padding:20px;margin-bottom:20px;}
.anormal{background:rgba(255,61,90,0.1);border:2px solid var(--red);border-radius:12px;padding:20px;margin-bottom:20px;}
.sugestivo{background:rgba(255,196,0,0.1);border:2px solid var(--yellow);border-radius:12px;padding:20px;margin-bottom:20px;}
.seccion{background:var(--card);border-radius:10px;padding:15px;margin:10px 0;border:1px solid #1a2540;}
.hallazgo-item{border-left:3px solid var(--cyan);padding-left:15px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    # Intentar cargar local, luego cache, luego descargar
    path = MODEL_LOCAL if os.path.exists(MODEL_LOCAL) else (MODEL_CACHE if os.path.exists(MODEL_CACHE) else None)
    if not path:
        try: urllib.request.urlretrieve(HF_URL, MODEL_CACHE); path = MODEL_CACHE
        except: pass
    if path:
        state_dict = torch.load(path, map_location="cpu")
        modelo.load_state_dict(state_dict, strict=False)
    modelo.eval()
    return modelo

# --- INTERFAZ ---
st.title("🩻 Proyecto RAIDEN")
modelo = cargar_modelo()

archivo = st.file_uploader("Subir Radiografía", type=["jpg","png","jpeg"])

if archivo:
    col1, col2 = st.columns([1, 1.2])
    img_pil = Image.open(archivo).convert("L")
    
    with col1:
        st.image(img_pil, use_container_width=True, caption="Imagen cargada")

    with col2:
        # Preprocesamiento idéntico al entrenamiento
        img_arr = np.array(img_pil)
        img_arr = xrv.datasets.normalize(img_arr, 255)
        t = torch.from_numpy(img_arr[None, None, :, :]).float()
        t = torch.nn.functional.interpolate(t, size=(224, 224))
        # Normalización a rango TXRV
        t = t * 2048.0 - 1024.0
        
        with torch.no_grad():
            preds = torch.sigmoid(modelo(t)).numpy()[0][:14]
        
        # Clasificación por umbrales
        positivos = sorted([(CATEGORIAS[i], preds[i]) for i in range(14) if preds[i] >= UMBRAL_ALTO], key=lambda x: -x[1])
        sugestivos = sorted([(CATEGORIAS[i], preds[i]) for i in range(14) if UMBRAL_MEDIO <= preds[i] < UMBRAL_ALTO], key=lambda x: -x[1])

        # --- LÓGICA DE SEMÁFORO CORREGIDA ---
        if positivos:
            st.markdown(f'<div class="anormal"><h3>⚠ ANORMAL</h3>Se han detectado {len(positivos)} hallazgos confirmados.</div>', unsafe_allow_html=True)
        elif sugestivos:
            st.markdown(f'<div class="sugestivo"><h3>🔍 HALLAZGOS SUGESTIVOS</h3>Existen sospechas leves que requieren correlación clínica.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal"><h3>✓ NORMAL</h3>No se detectan hallazgos patológicos significativos.</div>', unsafe_allow_html=True)

        # Mostrar hallazgos
        todos = positivos + sugestivos
        if todos:
            st.subheader("Informe Detallado")
            for cat, prob in todos:
                color = "red" if prob >= UMBRAL_ALTO else "orange"
                st.markdown(f'''
                <div class="seccion">
                    <b style="color:{color}">{cat} ({prob*100:.1f}%)</b><br>
                    <small><b>Zona:</b> {ZONA[cat]}</small><br>
                    <i>{HALLAZGO[cat]}</i>
                </div>
                ''', unsafe_allow_html=True)