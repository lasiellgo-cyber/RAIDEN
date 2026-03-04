import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request
from PIL import Image
import torchxrayvision as xrv

DEVICE = "cpu"
HF_URL = "https://huggingface.co/LASIELL/rx-modelo/resolve/main/raiden_modelo.pth?download=true"
MODEL_CACHE = "/tmp/raiden_modelo.pth"
MODEL_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos_entrenados", "raiden_modelo.pth")
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

CATEGORIAS = ["Atelectasia","Cardiomegalia","Efusión","Infiltración","Masa","Nódulo","Neumonía","Neumotórax","Consolidación","Edema","Enfisema","Fibrosis","Eng.Pleural","Hernia"]

ZONA = {"Atelectasia":"Pulmón lóbulo inferior","Cardiomegalia":"Corazón / Mediastino","Efusión":"Espacio pleural","Infiltración":"Parénquima pulmonar","Masa":"Pulmón / Mediastino","Nódulo":"Parénquima pulmonar","Neumonía":"Pulmón consolidación alveolar","Neumotórax":"Espacio pleural / Apex","Consolidación":"Parénquima pulmonar","Edema":"Pulmón bilateral / Hilio","Enfisema":"Pulmón hiperinsuflación","Fibrosis":"Intersticio pulmonar","Eng.Pleural":"Pleura","Hernia":"Diafragma / Mediastino inferior"}

HALLAZGO = {"Atelectasia":"Opacidad laminar con pérdida de volumen y desplazamiento de fisuras","Cardiomegalia":"Índice cardiotorácico > 0.5, silueta cardíaca aumentada","Efusión":"Opacificación del seno costofrénico con menisco pleural","Infiltración":"Opacidades heterogéneas de predominio peribronquial","Masa":"Opacidad redondeada > 3 cm con bordes bien definidos","Nódulo":"Opacidad redondeada < 3 cm, bordes pueden ser espiculados","Neumonía":"Consolidación alveolar con broncograma aéreo positivo","Neumotórax":"Línea pleural visible, ausencia de trama vascular periférica","Consolidación":"Opacidad homogénea con broncograma aéreo","Edema":"Opacidades bilaterales perihiliares en alas de mariposa","Enfisema":"Hiperinsuflación, aplanamiento diafragmático","Fibrosis":"Opacidades reticulares basales bilaterales","Eng.Pleural":"Opacidad pleural periférica sin menisco","Hernia":"Estructura abdominal por encima del diafragma"}

UMBRAL_ALTO = 0.45
UMBRAL_MEDIO = 0.25

st.set_page_config(page_title="RUBÉN — Diagnóstico RX", page_icon="🩻", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#05080f;--card:#0f1726;--border:#1a2540;--cyan:#00d4ff;--green:#00e676;--red:#ff3d5a;--yellow:#ffc400;--text:#c8d8f0;--muted:#4a6080;}
*{font-family:'IBM Plex Sans',sans-serif !important;}
.stApp{background:var(--bg) !important;color:var(--text) !important;}
.block-container{padding:1.5rem 2rem 3rem !important;max-width:1300px !important;}
#MainMenu,footer,header{visibility:hidden !important;}
.normal{background:linear-gradient(135deg,#001a0a,#002210);border:2px solid var(--green);border-radius:12px;padding:24px 28px;margin:12px 0;}
.anormal{background:linear-gradient(135deg,#1a0008,#220010);border:2px solid var(--red);border-radius:12px;padding:24px 28px;margin:12px 0;}
.seccion{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin:10px 0;}
.hallazgo{border-left:3px solid var(--yellow);padding:8px 14px;margin:8px 0;background:#0d1220;border-radius:0 6px 6px 0;}
.diagnostico{background:linear-gradient(135deg,#0a0f20,#0f1530);border:1px solid var(--cyan);border-radius:10px;padding:20px 24px;margin:10px 0;}
.zona-tag{display:inline-block;background:#0a1530;border:1px solid var(--border);color:var(--cyan);font-size:0.75rem;padding:4px 12px;border-radius:4px;margin:3px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    modelo = xrv.models.DenseNet(weights="densenet121-res224-all")
    modelo.op_threshs = None
    modelo.classifier = nn.Linear(modelo.classifier.in_features, 14)
    tipo = "BASE"
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) > 1000000:
        path = MODEL_LOCAL; tipo = "ENTRENADO LOCAL"
    elif os.path.exists(MODEL_CACHE) and os.path.getsize(MODEL_CACHE) > 1000000:
        path = MODEL_CACHE; tipo = "ENTRENADO HF"
    else:
        path = None
        try:
            urllib.request.urlretrieve(HF_URL, MODEL_CACHE)
            if os.path.getsize(MODEL_CACHE) > 1000000:
                path = MODEL_CACHE; tipo = "ENTRENADO HF"
        except: pass
    if path:
        try: modelo.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
        except:
            try: modelo.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
            except: pass
    modelo.eval()
    return modelo, tipo

def analizar(imagen_pil, modelo):
    img = np.array(imagen_pil.convert("L"))
    img = xrv.datasets.normalize(img, 255)
    t = torch.from_numpy(img[None, None, :, :]).float()
    t = torch.nn.functional.interpolate(t, size=(224, 224))
    with torch.no_grad():
        feats = modelo.features2(t)
        preds = torch.sigmoid(modelo.classifier(feats)).numpy()[0]
    return preds

# HEADER CON LOGO
col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
with col_titulo:
    st.markdown("""
    <div style="padding-top:10px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:700;color:#00d4ff;letter-spacing:3px;">RUBÉN</div>
        <div style="font-size:0.75rem;color:#4a6080;letter-spacing:2px;">RADIOLOGÍA CON IA PARA DIAGNÓSTICO ESPECIALIZADO NEURONAL · SCS CANARIAS</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
modelo, tipo_modelo = cargar_modelo()
col_img, col_res = st.columns([1, 1.4], gap="large")

with col_img:
    archivo = st.file_uploader("Subir radiografía", type=["jpg","jpeg","png"])
    if archivo:
        st.image(Image.open(archivo), use_container_width=True)
        st.caption(f"Modelo: {tipo_modelo}")

with col_res:
    if archivo:
        with st.spinner("Analizando..."):
            preds = analizar(Image.open(archivo), modelo)
        positivos  = sorted([(CATEGORIAS[i],preds[i]) for i in range(14) if preds[i]>=UMBRAL_ALTO], key=lambda x:-x[1])
        sugestivos = sorted([(CATEGORIAS[i],preds[i]) for i in range(14) if UMBRAL_MEDIO<=preds[i]<UMBRAL_ALTO], key=lambda x:-x[1])
        todas = positivos + sugestivos

        if positivos:
            st.markdown(f'<div class="anormal"><div style="font-size:0.7rem;color:#ff3d5a;letter-spacing:3px;">▸ RESULTADO</div><div style="font-size:2.2rem;font-weight:700;color:#ff3d5a;">⚠ ANORMAL</div><div style="font-size:0.85rem;color:#ff8095;margin-top:8px;">{len(positivos)} hallazgo{"s" if len(positivos)>1 else ""} detectado{"s" if len(positivos)>1 else ""}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal"><div style="font-size:0.7rem;color:#00e676;letter-spacing:3px;">▸ RESULTADO</div><div style="font-size:2.2rem;font-weight:700;color:#00e676;">✓ NORMAL</div><div style="font-size:0.85rem;color:#80e8a0;margin-top:8px;">No se detectan hallazgos patológicos significativos</div></div>', unsafe_allow_html=True)

        if todas:
            zonas = list(dict.fromkeys([ZONA[p[0]] for p in todas]))
            zonas_html = "".join([f'<span class="zona-tag">{z}</span>' for z in zonas])
            st.markdown(f'<div class="seccion"><div style="font-size:0.7rem;color:#00d4ff;letter-spacing:2px;margin-bottom:6px;">▸ PASO 2 / 4</div><div style="font-size:1rem;font-weight:600;margin-bottom:14px;">Zona anatómica afectada</div>{zonas_html}</div>', unsafe_allow_html=True)

            hallazgos_html = "".join([f'<div class="hallazgo"><div style="font-family:monospace;font-size:0.8rem;color:#ffc400;margin-bottom:3px;">{cat} — {prob*100:.0f}%</div><div style="font-size:0.85rem;">{HALLAZGO[cat]}</div></div>' for cat,prob in todas])
            st.markdown(f'<div class="seccion"><div style="font-size:0.7rem;color:#00d4ff;letter-spacing:2px;margin-bottom:6px;">▸ PASO 3 / 4</div><div style="font-size:1rem;font-weight:600;margin-bottom:14px;">Hallazgos radiológicos</div>{hallazgos_html}</div>', unsafe_allow_html=True)

            if positivos:
                items = "".join([f'<div style="font-size:1.05rem;color:white;font-weight:500;padding:7px 0;border-bottom:1px solid #1a2540;">{"🔴" if i==0 else "🟡"} {cat} <span style="color:#4a6080;font-size:0.8rem;">({prob*100:.0f}%)</span></div>' for i,(cat,prob) in enumerate(positivos[:3])])
            else:
                items = '<div style="font-size:1rem;color:#c8d8f0;padding:7px 0;">⚪ Hallazgos sugestivos — correlacionar clínicamente</div>'
            st.markdown(f'<div class="diagnostico"><div style="font-size:0.7rem;color:#00d4ff;letter-spacing:3px;margin-bottom:12px;">▸ PASO 4 / 4 · DIAGNÓSTICO PRINCIPAL</div>{items}</div>', unsafe_allow_html=True)

            with st.expander("Ver todas las probabilidades"):
                for cat,prob in sorted(zip(CATEGORIAS,preds), key=lambda x:-x[1]):
                    st.progress(float(prob), text=f"{cat}: {prob*100:.0f}%")
    else:
        st.markdown('<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:400px;color:#2a4060;text-align:center;"><div style="font-size:4rem;margin-bottom:16px;">🩻</div><div style="font-family:monospace;font-size:0.9rem;letter-spacing:2px;">ESPERANDO IMAGEN</div></div>', unsafe_allow_html=True)

st.caption("⚕️ Solo para uso en investigación. No reemplaza el criterio clínico.")