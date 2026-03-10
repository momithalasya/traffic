import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from yolo_detector import detect_objects, model as yolo_model
from vlm_module import analyze_scene
from llm_module import generate_response

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficSentinel AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c10;
    color: #e0e8f0;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Header */
.sentinel-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.sentinel-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #fff;
    text-transform: uppercase;
    margin: 0;
    text-shadow: 0 0 30px rgba(0,200,255,0.4);
}
.sentinel-header h1 span { color: #00c8ff; }
.sentinel-header p {
    font-family: 'Share Tech Mono', monospace;
    color: #4a7a99;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    margin-top: 0.4rem;
}

/* Upload zone */
.upload-zone {
    border: 1px solid #1a3a4a;
    border-radius: 4px;
    background: #0a1520;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Section headers */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #00c8ff;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1a3a4a, transparent);
}

/* Detection cards */
.detection-card {
    background: #0a1520;
    border: 1px solid #1a3a4a;
    border-left: 3px solid #00c8ff;
    border-radius: 4px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    font-family: 'Rajdhani', sans-serif;
}
.detection-card.fire { border-left-color: #ff4444; }
.detection-card .det-label {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #e0e8f0;
}
.detection-card .det-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a7a99;
    margin-top: 0.3rem;
}
.conf-bar-bg {
    background: #0f1e2a;
    border-radius: 2px;
    height: 4px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(to right, #00c8ff, #0088aa);
}
.conf-bar-fill.fire { background: linear-gradient(to right, #ff4444, #aa1111); }

/* Emergency level badge */
.emergency-badge {
    display: inline-block;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.2em;
    padding: 0.3rem 1rem;
    border-radius: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.badge-critical { background: #3a0808; color: #ff4444; border: 1px solid #ff4444; }
.badge-high     { background: #2a1a00; color: #ff8800; border: 1px solid #ff8800; }
.badge-medium   { background: #1a1a00; color: #ffcc00; border: 1px solid #ffcc00; }
.badge-low      { background: #001a0a; color: #00ff88; border: 1px solid #00ff88; }

/* LLM response box */
.llm-box {
    background: #0a1520;
    border: 1px solid #1a3a4a;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    color: #a0c8e0;
    white-space: pre-wrap;
}

/* VLM description box */
.vlm-box {
    background: #0a1520;
    border: 1px solid #1a3a4a;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #c0d8e8;
}

/* No incident */
.no-incident {
    text-align: center;
    padding: 2rem;
    font-family: 'Share Tech Mono', monospace;
    color: #00ff88;
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    border: 1px solid #003a1a;
    background: #001a0a;
    border-radius: 4px;
}

/* Analyze button */
div[data-testid="stButton"] button {
    background: transparent;
    border: 1px solid #00c8ff !important;
    color: #00c8ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2.5rem !important;
    border-radius: 2px !important;
    width: 100%;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover {
    background: rgba(0,200,255,0.08) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.2) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #00c8ff !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sentinel-header">
    <h1>Traffic<span>Sentinel</span> AI</h1>
    <p>// REAL-TIME INCIDENT DETECTION & EMERGENCY DISPATCH SYSTEM //</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="section-label">📁 Input Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an image here",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡  ANALYZE INCIDENT", disabled=not uploaded_file)

with right_col:
    if not uploaded_file:
        st.markdown("""
        <div style="height:400px; display:flex; align-items:center; justify-content:center;
                    border: 1px dashed #1a3a4a; border-radius:4px; color:#1a3a4a;
                    font-family:'Share Tech Mono',monospace; font-size:0.8rem; letter-spacing:0.15em;">
            AWAITING IMAGE INPUT
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        # ── YOLO ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">🎯 YOLO Detections</div>', unsafe_allow_html=True)

        with st.spinner("Running YOLO detection..."):
            detections = detect_objects(tmp_path)

            # Draw bounding boxes
            img_cv = cv2.imread(tmp_path)
            results = yolo_model(tmp_path)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = "Accident" if cls == 0 else "Car Fire"
                    color = (0, 200, 255) if cls == 0 else (0, 60, 255)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_cv, f"{label} {conf:.0%}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            annotated_path = tmp_path.replace(".png", "_annotated.png")
            cv2.imwrite(annotated_path, img_cv)

        # Show annotated image
        st.markdown('<div class="section-label">🖼 Annotated Image</div>', unsafe_allow_html=True)
        st.image(annotated_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Detection cards
        if detections:
            for label, conf in detections:
                card_class = "fire" if "Fire" in label else ""
                bar_class = "fire" if "Fire" in label else ""
                st.markdown(f"""
                <div class="detection-card {card_class}">
                    <div class="det-label">{'🔥' if 'Fire' in label else '💥'} {label.upper()}</div>
                    <div class="det-conf">CONFIDENCE: {conf*100:.1f}%</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill {bar_class}" style="width:{conf*100:.1f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-incident">✓ NO THREATS DETECTED</div>', unsafe_allow_html=True)

        incident_detected = any(conf > 0.5 for _, conf in detections)

        if incident_detected:
            # ── VLM ───────────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">👁 Scene Analysis (VLM)</div>', unsafe_allow_html=True)

            with st.spinner("Analyzing scene with Moondream2..."):
                scene_description = analyze_scene(tmp_path, detections=detections)

            st.markdown(f'<div class="vlm-box">{scene_description}</div>', unsafe_allow_html=True)

            # ── LLM ───────────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">🚨 Emergency Response (LLM)</div>', unsafe_allow_html=True)

            with st.spinner("Generating emergency response..."):
                llm_response = generate_response(scene_description, detections)

            # Extract emergency level for badge
            badge_html = ""
            for line in llm_response.splitlines():
                if "Emergency Level:" in line:
                    level = line.split(":")[-1].strip().lower()
                    badge_class = f"badge-{level}" if level in ["critical","high","medium","low"] else "badge-medium"
                    badge_html = f'<div class="emergency-badge {badge_class}">⚠ EMERGENCY LEVEL: {level.upper()}</div>'
                    break

            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(f'<div class="llm-box">{llm_response}</div>', unsafe_allow_html=True)

        # Cleanup
        os.unlink(tmp_path)
        if os.path.exists(annotated_path):
            os.unlink(annotated_path)