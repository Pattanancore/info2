# --- Imports ---
import os
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
import scipy.io
from io import BytesIO
from scipy.spatial import ConvexHull
import alphashape

# --- Streamlit setup ---
st.set_page_config(page_title="Gaze Hull Visualizer", layout="wide")

# --- Constants ---
video_files = {
    "APPAL_2a": "APPAL_2a_hull_area.mp4",
    "FOODI_2a": "FOODI_2a_hull_area.mp4",
    "MARCH_12a": "MARCH_12a_hull_area.mp4",
    "NANN_3a": "NANN_3a_hull_area.mp4",
    "SHREK_3a": "SHREK_3a_hull_area.mp4",
    "SIMPS_19a": "SIMPS_19a_hull_area.mp4",
    "SIMPS_9a": "SIMPS_9a_hull_area.mp4",
    "SUND_36a_POR": "SUND_36a_POR_hull_area.mp4",
}

gif_files = {
    k: k + ".gif" for k in video_files
}

base_video_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/processed%20hull%20area%20overlay/"
base_gif_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/gif_sample/"
user = "nutteerabn"
repo = "InfoVisual"
clips_folder = "clips_folder"

# --- Init session ---
st.session_state.setdefault("analyzed_videos", {})
st.session_state.setdefault("current_frame", 0)

# --- Helper functions ---
@st.cache_data(ttl=3600)
def list_mat_files(user, repo, folder):
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    return [f["name"] for f in r.json() if f["name"].endswith(".mat")]

@st.cache_data(ttl=3600)
def load_gaze_data(user, repo, folder):
    mat_files = list_mat_files(user, repo, folder)
    gaze_data = []
    for file in mat_files:
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{folder}/{file}"
        res = requests.get(raw_url)
        if res.status_code == 200:
            mat = scipy.io.loadmat(BytesIO(res.content))
            record = mat['eyetrackRecord']
            x, y, t = record['x'][0, 0].flatten(), record['y'][0, 0].flatten(), record['t'][0, 0].flatten()
            valid = (x != -32768) & (y != -32768)
            gaze_data.append({
                'x': x[valid] / np.max(x[valid]),
                'y': y[valid] / np.max(y[valid]),
                't': t[valid] - t[valid][0]
            })
    return [(d['x'], d['y'], d['t']) for d in gaze_data]

@st.cache_data(ttl=3600)
def download_video(video_url, video_name):
    path = f"{video_name}.mp4"
    if not os.path.exists(path):
        r = requests.get(video_url)
        with open(path, "wb") as f:
            f.write(r.content)
    return path

@st.cache_data(ttl=3600)
def analyze_gaze(gaze_data, video_path, alpha=0.007, window=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, convex, concave, images = [], [], [], []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        points = []
        for x, y, t in gaze_data:
            idx = (t / 1000 * fps).astype(int)
            if i in idx:
                for p in np.where(idx == i)[0]:
                    px = int(np.clip(x[p], 0, 1) * (w - 1))
                    py = int(np.clip(y[p], 0, 1) * (h - 1))
                    points.append((px, py))
        if len(points) >= 3:
            arr = np.array(points)
            try: convex_area = ConvexHull(arr).volume
            except: convex_area = 0
            try:
                shape = alphashape.alphashape(arr, alpha)
                concave_area = shape.area if shape.geom_type == 'Polygon' else 0
            except: concave_area = 0
        else:
            convex_area = concave_area = 0
        frames.append(i)
        convex.append(convex_area)
        concave.append(concave_area)
        images.append(frame)
        i += 1
    cap.release()
    df = pd.DataFrame({'Frame': frames, 'Convex Area': convex, 'Concave Area': concave}).set_index('Frame')
    df['Convex Area (Rolling)'] = df['Convex Area'].rolling(window, min_periods=1).mean()
    df['Concave Area (Rolling)'] = df['Concave Area'].rolling(window, min_periods=1).mean()
    df['F-C score'] = 1 - (df['Convex Area (Rolling)'] - df['Concave Area (Rolling)']) / (df['Convex Area (Rolling)'] + 1e-8)
    df['F-C score'] = df['F-C score'].fillna(0)
    return df, images

# --- Layout ---
st.title("🎯 Understanding Viewer Focus Through Gaze Visualization")

col1, col2 = st.columns(2)
with col1:
    with st.expander("📌 Goal of This Visualization", expanded=True):
        st.markdown("...")  # Your text here
    with st.expander("📐 Convex vs Concave"):
        st.markdown("...")

with col2:
    with st.expander("📊 Focus-Concentration (F-C) Score"):
        st.markdown("...")
    with st.expander("🎥 Example: High vs Low"):
        st.markdown("...")

# --- Select Video ---
st.markdown("### 🎬 Select a Video")
selected_video = st.selectbox("🎥 เลือกวิดีโอ", list(video_files.keys()))

# --- Load & Analyze ---
if selected_video:
    if selected_video in gif_files:
        gif_url = base_gif_url + gif_files[selected_video]
        st.image(gif_url, use_container_width=True, caption=f"{selected_video} Gaze Visualization")
    else:
        st.video(base_video_url + video_files[selected_video])

    if selected_video not in st.session_state.analyzed_videos:
        with st.spinner("Analyzing video..."):
            folder = f"{clips_folder}/{selected_video}"
            gaze = load_gaze_data(user, repo, folder)
            video_path = download_video(base_video_url + video_files[selected_video], selected_video)
            df, frames = analyze_gaze(gaze, video_path)
            st.session_state.analyzed_videos[selected_video] = {
                "df": df, "frames": frames,
                "min_frame": int(df.index.min()),
                "max_frame": int(df.index.max())
            }
            st.session_state.current_frame = int(df.index.min())

    data = st.session_state.analyzed_videos[selected_video]
    df, frames = data["df"], data["frames"]
    min_frame, max_frame = data["min_frame"], data["max_frame"]

    # --- Frame Selector ---
    current_frame = st.slider("🎞️ Select Frame", min_frame, max_frame, st.session_state.current_frame)
    st.session_state.current_frame = current_frame

    col_plot, col_video = st.columns([2, 1])
    with col_plot:
        melt = df.reset_index().melt(id_vars="Frame", value_vars=["Convex Area (Rolling)", "Concave Area (Rolling)"])
        chart = alt.Chart(melt).mark_line().encode(
            x="Frame", y="value", color=alt.Color("variable:N", scale=alt.Scale(range=["green", "blue"]))
        ).properties(height=300)
        rule = alt.Chart(pd.DataFrame({"Frame": [current_frame]})).mark_rule(color="red").encode(x="Frame")
        st.altair_chart(chart + rule, use_container_width=True)

    with col_video:
        if current_frame < len(frames):
            rgb = cv2.cvtColor(frames[current_frame], cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=f"Frame {current_frame}", use_container_width=True)
        score = df.loc[current_frame, 'F-C score']
        st.metric("Focus-Concentration Score", f"{score:.3f}")
