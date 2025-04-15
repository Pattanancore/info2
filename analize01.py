import os
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import scipy.io
from io import BytesIO
from scipy.spatial import ConvexHull
import alphashape


# Set page layout
st.set_page_config(layout="wide")

# CONFIG
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

base_video_url = "https://raw.githubusercontent.com/nutteerabn/InfoVisual/main/processed%20hull%20area%20overlay/"
user = "nutteerabn"
repo = "InfoVisual"
clips_folder = "clips_folder"

# HELPERS
@st.cache_data(show_spinner=True)
def download_video(video_url: str, filename: str) -> str:
    if not os.path.exists(filename):
        r = requests.get(video_url)
        with open(filename, "wb") as f:
            f.write(r.content)
    return filename

@st.cache_data(show_spinner=True)
def load_gaze_data(user, repo, folder):
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}"
    r = requests.get(url)
    mat_files = [f["name"] for f in r.json() if f["name"].endswith(".mat")] if r.status_code == 200 else []
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

@st.cache_data(show_spinner=True)
def analyze_gaze(gaze_data, video_path, alpha=0.007, window=20):
    cap = cv2.VideoCapture(video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, convex, concave, images, i = [], [], [], [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        points = [(int(np.clip(x[p], 0, 1) * (w - 1)), int(np.clip(y[p], 0, 1) * (h - 1)))
                  for x, y, t in gaze_data
                  for p in np.where((t / 1000 * fps).astype(int) == i)[0]]
        if len(points) >= 3:
            arr = np.array(points)
            convex_area = ConvexHull(arr).volume if len(arr) >= 3 else 0
            try:
                shape = alphashape.alphashape(arr, alpha)
                concave_area = shape.area if shape.geom_type == 'Polygon' else 0
            except:
                concave_area = 0
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
    df['F-C score'] = 1 - (df['Convex Area (Rolling)'] - df['Concave Area (Rolling)']) / df['Convex Area (Rolling)']
    df['F-C score'] = df['F-C score'].fillna(0)
    return df, images

# UI
st.title("🎯 Stay Focused or Float Away? : Focus-Concentration Analysis")
selected_video = st.selectbox("🎬 Select a video", list(video_files.keys()))

if selected_video:
    st.video(base_video_url + video_files[selected_video])
    folder = f"{clips_folder}/{selected_video}"
    with st.spinner("Running analysis..."):
        gaze = load_gaze_data(user, repo, folder)
        video_filename = f"{selected_video}.mp4"
        video_filename = download_video(base_video_url + video_files[selected_video], video_filename)
        df, frames = analyze_gaze(gaze, video_filename)
        st.session_state.df = df
        st.session_state.frames = frames
        st.session_state.frame = int(df.index.min())

if "df" in st.session_state:
    df = st.session_state.df
    frames = st.session_state.frames
    frame = st.slider("🎞️ Select Frame", int(df.index.min()), int(df.index.max()), st.session_state.frame)
    

    col1, col2 = st.columns([2, 1])
    with col1:
        data = df.reset_index().melt(id_vars="Frame", value_vars=["Convex Area (Rolling)", "Concave Area (Rolling)"], var_name="Metric", value_name="Area")
        chart = alt.Chart(data).mark_line().encode(x="Frame:Q", y="Area:Q", color="Metric:N").properties(width=600, height=300)
        rule = alt.Chart(pd.DataFrame({'Frame': [frame]})).mark_rule(color='red').encode(x='Frame')
        st.altair_chart(chart + rule, use_container_width=True)
    with col2:
        rgb = cv2.cvtColor(frames[frame], cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Frame {frame}", use_container_width=True)
        st.metric("F-C Score", f"{df.loc[frame, 'F-C score']:.3f}")
