# appfereth.py (KODE LENGKAP UNTUK DEPLOYMENT STREAMLIT DENGAN CACHING SEMUA FILE BESAR)

import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import logging
import requests
import os

# Konfigurasi logger untuk menekan spam TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# ===================== CONFIG MODEL & LABEL =====================

# URL Google Drive Model (Pastikan ini adalah tautan Unduhan Langsung)
EMOTION_MODEL_URL = "https://drive.google.com/uc?export=download&id=165xIiid5rsRIT3n8X5NfTi73B3OL2YhL"
ETHNICITY_MODEL_URL = "https://drive.google.com/uc?export=download&id=1URsi1OFfjUIaLI33GI7LSNrLJzygXn63"

# URL MobileNetV2 Weights (ImageNet, no_top)
MOBILE_NET_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"

# Nama file lokal yang akan disimpan/dimuat
EMOTION_MODEL_FILE = "modelensembleemosi.joblib"
ETHNICITY_MODEL_FILE = "modeletnisrf.joblib"
MOBILE_NET_FILE = "mobilenet_v2_weights.h5" 

# Label mapping
EMOTION_LABELS = {0: 'fear', 1: 'surprised', 2: 'angry', 3: 'sad', 4: 'disgusted', 5: 'happy'}
ETHNICITY_LABELS = {0: 'Ambon (A)', 1: 'Toraja (T)', 2: 'Kaukasia (K)', 3: 'Jepang (J)'}

# Konfigurasi CNN
CNN_INPUT_SIZE = (160, 160)
CNN_POOLING = 'avg'
CNN_LAYER_TRAINABLE = False

# ===================== FUNGSI CACHING & UNDUHAN MODEL =====================

@st.cache_resource
def cache_external_file(url, filename, display_name):
    """Mengunduh file (weights/joblib) dari URL dan menyimpannya secara lokal jika belum ada."""
    if not os.path.exists(filename):
        try:
            st.info(f"Mengunduh {display_name} dari server...")
            response = requests.get(url, stream=True)
            response.raise_for_status() # Cek error HTTP/koneksi
            
            with open(filename, 'wb') as f:
                # Menyimpan file dalam potongan (chunks)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"✅ {display_name} berhasil diunduh dan disimpan.")
        
        except Exception as e:
            st.error(f"Gagal mengunduh {display_name} dari {url}. Error: {e}")
            raise
    return filename # Mengembalikan nama file lokal

# Fungsi terpisah untuk memuat Joblib (di luar cache resource agar tidak menimpa)
def load_joblib_model(filename):
    return joblib.load(filename)


# ===================== LOAD MODEL & EMBEDDER =====================

class CNNEmbedder:
    # Menerima path_to_weights sebagai argumen
    def __init__(self, path_to_weights, input_size=CNN_INPUT_SIZE, pooling=CNN_POOLING, trainable=CNN_LAYER_TRAINABLE):
        self.input_size = input_size
        
        # Muat model dasar MobileNetV2 menggunakan path file lokal
        base = MobileNetV2(
            include_top=False, 
            weights=path_to_weights, # Menggunakan path file lokal
            input_shape=(input_size[0], input_size[1], 3), 
            pooling=pooling
        )
        
        base.trainable = trainable
        self.model = base

    def compute(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
             return np.zeros((1280,))
             
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_AREA)
        arr = img_to_array(img_resized)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        # Gunakan tf.convert_to_tensor
        emb = self.model.predict(tf.convert_to_tensor(arr), verbose=0) 
        return emb.flatten()

try:
    # 1. Cache weights MobileNetV2 (file H5)
    weights_path = cache_external_file(MOBILE_NET_URL, MOBILE_NET_FILE, "MobileNetV2 Weights")
    
    # 2. Cache dan Muat model Joblib (emosi dan etnisitas)
    cache_external_file(EMOTION_MODEL_URL, EMOTION_MODEL_FILE, "Model Emosi")
    en_emotion_model = load_joblib_model(EMOTION_MODEL_FILE)
    
    cache_external_file(ETHNICITY_MODEL_URL, ETHNICITY_MODEL_FILE, "Model Etnisitas")
    rf_ethnicity_model = load_joblib_model(ETHNICITY_MODEL_FILE)
    
    # 3. Inisialisasi CNNEmbedder menggunakan path weights lokal
    cnn_embedder = CNNEmbedder(path_to_weights=weights_path)

    st.sidebar.success(f"✅ Semua model dan weights berhasil dimuat.")

except Exception as e:
    st.error(f"❌ ERROR Fatal: Tidak dapat menyiapkan model untuk klasifikasi. Pastikan Anda telah menambahkan 'dill' dan 'xgboost' ke requirements.txt. Detail: {e}")
    st.stop()


# ===================== MEDIA PIPE & UTILITY FUNCTIONS =====================

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- REPLIKA UTAMA FUNGSI GEOMETRIS LAMA (UNTUK MODEL EMOSI) ---

def angle_between_old(p1, p2, p3): 
    v1 = p1 - p2
    v2 = p3 - p2
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cosang)

def triangle_area_old(p1, p2, p3): 
    v1 = p2 - p1
    v2 = p3 - p1
    if len(v1) == 2:
        v1 = np.append(v1, 0)
        v2 = np.append(v2, 0)
    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
    return area

def extract_features_basic(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    feats = [
        dist(33, 133), dist(362, 263), dist(61, 291), dist(13, 14),
        dist(159, 145), dist(386, 374), dist(10, 152),
    ]
    ear_left = (dist(159, 145) + dist(160, 144)) / (2.0 * dist(33, 133) + 1e-6)
    ear_right = (dist(386, 374) + dist(387, 373)) / (2.0 * dist(362, 263) + 1e-6)
    mar = dist(13, 14) / (dist(61, 291) + 1e-6)
    brow_left = dist(70, 105)
    brow_right = dist(336, 334)
    feats.extend([ear_left, ear_right, mar, brow_left, brow_right])
    return np.array(feats)

def extract_features_symmetry_ratio(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    ear_left = (dist(159, 145) + dist(160, 144)) / (2.0 * dist(33, 133) + 1e-6)
    ear_right = (dist(386, 374) + dist(387, 373)) / (2.0 * dist(362, 263) + 1e-6)
    mar = dist(13, 14) / (dist(61, 291) + 1e-6)
    face_len = dist(10, 152) + 1e-6
    brow_left = dist(70, 105)
    brow_right = dist(336, 334)
    
    left_mask = lm[:,0] < 0
    right_mask = lm[:,0] >= 0
    sym_x = abs(np.mean(np.abs(lm[left_mask,0])) - np.mean(np.abs(lm[right_mask,0]))) if left_mask.any() and right_mask.any() else 0.0
    sym_y = abs(np.mean(lm[left_mask,1]) - np.mean(lm[right_mask,1])) if left_mask.any() and right_mask.any() else 0.0
    
    ear_sym = ear_left / (ear_right + 1e-6)
    mar_norm = mar / face_len
    brow_asym = abs(brow_left - brow_right)
    avg_ear = (ear_left + ear_right) / 2.0
    mar_over_ear = mar / (avg_ear + 1e-6)
    mar_over_face = mar / (face_len + 1e-6)
    ear_diff = abs(ear_left - ear_right)
    return np.array([
        ear_sym, mar_norm, brow_asym, avg_ear, mar,
        sym_x, sym_y, mar_over_ear, mar_over_face, ear_diff
    ])

def extract_features_angles_areas(lm):
    idx = {
        'mouth_l': 61, 'mouth_r': 291, 'lip_up': 13,
        'nose': 1, 'eye_l_o': 33, 'eye_l_i': 133, 'eye_r_o': 362, 'eye_r_i': 263
    }
    p = {k: lm[v] for k, v in idx.items()}
    
    ang_mouth_nose_l = angle_between_old(p['mouth_l'], p['nose'], p['lip_up'])
    ang_mouth_nose_r = angle_between_old(p['mouth_r'], p['nose'], p['lip_up'])
    ang_eye_left = angle_between_old(p['eye_l_o'], p['eye_l_i'], p['mouth_l'])
    ang_eye_right = angle_between_old(p['eye_r_o'], p['eye_r_i'], p['mouth_r'])
    area_eye_left = triangle_area_old(p['eye_l_o'], p['eye_l_i'], p['nose'])
    area_eye_right = triangle_area_old(p['eye_r_o'], p['eye_r_i'], p['nose'])
    area_mouth = triangle_area_old(p['mouth_l'], p['mouth_r'], p['lip_up'])
    return np.array([
        ang_mouth_nose_l, ang_mouth_nose_r,
        ang_eye_left, ang_eye_right,
        area_eye_left, area_eye_right, area_mouth
    ])

def extract_class_specific_features(lm):
    def dist(a, b): return np.linalg.norm(lm[a] - lm[b])
    face_len = dist(10, 152) + 1e-6
    mouth_left_nose = dist(61, 1) / face_len
    mouth_right_nose = dist(291, 1) / face_len
    mouth_corner_asym = abs(mouth_left_nose - mouth_right_nose)
    
    eye_left_center = (lm[33] + lm[133]) / 2.0
    eye_right_center = (lm[362] + lm[263]) / 2.0
    brow_left_center = (lm[70] + lm[105]) / 2.0
    brow_right_center = (lm[336] + lm[334]) / 2.0
    
    brow_lift_left = np.linalg.norm(brow_left_center - eye_left_center) / face_len
    brow_lift_right = np.linalg.norm(brow_right_center - eye_right_center) / face_len
    brow_lift_asym = abs(brow_lift_left - brow_lift_right)
    lip_up = 13
    lip_low = 14
    mouth_open_ratio = dist(lip_up, lip_low) / (dist(61, 291) + 1e-6)
    return np.array([
        mouth_left_nose, mouth_right_nose, mouth_corner_asym,
        brow_lift_left, brow_lift_right, brow_lift_asym,
        mouth_open_ratio
    ])

# FUNGSI UNTUK MODEL EMOSI (GEOMETRI LAMA + TEKSTUR)
def build_feature_vector_emotion(lm_norm, cnn_emb=None):
    parts = [
        lm_norm.flatten(),
        extract_features_basic(lm_norm),
        extract_features_symmetry_ratio(lm_norm),
        extract_features_angles_areas(lm_norm),
        extract_class_specific_features(lm_norm)
    ]
    if cnn_emb is not None:
        parts.append(cnn_emb) 
    
    return np.concatenate(parts)

# --- REPLIKA FUNGSI GEOMETRIS BARU (UNTUK MODEL ETNISITAS) ---

def _distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def _slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx + 1e-6))

def _curvature(p1, p2, p3):
    angle = _angle(p1, p2, p3)
    return 180 - angle

def _eye_aspect_ratio(landmarks, eye_indices):
    points = landmarks[eye_indices]
    v1 = _distance(points[1], points[5]) 
    v2 = _distance(points[2], points[4]) 
    h = _distance(points[0], points[3]) 
    return (v1 + v2) / (2.0 * h + 1e-6)

def _mouth_aspect_ratio(landmarks):
    v = _distance(landmarks[13], landmarks[14])
    h = _distance(landmarks[61], landmarks[291])
    return v / (h + 1e-6)

def calculate_all_features_ethnicity(landmarks):
    lm_2d = landmarks[:, :2] 
    
    FEAT_INDICES = {
        'LEFT_EYE': [33, 160, 158, 133, 153, 144], 
        'RIGHT_EYE': [362, 385, 387, 263, 373, 380],
    }

    features = {}

    # 1. EYE FEATURES (menggunakan lm_2d)
    left_ear = _eye_aspect_ratio(lm_2d, FEAT_INDICES['LEFT_EYE'])
    right_ear = _eye_aspect_ratio(lm_2d, FEAT_INDICES['RIGHT_EYE'])
    features['left_ear'] = left_ear
    features['right_ear'] = right_ear
    features['ear_diff'] = abs(left_ear - right_ear)
    features['ear_avg'] = (left_ear + right_ear) / 2

    left_eye_width = _distance(lm_2d[33], lm_2d[133])
    right_eye_width = _distance(lm_2d[362], lm_2d[263])
    features['left_eye_width'] = left_eye_width
    features['right_eye_width'] = right_eye_width
    features['eye_width_ratio'] = left_eye_width / (right_eye_width + 1e-6)

    left_eye_height = _distance(lm_2d[159], lm_2d[145])
    right_eye_height = _distance(lm_2d[386], lm_2d[374])
    features['left_eye_height'] = left_eye_height
    features['right_eye_height'] = right_eye_height
    features['eye_distance'] = _distance(lm_2d[33], lm_2d[263]) # Inter-eye

    # 2. EYEBROW FEATURES
    left_brow_height = _distance(lm_2d[70], lm_2d[23])
    right_brow_height = _distance(lm_2d[300], lm_2d[253])
    features['left_brow_height'] = left_brow_height
    features['right_brow_height'] = right_brow_height
    features['brow_height_diff'] = abs(left_brow_height - right_brow_height)

    left_brow_slope = _slope(lm_2d[66], lm_2d[107])
    right_brow_slope = _slope(lm_2d[296], lm_2d[336])
    features['left_brow_slope'] = left_brow_slope
    features['right_brow_slope'] = right_brow_slope
    features['brow_slope_diff'] = abs(left_brow_slope - right_brow_slope)

    left_brow_curve = _curvature(lm_2d[66], lm_2d[70], lm_2d[107])
    right_brow_curve = _curvature(lm_2d[296], lm_2d[300], lm_2d[336])
    features['left_brow_curve'] = left_brow_curve
    features['right_brow_curve'] = right_brow_curve

    # 3. MOUTH FEATURES
    mar = _mouth_aspect_ratio(lm_2d)
    features['mouth_ar'] = mar
    mouth_width = _distance(lm_2d[61], lm_2d[291])
    features['mouth_width'] = mouth_width
    mouth_height_outer = _distance(lm_2d[0], lm_2d[17])
    mouth_height_inner = _distance(lm_2d[13], lm_2d[14])
    features['mouth_height_outer'] = mouth_height_outer
    features['mouth_height_inner'] = mouth_height_inner

    left_corner_y = lm_2d[61][1]
    right_corner_y = lm_2d[291][1]
    mouth_center_y = lm_2d[13][1]
    features['left_smile'] = mouth_center_y - left_corner_y
    features['right_smile'] = mouth_center_y - right_corner_y
    features['smile_symmetry'] = abs(features['left_smile'] - features['right_smile'])
    features['mouth_corner_angle'] = _angle(lm_2d[61], lm_2d[13], lm_2d[291])

    upper_lip_curve = _curvature(lm_2d[61], lm_2d[0], lm_2d[291])
    features['upper_lip_curve'] = upper_lip_curve
    lower_lip_curve = _curvature(lm_2d[61], lm_2d[17], lm_2d[291])
    features['lower_lip_curve'] = lower_lip_curve

    # 4. JAW & FACE SHAPE FEATURES
    jaw_width = _distance(lm_2d[234], lm_2d[454])
    features['jaw_width'] = jaw_width
    face_height = _distance(lm_2d[10], lm_2d[152])
    features['face_height'] = face_height
    features['face_aspect_ratio'] = face_height / (jaw_width + 1e-6)

    jaw_angle_left = _angle(lm_2d[234], lm_2d[152], lm_2d[10])
    jaw_angle_right = _angle(lm_2d[454], lm_2d[152], lm_2d[10])
    features['jaw_angle_left'] = jaw_angle_left
    features['jaw_angle_right'] = jaw_angle_right

    # 5. NOSE FEATURES
    nose_width = _distance(lm_2d[98], lm_2d[327]) 
    features['nose_width'] = nose_width
    nose_height = _distance(lm_2d[168], lm_2d[2])
    features['nose_height'] = nose_height
    nose_bridge_angle = _angle(lm_2d[168], lm_2d[6], lm_2d[2])
    features['nose_bridge_angle'] = nose_bridge_angle
    features['nose_tip_y'] = lm_2d[1][1]

    # 6. PROPORTIONS
    features['eye_to_face_ratio'] = features['eye_distance'] / (jaw_width + 1e-6)
    features['nose_to_face_ratio'] = nose_width / (jaw_width + 1e-6)
    features['eye_height_to_face'] = features['ear_avg'] / (face_height + 1e-6)
    features['mouth_to_face_ratio'] = mouth_width / (jaw_width + 1e-6)
    eye_center_y = (lm_2d[33][1] + lm_2d[263][1]) / 2
    mouth_center_y = lm_2d[13][1]
    features['eye_mouth_distance'] = abs(mouth_center_y - eye_center_y)

    # 7. SYMMETRY FEATURES
    left_face_width = _distance(lm_2d[234], lm_2d[10])
    right_face_width = _distance(lm_2d[454], lm_2d[10])
    features['face_symmetry'] = abs(left_face_width - right_face_width)
    features['vertical_symmetry'] = abs(lm_2d[10][0] - 0.5) 

    # 8. ANGLE FEATURES
    features['face_tilt'] = _slope(lm_2d[234], lm_2d[454])
    features['eye_tilt'] = _slope(lm_2d[33], lm_2d[263])

    return features

# FUNGSI UNTUK MODEL ETNISITAS (GEOMETRI BARU + TEKSTUR)
def build_feature_vector_ethnicity(lm_raw_mp, cnn_emb=None):
    geom_features_dict = calculate_all_features_ethnicity(lm_raw_mp)
    geom_features_vector = np.array(list(geom_features_dict.values()))
    
    parts = [geom_features_vector]
    
    if cnn_emb is not None:
        parts.append(cnn_emb)
    
    return np.concatenate(parts)

# ===================== FACE CROP FOR CNN (SHARED) =====================
def crop_face_from_raw_landmarks(img_bgr, lm_raw, pad=0.2):
    h, w = img_bgr.shape[:2]
    xs = (lm_raw[:,0] * w).astype(np.float32)
    ys = (lm_raw[:,1] * h).astype(np.float32)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_pad = pad * (x_max - x_min)
    y_pad = pad * (y_max - y_min)
    x1 = max(0, int(x_min - x_pad))
    y1 = max(0, int(y_min - y_pad))
    x2 = min(w, int(x_max + x_pad))
    y2 = min(h, int(y_max + y_pad))
    
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


# ===================== KELAS VIDEO TRANSFORMER UNTUK WEBRTC =====================

class RealTimeClassifier(VideoTransformerBase):
    def __init__(self):
        # Inisialisasi MediaPipe FaceMesh
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Gunakan model global yang sudah dimuat di awal
        self.en_emotion_model = en_emotion_model
        self.rf_ethnicity_model = rf_ethnicity_model
        self.cnn_embedder = cnn_embedder
        
    # Fungsi utama yang dipanggil untuk setiap frame
    def transform(self, frame):
        # Frame dari webrtc_streamer datang sebagai VideoFrame (RGB), diubah menjadi array NumPy BGR untuk OpenCV
        image = frame.to_ndarray(format="bgr24")
        
        # Balik gambar secara horizontal
        image = cv2.flip(image, 1)
        
        # Konversi ke RGB untuk MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        
        emotion_result = "Tidak Terdeteksi"
        ethnicity_result = "Tidak Terdeteksi"
        emo_confidence = 0.0
        eth_confidence = 0.0
        bbox_coords = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # --- 1. Ekstraksi Landmark Mentah dan Normalisasi ---
                lm_raw_mp = np.array([[p.x, p.y, p.z] for p in face_landmarks.landmark])
                lm_norm = lm_raw_mp.copy()
                lm_norm = lm_norm - lm_norm.mean(axis=0)
                lm_norm = lm_norm / (lm_norm.std(axis=0) + 1e-6)

                # --- 2. Crop Wajah untuk CNN Embedding ---
                cropped_face, bbox_coords = crop_face_from_raw_landmarks(image, lm_raw_mp, pad=0.2)
                x1, y1, x2, y2 = bbox_coords 
                
                # --- 3. Ekstraksi CNN Embedding (shared) ---
                cnn_emb = self.cnn_embedder.compute(cropped_face)
                
                # --- 4. Buat Vektor Fitur KHUSUS untuk SETIAP MODEL ---
                feature_vector_emotion = build_feature_vector_emotion(lm_norm, cnn_emb)
                X_live_emo = feature_vector_emotion.reshape(1, -1)
                feature_vector_ethnicity = build_feature_vector_ethnicity(lm_raw_mp, cnn_emb)
                X_live_eth = feature_vector_ethnicity.reshape(1, -1)
                
                # --- 5. Prediksi ---
                try:
                    emo_proba = self.en_emotion_model.predict_proba(X_live_emo)[0]
                    emo_pred_idx = np.argmax(emo_proba)
                    emo_confidence = emo_proba[emo_pred_idx] * 100
                    emotion_result = EMOTION_LABELS.get(emo_pred_idx, "UNKNOWN EMO")
                    
                    eth_proba = self.rf_ethnicity_model.predict_proba(X_live_eth)[0]
                    eth_pred_idx = np.argmax(eth_proba)
                    eth_confidence = eth_proba[eth_pred_idx] * 100
                    ethnicity_result = ETHNICITY_LABELS.get(eth_pred_idx, "UNKNOWN ETH")
                
                except AttributeError:
                    emotion_result = "Prediksi Gagal (No Proba)"
                    ethnicity_result = "Prediksi Gagal (No Proba)"

                # --- 6. Visualisasi (Di dalam loop face_landmarks) ---
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION, 
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Visualisasi Bounding Box dan Teks (di luar loop face_landmarks)
            if bbox_coords is not None:
                x1, y1, x2, y2 = bbox_coords
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Tampilkan hasil dengan confidence
                text_emo = f"Emotion: {emotion_result} ({emo_confidence:.2f}%)"
                text_eth = f"Ethnicity: {ethnicity_result} ({eth_confidence:.2f}%)"
                
                cv2.rectangle(image, (x1, y1 - 60), (x2, y1 - 30), (0, 0, 0), -1)
                cv2.putText(image, text_emo, (x1 + 5, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.rectangle(image, (x1, y1 - 30), (x2, y1), (0, 0, 0), -1)
                cv2.putText(image, text_eth, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Kembalikan frame yang sudah dimodifikasi (BGR)
        return image.copy()


# ===================== MAIN STREAMLIT APP =====================

def main():
    st.title("Klasifikasi Emosi & Etnisitas Wajah Real-Time")
    st.markdown("""
    Deploy Model Machine Learning Berbasis Web ini menggunakan model ML (Random Forest, Support Vector Machine, XGBoost, Ensemble Voting Classifier) 
    yang dikombinasikan dengan **Raw Landmarks &Geometric Features (MediaPipe FaceMesh)** dan **Texture/CCN Feature Embedder (MobileNetV2)** untuk klasifikasi Emosi dan Etnisitas secara bersamaan.
    """)
    st.warning("⚠️ **PENTING:** Pastikan Anda memberikan izin akses kamera saat diminta oleh browser.")

    # Implementasi webrtc_streamer
    webrtc_streamer(
        key="realtime-face-classifier",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=RealTimeClassifier,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    st.sidebar.header("Detail Model")
    st.sidebar.markdown(f"**Emosi:** {EMOTION_MODEL_FILE} (Geometric Features V2 + CNN Features)")
    st.sidebar.markdown(f"**Etnisitas:** {ETHNICITY_MODEL_FILE} (Geometric Features V1 + CNN Features)")
    st.sidebar.markdown(f"**CNN Embedder:** MobileNetV2 ({CNN_INPUT_SIZE})")

if __name__ == "__main__":
    main()

