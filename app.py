from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import cv2
import facer
from PIL import Image
import joblib
import io

app = Flask(__name__)
CORS(app)

# K-Means 모델 로드
base_model_path = 'kmeans_model_L2.pkl'
kmeans_model = joblib.load(base_model_path)

# CUDA 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_image_url_for_season(season, detailed_season):
    base_url = 'http://localhost:5000'
    image_map = {
        "봄": {
            "라이트 봄": f"{base_url}/static/images/spring_light.jpg",
            "브라이트 봄": f"{base_url}/static/images/spring_bright.jpg"
        },
        "여름": {
            "라이트 여름": f"{base_url}/static/images/summer_light.jpg",
            "브라이트 여름": f"{base_url}/static/images/summer_bright.jpg",
            "뮤트 여름": f"{base_url}/static/images/summer_muted.jpg"
        },
        "가을": {
            "뮤트 가을": f"{base_url}/static/images/fall_muted.jpg",
            "스트롱 가을": f"{base_url}/static/images/fall_strong.jpg",
            "딥 가을": f"{base_url}/static/images/fall_deep.jpg"
        },
        "겨울": {
            "브라이트 겨울": f"{base_url}/static/images/winter_bright.jpg",
            "딥 겨울": f"{base_url}/static/images/winter_deep.jpg"
        }
    }
    return image_map.get(season, {}).get(detailed_season, f"{base_url}/static/images/default.jpg")


def detect_and_parse_face(sample):
    image = torch.from_numpy(sample).permute(2, 0, 1).unsqueeze(0).float().to(device=device)

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    return faces['seg']['logits'].softmax(dim=1).cpu()


def get_face_skin_mask(seg_probs):
    tensor = seg_probs.permute(0, 2, 3, 1).squeeze().numpy()
    return (tensor[:, :, 1] >= 0.5).astype(int)


def extract_skin_rgb(sample, binary_mask):
    if binary_mask.ndim == 3:
        binary_mask = binary_mask[:, :, 0]

    binary_mask_resized = cv2.resize(binary_mask, (sample.shape[1], sample.shape[0]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    skin_pixels = img[binary_mask_resized == 1]
    return skin_pixels


def calculate_quartile_means(rgb_codes):
    r = np.sort(rgb_codes[:, 0])
    g = np.sort(rgb_codes[:, 1])
    b = np.sort(rgb_codes[:, 2])

    q1_r, q3_r = np.percentile(r, 25), np.percentile(r, 75)
    q1_g, q3_g = np.percentile(g, 25), np.percentile(g, 75)
    q1_b, q3_b = np.percentile(b, 25), np.percentile(b, 75)

    r_filtered = r[(r >= q1_r) & (r <= q3_r)]
    g_filtered = g[(g >= q1_g) & (g <= q3_g)]
    b_filtered = b[(b >= q1_b) & (b <= q3_b)]

    mean_r = np.mean(r_filtered)
    mean_g = np.mean(g_filtered)
    mean_b = np.mean(b_filtered)

    return mean_r, mean_g, mean_b


def rgb_to_hsv_and_lab(mean_r, mean_g, mean_b):
    rgb_pixel = np.uint8([[[mean_r, mean_g, mean_b]]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2Lab)[0][0]
    v = hsv_pixel[2]  # V from HSV
    s = hsv_pixel[1]  # S from HSV
    b = lab_pixel[2]  # b from Lab
    return v, s, b


def predict_cluster(kmeans_model, vsb_values):
    cluster = kmeans_model.predict([vsb_values])
    return cluster[0]


def load_kmeans_model_for_season(cluster):
    model_path = f'kmeans_model_L2_{cluster}.pkl'
    return joblib.load(model_path)


def predict_detailed_season(cluster_model, vs_values):
    detailed_cluster = cluster_model.predict([vs_values])
    return detailed_cluster[0]


def cluster_to_season(cluster):
    seasons = {0: "봄", 1: "여름", 2: "가을", 3: "겨울"}
    return seasons.get(cluster, "Unknown")


def cluster_to_detailed_season(detailed_cluster, season_type):
    detailed_seasons = {
        0: {0: "라이트 봄", 1: "브라이트 봄"},
        1: {0: "라이트 여름", 1: "브라이트 여름", 2: "뮤트 여름"},
        2: {0: "뮤트 가을", 1: "스트롱 가을", 2: "딥 가을"},
        3: {0: "브라이트 겨울", 1: "딥 겨울"}
    }
    return detailed_seasons.get(season_type, {}).get(detailed_cluster, "Unknown")


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'photo' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['photo']
    image = Image.open(io.BytesIO(image_file.read()))
    sample = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 얼굴 인식 및 피부 영역 추출
    seg_probs = detect_and_parse_face(sample)
    binary_mask = get_face_skin_mask(seg_probs)
    skin_rgb_codes = extract_skin_rgb(sample, binary_mask)
    mean_r, mean_g, mean_b = calculate_quartile_means(skin_rgb_codes)

    # VSb 값 계산
    v, s, b = rgb_to_hsv_and_lab(mean_r, mean_g, mean_b)

    # 1차 K-Means 클러스터 예측 (대분류)
    cluster = predict_cluster(kmeans_model, [v, s, b])
    season = cluster_to_season(cluster)

    # 2차 K-Means 클러스터 예측 (세부 계절) - V와 S만 사용
    cluster_model = load_kmeans_model_for_season(cluster)
    detailed_cluster = predict_detailed_season(cluster_model, [v, s])
    detailed_season = cluster_to_detailed_season(detailed_cluster, cluster)

    image_url = get_image_url_for_season(season, detailed_season)

    return jsonify({
        'mean_rgb': {
            'r': float(mean_r),
            'g': float(mean_g),
            'b': float(mean_b)
        },
        'vsb': {
            'v': float(v),
            's': float(s),
            'b': float(b)
        },
        'season': season,
        'detailed_season': detailed_season,
        'image_url': image_url  # 이미지 URL 추가
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
