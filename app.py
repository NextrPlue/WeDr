from flask import Flask, request, jsonify, render_template
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
            "계란빵 - 봄 라이트": f"{base_url}/static/images/spring_light.png",
            "초당옥수수 - 봄 브라이트": f"{base_url}/static/images/spring_bright.png"
        },
        "여름": {
            "수박주스 - 여름 라이트": f"{base_url}/static/images/summer_light.png",
            "솜사탕 - 여름 브라이트": f"{base_url}/static/images/summer_bright.png",
            "콩국수 - 여름 뮤트": f"{base_url}/static/images/summer_muted.png"
        },
        "가을": {
            "카스테라 - 가을 뮤트": f"{base_url}/static/images/fall_muted.png",
            "군밤 - 가을 스트롱": f"{base_url}/static/images/fall_strong.png",
            "팥죽 - 가을 딥": f"{base_url}/static/images/fall_deep.png"
        },
        "겨울": {
            "블사탕 - 겨울 브라이트": f"{base_url}/static/images/winter_bright.png",
            "밤양갱 - 겨울 딥": f"{base_url}/static/images/winter_deep.png"
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
    binary_mask_resized = cv2.resize(binary_mask, (sample.shape[1], sample.shape[0]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    return img[binary_mask_resized == 1]

def calculate_quartile_means(rgb_codes):
    r = np.sort(rgb_codes[:, 0])
    g = np.sort(rgb_codes[:, 1])
    b = np.sort(rgb_codes[:, 2])

    r_filtered = r[(r >= np.percentile(r, 25)) & (r <= np.percentile(r, 75))]
    g_filtered = g[(g >= np.percentile(g, 25)) & (g <= np.percentile(g, 75))]
    b_filtered = b[(b >= np.percentile(b, 25)) & (b <= np.percentile(b, 75))]

    return np.mean(r_filtered), np.mean(g_filtered), np.mean(b_filtered)

def rgb_to_hsv_and_lab(mean_r, mean_g, mean_b):
    rgb_pixel = np.uint8([[[mean_r, mean_g, mean_b]]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2Lab)[0][0]
    return hsv_pixel[2], hsv_pixel[1], lab_pixel[2]  # V, S, b

def predict_cluster(kmeans_model, vsb_values):
    return kmeans_model.predict([vsb_values])[0]

def load_kmeans_model_for_season(cluster):
    model_path = f'kmeans_model_L2_{cluster}.pkl'
    return joblib.load(model_path)

def predict_detailed_season(cluster_model, vs_values):
    return cluster_model.predict([vs_values])[0]

def cluster_to_season(cluster):
    seasons = {0: "봄", 1: "여름", 2: "가을", 3: "겨울"}
    return seasons.get(cluster, "Unknown")

def cluster_to_detailed_season(detailed_cluster, season_type):
    detailed_seasons = {
        0: {0: "계란빵 - 봄 라이트", 1: "초당옥수수 - 봄 브라이트"},
        1: {0: "수박주스 - 여름 라이트", 1: "솜사탕 - 여름 브라이트", 2: "콩국수 - 여름 뮤트"},
        2: {0: "카스테라 - 가을 뮤트", 1: "군밤 - 가을 스트롱", 2: "팥죽 - 가을 딥"},
        3: {0: "블사탕 - 겨울 브라이트", 1: "밤양갱 - 겨울 딥"}
    }
    return detailed_seasons.get(season_type, {}).get(detailed_cluster, "Unknown")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second_page():
    return render_template('second-page.html')

@app.route('/result')
def result_page():
    return render_template('result-page.html')

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
        'mean_rgb': {'r': float(mean_r), 'g': float(mean_g), 'b': float(mean_b)},
        'vsb': {'v': float(v), 's': float(s), 'b': float(b)},
        'season': season,
        'detailed_season': detailed_season,
        'image_url': image_url
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
