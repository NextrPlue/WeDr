from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import cv2
import dlib
from PIL import Image
import joblib
import io
import os

app = Flask(__name__)
CORS(app)

# SQLite 데이터베이스 설정
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'site_stats.db')
db = SQLAlchemy(app)

# K-Means 모델 로드
base_model_path = 'kmeans_model_L2.pkl'
kmeans_model = joblib.load(base_model_path)

# dlib의 얼굴 랜드마크 예측기 로드
predictor_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(predictor_path):
    raise FileNotFoundError("Facial landmark predictor data file not found. Please download 'shape_predictor_68_face_landmarks.dat'.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

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

def get_face_landmarks(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return None
    # For simplicity, take the first detected face
    face = faces[0]
    # Get the landmarks/parts for the face
    landmarks = predictor(gray, face)
    # Convert landmarks to a NumPy array
    landmark_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
    return np.array(landmark_points, np.int32)

def create_skin_mask(image, landmarks):
    # Define regions to exclude (eyes, eyebrows, lips)
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))
    eyebrow_indices = list(range(17, 27))
    mouth_indices = list(range(48, 61))

    # Create an all-white mask
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # Exclude eyes, eyebrows, and mouth by filling them with black color on the mask
    cv2.fillPoly(mask, [landmarks[left_eye_indices]], 0)
    cv2.fillPoly(mask, [landmarks[right_eye_indices]], 0)
    cv2.fillPoly(mask, [landmarks[eyebrow_indices]], 0)
    cv2.fillPoly(mask, [landmarks[mouth_indices]], 0)

    # Create face convex hull to include the facial region
    face_indices = list(range(0, 17)) + list(range(17, 27)) + list(range(27, 36)) + list(range(48, 68))
    face_hull = cv2.convexHull(landmarks[face_indices])
    # Fill outside of face convex hull with black color
    face_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(face_mask, face_hull, 255)
    # Combine face mask with the exclusion mask
    skin_mask = cv2.bitwise_and(mask, face_mask)
    return skin_mask

def detect_and_parse_face(sample):
    landmarks = get_face_landmarks(sample)
    if landmarks is None:
        return None, None
    skin_mask = create_skin_mask(sample, landmarks)
    return sample, skin_mask

def extract_skin_rgb(image, skin_mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img[skin_mask > 0]

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


# 데이터베이스 모델 정의
class SiteStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    visitors = db.Column(db.Integer, default=0)
    shares = db.Column(db.Integer, default=0)

def initialize_database():
    if SiteStats.query.count() == 0:
        # 방문자 수 및 공유 수 데이터가 없을 경우 초기화
        initial_stats = SiteStats(visitors=0, shares=0)
        db.session.add(initial_stats)
        db.session.commit()

# 데이터베이스 초기화
@app.before_request
def before_request():
    with app.app_context():
        initialize_database()

@app.route('/')
def index():
    stats = SiteStats.query.first()
    visitors = stats.visitors
    shares = stats.shares

    # 쿠키를 확인하여 새로운 방문자인지 판단
    if not request.cookies.get('visited'):
        # 방문자 수 업데이트
        stats.visitors += 1
        db.session.commit()

        # 쿠키 설정
        response = make_response(render_template('index.html', visitors=visitors, shares=shares))
        response.set_cookie('visited', 'true', max_age=60*60*24*30)  # 30일 동안 유지
        return response
    else:
        # 기존 방문자는 방문자 수 증가 없이 그대로 반환
        return render_template('index.html', visitors=visitors, shares=shares)


@app.route('/second')
def second_page():
    return render_template('second-page.html')

@app.route('/result')
def result_page():
    stats = SiteStats.query.first()
    shares = stats.shares

    return render_template('result-page.html', shares=shares)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'photo' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['photo']
    image = Image.open(io.BytesIO(image_file.read()))
    sample = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 얼굴 인식 및 피부 영역 추출
    face_image, skin_mask = detect_and_parse_face(sample)
    if face_image is None or skin_mask is None:
        return jsonify({'error': 'No face detected'}), 400

    skin_rgb_codes = extract_skin_rgb(face_image, skin_mask)
    if skin_rgb_codes.size == 0:
        return jsonify({'error': 'No skin detected'}), 400

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

@app.route('/increase-share', methods=['POST'])
def increase_share():
    stats = SiteStats.query.first()
    stats.shares += 1
    db.session.commit()
    return jsonify({'shares': stats.shares})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 데이터베이스 테이블 생성
    app.run(host='0.0.0.0', port=5000, debug=True)
