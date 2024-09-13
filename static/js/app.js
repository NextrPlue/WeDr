function previewAndResizeImage(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(event) {
        const img = new Image();
        img.src = event.target.result;
        img.onload = function() {
            const canvas = document.createElement('canvas');
            const maxSize = 1024; // 최대 크기 제한

            let width = img.width;
            let height = img.height;

            // 비율에 맞게 크기 조정
            if (width > height) {
                if (width > maxSize) {
                    height *= maxSize / width;
                    width = maxSize;
                }
            } else {
                if (height > maxSize) {
                    width *= maxSize / height;
                    height = maxSize;
                }
            }

            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);

            // 미리보기 이미지 업데이트
            document.getElementById('previewImage').src = canvas.toDataURL('image/jpeg');

            // 리사이즈된 이미지를 서버로 전송할 준비
            canvas.toBlob(function(blob) {
                resizedImageBlob = blob;
            }, 'image/jpeg', 0.95);
        }
    };

    reader.readAsDataURL(file);
}

let messageInterval;
let isAnalyzing = true;
let shareImageUrl;

function submitPhoto() {
    if (!resizedImageBlob) {
        alert("이미지를 선택해 주세요.");
        return;
    }

    const formData = new FormData();
    formData.append('photo', resizedImageBlob, 'resized-photo.jpg');

    const submitButton = document.getElementById('submitButton');
    submitButton.classList.add('loading');
    submitButton.querySelector('.button-text').style.display = 'none';  // 텍스트 숨기기
    submitButton.querySelector('.loading-indicator').style.display = 'inline-flex';  // 로딩 표시

    startMessageCycle();  // 메시지 전환 시작

    fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            localStorage.setItem('season', result.season);
            localStorage.setItem('detailed_season', result.detailed_season);
            localStorage.setItem('image_url', result.image_url);
            window.location.href = "/result";
        })
        .catch(error => {
            console.error('Error:', error);
            alert('사진 분석 중 오류가 발생했습니다.');
        })
        .finally(() => {
            stopMessageCycle();  // 메시지 전환 중지
            submitButton.classList.remove('loading');
            submitButton.querySelector('.button-text').style.display = 'inline-block';  // 텍스트 복원
            submitButton.querySelector('.loading-indicator').style.display = 'none';  // 로딩 숨기기
        });
}

function startMessageCycle() {
    const loadingText = document.querySelector('.loading-text');
    let isAnalyzing = true;  // 메시지 상태를 저장하는 변수

    messageInterval = setInterval(() => {
        if (isAnalyzing) {
            loadingText.innerText = "3~5분 소요됩니다\u00a0";
        } else {
            loadingText.innerText = "분석 중\u00a0";
        }
        isAnalyzing = !isAnalyzing;  // 메시지를 교체
    }, 5000);  // 5초마다 메시지 교체
}

function stopMessageCycle() {
    clearInterval(messageInterval);
}

function startTest() {
    window.location.href = "/second";
}

function retakeTest() {
    window.location.href = "/";
}

function loadResults() {
    const season = localStorage.getItem('season');
    const detailed_season = localStorage.getItem('detailed_season');
    const imageUrl = localStorage.getItem('image_url');
    shareImageUrl = imageUrl;

    const resultPersonalcolor = document.getElementById('result-personal');
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';  // 기존 내용을 지우고 새로운 결과를 표시

    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';

    // 이미지 추가
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = `${season} ${detailed_season}`;
    resultItem.appendChild(img);

    // 텍스트 추가
    const text = document.createElement('h2');
    text.innerText = `${detailed_season}`;
    resultPersonalcolor.appendChild(text);

    resultsContainer.appendChild(resultItem);
}

function goBack() {
    window.history.back();
}

// Kakao SDK 로드 및 초기화 함수
function loadKakaoSDK(callback) {
    if (window.Kakao && Kakao.isInitialized()) {
        callback();
    } else {
        var script = document.createElement('script');
        script.onload = function () {
            Kakao.init('4481fb28d946215027263817405d3c9c'); // 사용하려는 앱의 JavaScript 키 입력
            callback();
        };
        script.onerror = function () {
            console.error('Failed to load Kakao SDK.');
        };
        script.src = "https://t1.kakaocdn.net/kakao_js_sdk/2.7.2/kakao.min.js";
        script.crossOrigin = "anonymous";
        script.integrity = "sha384-TiCUE00h649CAMonG018J2ujOgDKW/kVWlChEuu4jK2vxfAAD0eZxzCKakxg55G4";
        document.head.appendChild(script);
    }
}

// 공유 버튼 이벤트 설정 함수
function setupShareButtons() {
    // 카카오톡 공유 버튼 클릭 이벤트 설정
    document.getElementById('kakao-share').addEventListener('click', () => {
        increaseShareCount();

        Kakao.Share.sendDefault({
            objectType: 'feed',
            content: {
                title: '길거리 음식 퍼스널컬러 테스트',
                description: '내 사진으로 퍼스널컬러를 간단하게 알아보세요!',
                imageUrl: shareImageUrl,
                link: {
                    mobileWebUrl: 'http://we-dr.com',
                    webUrl: 'http://we-dr.com',
                },
            },
            buttons: [
                {
                    title: '테스트 하러 가기',
                    link: {
                        mobileWebUrl: 'http://we-dr.com',
                        webUrl: 'http://we-dr.com',
                    },
                },
            ],
        });
    });

    // 페이스북 공유 버튼 클릭 이벤트 설정
    document.getElementById('facebook-share').addEventListener('click', () => {
        increaseShareCount();
        window.open("http://www.facebook.com/sharer/sharer.php?u=we-dr.com/")
    });

    // X 공유 버튼 클릭 이벤트 설정
    document.getElementById('x-share').addEventListener('click', () => {
        increaseShareCount();
        window.open("https://x.com/intent/post?text=길거리 음식 퍼스널컬러 테스트 - 내 사진으로 퍼스널컬러를 간단하게 알아보세요!&url=we-dr.com&hashtags=WeDr, 퍼스널컬러, 테스트")
    });

    // 링크 복사 버튼 클릭 이벤트 설정
    document.getElementById('copy-link').addEventListener('click', () => {
        increaseShareCount();
        if (window.location.pathname.endsWith('/result')) {
            copyToClipboard(shareImageUrl);
        } else {
            copyToClipboard(window.location.href);
        }

        alert("링크가 복사되었습니다!");
    });
}


document.addEventListener('DOMContentLoaded', (event) => {
    loadKakaoSDK(setupShareButtons);

    if (window.location.pathname.endsWith('/result')) {
        loadResults();
    } else {
        shareImageUrl = 'http://we-dr.com/static/images/placeholder.png';
    }
});

function startTest() {
    window.location.href = "/second";
}

// 공유 수 증가
function increaseShareCount() {
    fetch('/increase-share', {
        method: 'POST',
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('share-count').textContent = data.shares;
        })
        .catch(error => {
            console.error('Error increasing share count:', error);
        });
}

// 링크 복사하기 기능
function copyToClipboard(text) {
    const tempInput = document.createElement('input');
    document.body.appendChild(tempInput);
    tempInput.value = text;
    tempInput.select();
    document.execCommand('copy');
    document.body.removeChild(tempInput);
}
