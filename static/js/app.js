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

document.addEventListener('DOMContentLoaded', (event) => {
    if (window.location.pathname.endsWith('/result')) {
        loadResults();
    }
});

function goBack() {
    window.history.back();
}
