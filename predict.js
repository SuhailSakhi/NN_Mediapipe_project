import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

ml5.setBackend("cpu");
const nn = ml5.neuralNetwork({ task: "classification", debug: true });
const modelDetails = {
    model: "model/model.json",
    metadata: "model/model_meta.json",
    weights: "model/model.weights.bin",
};

let modelReady = false;
nn.load(modelDetails, () => {
    console.log("✅ ML5 model geladen");
    modelReady = true;
});

let handLandmarker, drawUtils;
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const photos = [
    "https://picsum.photos/id/1018/600/800",
    "https://picsum.photos/id/1025/600/800",
    "https://picsum.photos/id/1033/600/800",
    "https://picsum.photos/id/1040/600/800",
    "https://picsum.photos/id/1041/600/800",
    "https://picsum.photos/id/1042/600/800",
    "https://picsum.photos/id/1043/600/800",
    "https://picsum.photos/id/1044/600/800",
    "https://picsum.photos/id/1045/600/800",
    "https://picsum.photos/id/1046/600/800",
    "https://picsum.photos/id/1047/600/800",
    "https://picsum.photos/id/1048/600/800",

];

let currentIndex = 0;
let liked = new Array(photos.length).fill(false);
let lastActionTime = 0;
let lastLabel = "";

function renderPhotos() {
    const container = document.getElementById("photoList");
    container.innerHTML = "";
    photos.forEach((src, i) => {
        const wrapper = document.createElement("div");
        wrapper.className = "photo" + (i === currentIndex ? " active" : "") + (liked[i] ? " liked" : "");

        const img = document.createElement("img");
        img.src = src;
        img.style.width = "100%";
        img.style.height = "100%";
        img.style.borderRadius = "16px";
        img.style.objectFit = "cover";

        const heart = document.createElement("div");
        heart.className = "heart";
        heart.innerText = "❤️";

        wrapper.appendChild(img);
        wrapper.appendChild(heart);
        container.appendChild(wrapper);
    });
}

renderPhotos();

async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
    });
    drawUtils = new DrawingUtils(ctx);
}

async function start() {
    await createHandLandmarker();
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.width = video.videoWidth;
        video.height = video.videoHeight;
        requestAnimationFrame(predictLoop);
    });
}

start();

let lastScrollTime = 0;
let lastLikeTime = 0;
const scrollCooldown = 1000;
const likeCooldown = 1500;

async function predictLoop() {
    const results = await handLandmarker.detectForVideo(video, performance.now());

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let hand of results.landmarks) {
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
        drawUtils.drawLandmarks(hand, { color: "#FF0000", radius: 3 });
    }

    const now = Date.now();

    if (modelReady && results.landmarks.length) {
        const flat = results.landmarks[0].map((p) => [p.x, p.y, p.z]).flat();
        const res = await nn.classify(flat);
        const label = res[0].label;

        document.getElementById("prediction").textContent =
            `🔮 ${label} (${(res[0].confidence * 100).toFixed(1)}%)`;

        // continuous scroll
        if (label === "up" && currentIndex > 0 && now - lastScrollTime > scrollCooldown) {
            currentIndex--;
            renderPhotos();
            lastScrollTime = now;
        } else if (label === "down" && currentIndex < photos.length - 1 && now - lastScrollTime > scrollCooldown) {
            currentIndex++;
            renderPhotos();
            lastScrollTime = now;
        }

         if (label === "like" && now - lastLikeTime > likeCooldown) {
            liked[currentIndex] = !liked[currentIndex];
            renderPhotos();
            lastLikeTime = now;
        }
    }

    requestAnimationFrame(predictLoop);
}
