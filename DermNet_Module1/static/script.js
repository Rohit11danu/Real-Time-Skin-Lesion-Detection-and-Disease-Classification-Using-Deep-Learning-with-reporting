// ===============================
// GLOBAL STATE
// ===============================
let currentTab = "upload";
let videoStream = null;
let lastAnalysis = null; // stores { name, age, detections }

// ===============================
// TAB SWITCHING
// ===============================
function showTab(tab) {
  currentTab = tab;

  document.querySelectorAll(".tab").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(div => div.classList.remove("active"));

  document.querySelector(`button[onclick="showTab('${tab}')"]`).classList.add("active");
  document.getElementById(tab).classList.add("active");
}

// ===============================
// IMAGE UPLOAD
// ===============================
document.getElementById("imageInput")?.addEventListener("change", function () {
  const file = this.files[0];
  if (!file) return;

  const img = document.getElementById("previewImage");
  img.src = URL.createObjectURL(file);
  img.style.display = "block";
});

async function analyzeImage() {
  const fileInput = document.getElementById("imageInput");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first.");
    return;
  }

  const name = document.getElementById("name")?.value || "";
  const age = document.getElementById("age")?.value || "";

  const formData = new FormData();
  formData.append("name", name);
  formData.append("age", age);
  formData.append("image", file);

  setReportStatus("Analyzing image...");

  try {
    const res = await fetch("/analyze-image", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");

    renderResult(data);
  } catch (err) {
    console.error(err);
    setReportStatus("Error analyzing image.");
  }
}

// ===============================
// LIVE CAMERA
// ===============================
async function startCamera() {
  const video = document.getElementById("video");

  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = videoStream;
  } catch (e) {
    alert("Camera access denied.");
  }
}

function captureImage() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  alert("Image captured. Click Analyze.");
}

async function analyzeCamera() {
  const canvas = document.getElementById("canvas");
  const name = document.getElementById("name")?.value || "";
  const age = document.getElementById("age")?.value || "";

  canvas.toBlob(async (blob) => {
    if (!blob) {
      alert("Capture an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("age", age);
    formData.append("image", blob, "camera.jpg");

    setReportStatus("Analyzing camera image...");

    try {
      const res = await fetch("/analyze-image", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Analysis failed");

      renderResult(data);
    } catch (err) {
      console.error(err);
      setReportStatus("Error analyzing camera image.");
    }
  }, "image/jpeg");
}

// ===============================
// RENDER RESULTS
// ===============================
function renderResult(data) {
  const report = document.getElementById("reportContent");
  const aiDiv = document.getElementById("aiReport");

  const dets = data.detections || [];

  // store for AI report
  lastAnalysis = {
    name: data.name || "",
    age: data.age || "",
    detections: dets
  };

  if (aiDiv) aiDiv.textContent = "";

  if (dets.length === 0) {
    report.innerHTML = `
      <p><strong>Status:</strong> No lesions detected</p>
      <p><strong>Condition:</strong> None</p>
      <p><strong>Confidence:</strong> --</p>
      <p><strong>Summary:</strong> No visible skin lesions were found.</p>
    `;
    return;
  }

  const labels = dets.map(d => d.label).join(", ");
  const avgConf = (
    dets.reduce((s, d) => s + (d.det_conf || 0), 0) / dets.length
  ).toFixed(2);

  report.innerHTML = `
    <p><strong>Status:</strong> Lesions detected</p>
    <p><strong>Condition:</strong> ${labels}</p>
    <p><strong>Confidence:</strong> ${avgConf}</p>
    <p><strong>Summary:</strong> ${dets.length} suspicious skin regions detected.</p>
  `;
}

function setReportStatus(text) {
  const report = document.getElementById("reportContent");
  report.innerHTML = `<p><strong>Status:</strong> ${text}</p>`;
}

// ===============================
// AI REPORT
// ===============================
async function generateReport() {
  const aiDiv = document.getElementById("aiReport");

  if (!lastAnalysis) {
    alert("Please analyze an image first.");
    return;
  }

  aiDiv.textContent = "Generating AI medical report... Please wait.";

  try {
    const res = await fetch("/generate-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastAnalysis)
    });

    const data = await res.json();

    if (!res.ok) {
      aiDiv.textContent = data.error || "Failed to generate report.";
      return;
    }

    aiDiv.textContent = data.report;
  } catch (err) {
    console.error(err);
    aiDiv.textContent = "Server connection failed.";
  }
}
static/script.js
// Works with your templates/index.html IDs:
// name, age, imageInput, previewImage, reportContent, video, canvas

function showTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(el => el.classList.remove("active"));

  document.getElementById(tabId).classList.add("active");

  const btns = document.querySelectorAll(".tab");
  if (tabId === "upload") btns[0].classList.add("active");
  if (tabId === "camera") btns[1].classList.add("active");
}

function setReport(status, condition, confidence, summary) {
  const report = document.getElementById("reportContent");
  report.innerHTML = `
    <p><strong>Status:</strong> ${status}</p>
    <p><strong>Condition:</strong> ${condition}</p>
    <p><strong>Confidence:</strong> ${confidence}</p>
    <p><strong>Summary:</strong> ${summary}</p>
  `;
}

function getNameAgeOrAlert() {
  const name = document.getElementById("name")?.value?.trim();
  const age = document.getElementById("age")?.value?.trim();

  if (!name || !age) {
    alert("Please enter Name and Age first.");
    return null;
  }
  return { name, age };
}

// Preview image on selection
document.addEventListener("DOMContentLoaded", () => {
  const imgInput = document.getElementById("imageInput");
  const preview = document.getElementById("previewImage");

  if (imgInput && preview) {
    imgInput.addEventListener("change", () => {
      const file = imgInput.files?.[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
    });
  }
});

// -------- Upload Image --------
async function analyzeImage() {
  const meta = getNameAgeOrAlert();
  if (!meta) return;

  const imgInput = document.getElementById("imageInput");
  const file = imgInput?.files?.[0];

  if (!file) {
    alert("Please choose an image first.");
    return;
  }

  setReport("Analyzing image...", "--", "--", "Please wait...");

  const formData = new FormData();
  formData.append("name", meta.name);
  formData.append("age", meta.age);
  formData.append("image", file);

  try {
    const res = await fetch("/analyze-image", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      setReport("Error", "--", "--", data.error || "Backend error");
      return;
    }

    renderResult(data, "Upload image analyzed.");
  } catch (e) {
    console.error(e);
    setReport("Error", "--", "--", "Failed to connect to server.");
  }
}

// -------- Live Camera --------
let stream = null;

async function startCamera() {
  const video = document.getElementById("video");
  if (!video) return;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    setReport("Camera started ✅", "--", "--", "Click Capture, then Analyze.");
  } catch (e) {
    console.error(e);
    setReport("Error", "--", "--", "Camera permission denied or not available.");
    alert("Camera permission denied or not available.");
  }
}

function captureImage() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  if (!video || !canvas) return;

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  setReport("Captured ✅", "--", "--", "Now click Analyze to process this frame.");
}

async function analyzeCamera() {
  const meta = getNameAgeOrAlert();
  if (!meta) return;

  const canvas = document.getElementById("canvas");
  if (!canvas) return;

  setReport("Analyzing camera frame...", "--", "--", "Please wait...");

  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.95));
  if (!blob) {
    setReport("Error", "--", "--", "Could not read captured image.");
    return;
  }

  const formData = new FormData();
  formData.append("name", meta.name);
  formData.append("age", meta.age);
  formData.append("image", blob, "camera.jpg");

  try {
    const res = await fetch("/analyze-image", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      setReport("Error", "--", "--", data.error || "Backend error");
      return;
    }

    renderResult(data, "Camera frame analyzed.");
  } catch (e) {
    console.error(e);
    setReport("Error", "--", "--", "Failed to connect to server.");
  }
}

// -------- Render backend result into UI --------
function renderResult(data, footerText) {
  const dets = data.detections || [];
  const n = data.num_detections ?? dets.length;

  if (n === 0) {
    setReport("Done ✅", "No lesion detected", "--",
      `${footerText} If you still have symptoms, consult a dermatologist.`
    );
    return;
  }

  // pick best detection by label_conf if available, else first
  let best = dets[0];
  for (const d of dets) {
    if ((d.label_conf ?? 0) > (best.label_conf ?? 0)) best = d;
  }

  const label = best.label || "Unknown";
  const confPct = best.label_conf != null ? (best.label_conf * 100).toFixed(1) + "%" : "--";

  setReport(
    "Done ✅",
    `${label} (detections: ${n})`,
    confPct,
    `${footerText} This is not a medical diagnosis. If it worsens or persists, see a dermatologist.`
  );
}


   
