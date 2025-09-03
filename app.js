// ====== UI ======
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');

const btnStart = document.getElementById('btnStart');
const btnStop  = document.getElementById('btnStop');
const selRes   = document.getElementById('selRes');

const sBlink = document.getElementById('sBlink');
const sBrow  = document.getElementById('sBrow');
const sMouth = document.getElementById('sMouth');
const lblBlink = document.getElementById('lblBlink');
const lblBrow  = document.getElementById('lblBrow');
const lblMouth = document.getElementById('lblMouth');

const statusEl = document.getElementById('status');
const calibAlert = document.getElementById('calibAlert');
const logEl = document.getElementById('log');

const blinkCountEl = document.getElementById('blinkCount');
const browCountEl  = document.getElementById('browCount');
const mouthCountEl = document.getElementById('mouthCount');

function log(s){ const t=new Date().toLocaleTimeString(); logEl.textContent = `[${t}] ${s}\n`+logEl.textContent; }
function setStatus(s){ statusEl.textContent = `Estado: ${s}`; }
function updLabels(){ lblBlink.textContent=(sBlink.value/100).toFixed(2); lblBrow.textContent=(sBrow.value/100).toFixed(2); lblMouth.textContent=(sMouth.value/100).toFixed(2); }
[sBlink,sBrow,sMouth].forEach(e=>e.addEventListener('input',updLabels)); updLabels();

// ====== Estado general ======
let stream=null, running=false, rafSend=null, faceMesh=null;

// OpenCV buffers
let cap=null, frame=null, gray=null, clahe=null;

// contadores UI
let blinkCount=0, browCount=0, mouthCount=0;

// EMA (suavizado)
const ema = { ear: null, brow: null, mouth: null };
const EMA_A = 0.35; // alpha

// tiempos y estados
const now = () => performance.now();

// ---- Parpadeo (state machine) ----
let eyeState = 'OPEN';           // OPEN | CLOSED
let eyeClosedAt = 0;
let eyeOpenedAt = 0;
let lastBlinkAt = 0;
const BLINK_MIN_MS = 80;
const BLINK_MAX_MS = 300;
const BLINK_REFRACT_MS = 250;
const EYE_MIN_OPEN_MS = 140;

// ---- Boca (state machine) ----
let mouthState = 'CLOSED';       // CLOSED | OPEN
let mouthOpenedAt = 0;
let mouthClosedAt = 0;
let lastMouthAt = 0;
const MOUTH_OPEN_MIN_MS = 300;
const MOUTH_CLOSE_MIN_MS = 250;
const MOUTH_REFRACT_MS = 280;

// ---- Cejas (peak detection) ----
let browAbove = false;
let browStartAt = 0;
let browPeak = 0;
let lastBrowAt = 0;
const BROW_MIN_HOLD_MS = 120;
const BROW_REFRACT_MS = 380;

// calibración
let calibrating=false, calibSamples=[], calib={earBase:0,browBase:0,mouthBase:0};

// landmarks
const LEFT  = {EYE:[33,160,158,133,153,144], BROW_TOP:65,  IRIS:468};
const RIGHT = {EYE:[362,385,387,263,373,380], BROW_TOP:295, IRIS:473};
const MOUTH_UP=13, MOUTH_LO=14, LFACE=234, RFACE=454;

// ====== OpenCV init ======
if (typeof cv !== 'undefined'){
  cv.onRuntimeInitialized = () => { log('OpenCV.js listo'); };
}

// ====== MediaPipe ======
function setupFaceMesh(){
  if (typeof FaceMesh === 'undefined') throw new Error('MediaPipe FaceMesh no cargó');
  faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
  faceMesh.setOptions({ maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.6, minTrackingConfidence:0.6 });
  faceMesh.onResults(onResults);
}

function d2(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }
function ear(lmk, ids){ const p=i=>lmk[i]; const v1=d2(p(ids[1]),p(ids[5])), v2=d2(p(ids[2]),p(ids[4])), h=d2(p(ids[0]),p(ids[3])); return (v1+v2)/(2*h+1e-6); }

// ====== Calibración ======
function startCalibration(){
  calibrating=true; calibSamples=[]; calib={earBase:0,browBase:0,mouthBase:0};
  calibAlert.classList.remove('d-none');
  setTimeout(()=>{
    calibrating=false; calibAlert.classList.add('d-none');
    const avg=a=>a.reduce((s,v)=>s+v,0)/Math.max(1,a.length);
    calib.earBase  = avg(calibSamples.map(s=>s.ear));
    calib.browBase = avg(calibSamples.map(s=>s.brow));
    calib.mouthBase= avg(calibSamples.map(s=>s.mouth));
    log(`Calibración → EAR:${calib.earBase.toFixed(3)} | Brow:${calib.browBase.toFixed(3)} | Mouth:${calib.mouthBase.toFixed(3)}`);
  },2000);
}

// ====== Dibujo ======
function drawOverlay(lmk){
  const w=overlay.width, h=overlay.height;
  octx.clearRect(0,0,w,h);
  const pts=[...LEFT.EYE,...RIGHT.EYE,LEFT.BROW_TOP,RIGHT.BROW_TOP,LEFT.IRIS,RIGHT.IRIS,MOUTH_UP,MOUTH_LO,LFACE,RFACE];
  pts.forEach(i=>{ const x=lmk[i].x*w, y=lmk[i].y*h; octx.beginPath(); octx.arc(x,y,2,0,Math.PI*2); octx.fillStyle='#00e5ff'; octx.fill(); });
  function connect(ids,c){ octx.beginPath(); ids.forEach((i,k)=>{ const x=lmk[i].x*w,y=lmk[i].y*h; if(!k)octx.moveTo(x,y); else octx.lineTo(x,y); }); octx.closePath(); octx.strokeStyle=c; octx.lineWidth=2; octx.stroke(); }
  connect(LEFT.EYE,'#15ff72'); connect(RIGHT.EYE,'#15ff72');
}

// ====== OpenCV helpers (ROIs) ======
function roiFromLandmarksRect(lmk, a, b, pad, W, H){
  const x1=Math.min(a.x,b.x), y1=Math.min(a.y,b.y), x2=Math.max(a.x,b.x), y2=Math.max(a.y,b.y);
  const cx=(x1+x2)/2, cy=(y1+y2)/2, w=(x2-x1)*(1+pad), h=(y2-y1)*(1+pad);
  const rx=Math.max(0, (cx-w/2)*W)|0, ry=Math.max(0, (cy-h/2)*H)|0;
  const rw=Math.min(W-rx, w*W)|0, rh=Math.min(H-ry, h*H)|0;
  return new cv.Rect(rx, ry, Math.max(2,rw), Math.max(2,rh));
}
function darkRatio(mat){ const bin=new cv.Mat(); cv.threshold(mat,bin,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU); const d=cv.countNonZero(bin)/(bin.rows*bin.cols); bin.delete(); return d; }
function edgeDensity(mat){ const ed=new cv.Mat(); cv.Canny(mat,ed,60,120); const d=cv.countNonZero(ed)/(ed.rows*ed.cols); ed.delete(); return d; }

// ====== EMA ======
function smooth(key, val){
  if (ema[key]==null) { ema[key]=val; return val; }
  ema[key] = EMA_A*val + (1-EMA_A)*ema[key];
  return ema[key];
}

// ====== Resultados por frame ======
function onResults(r){
  if(!running) return;
  const faces=r.multiFaceLandmarks; if(!faces||!faces.length){ octx.clearRect(0,0,overlay.width,overlay.height); return; }
  const lmk=faces[0];
  drawOverlay(lmk);

  const W=video.videoWidth, H=video.videoHeight;
  if(!cap && typeof cv!=='undefined' && cv.Mat){
    cap = new cv.VideoCapture(video);
    frame = new cv.Mat(H, W, cv.CV_8UC4);
    gray  = new cv.Mat(H, W, cv.CV_8UC1);
    clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
  }
  // Refuerzo OpenCV
  try{ cap.read(frame); cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY); clahe.apply(gray, gray); }catch(_){}

  const faceW = d2(lmk[LFACE], lmk[RFACE]) + 1e-6;

  // Métricas base
  let earAvg = (ear(lmk, LEFT.EYE) + ear(lmk, RIGHT.EYE))/2;
  let browL = d2(lmk[LEFT.BROW_TOP],  lmk[LEFT.IRIS])  / faceW;
  let browR = d2(lmk[RIGHT.BROW_TOP], lmk[RIGHT.IRIS]) / faceW;
  let browDist = (browL + browR)/2;
  let mouthOpen = d2(lmk[MOUTH_UP], lmk[MOUTH_LO]) / faceW;

  // ROIs OpenCV
  if (gray){
    // Ojos: oscuridad media (si es alta, refuerza “cierre” y penaliza cejas/boca)
    const eyeRectL = roiFromLandmarksRect(lmk, lmk[LEFT.EYE[0]],  lmk[LEFT.EYE[3]], 0.6, W, H);
    const eyeRectR = roiFromLandmarksRect(lmk, lmk[RIGHT.EYE[3]], lmk[RIGHT.EYE[0]], 0.6, W, H);
    const eyeL = gray.roi(eyeRectL), eyeR = gray.roi(eyeRectR);
    const eyeDark = (darkRatio(eyeL) + darkRatio(eyeR))/2; eyeL.delete(); eyeR.delete();
    if (eyeDark > 0.40) { browDist *= 0.98; }
    if (eyeDark > 0.55) { mouthOpen *= 0.98; }

    // Cejas: densidad de bordes
    const browRectL = roiFromLandmarksRect(lmk, lmk[LEFT.BROW_TOP],  lmk[LEFT.IRIS],  1.2, W, H);
    const browRectR = roiFromLandmarksRect(lmk, lmk[RIGHT.BROW_TOP], lmk[RIGHT.IRIS], 1.2, W, H);
    const browLROI = gray.roi(browRectL), browRROI = gray.roi(browRectR);
    const browEdge = (edgeDensity(browLROI) + edgeDensity(browRROI))/2;
    browDist *= (1 + Math.min(0.08, browEdge*0.4));
    browLROI.delete(); browRROI.delete();

    // Boca: oscuridad interna
    const mouthRect = roiFromLandmarksRect(lmk, lmk[MOUTH_UP], lmk[MOUTH_LO], 1.6, W, H);
    const mouthROI = gray.roi(mouthRect);
    const mouthDark = darkRatio(mouthROI);
    mouthOpen *= (1 + Math.min(0.18, mouthDark*0.55));
    mouthROI.delete();
  }

  // Suavizado EMA
  earAvg   = smooth('ear',   earAvg);
  browDist = smooth('brow',  browDist);
  mouthOpen= smooth('mouth', mouthOpen);

  if (calibrating){ calibSamples.push({ ear:earAvg, brow:browDist, mouth:mouthOpen }); return; }

  // ====== UMBRALES ======
  const kClose = sBlink.value/100;                 // p.ej. 0.80
  const kOpen  = Math.min(kClose + 0.10, 0.97);    // histéresis
  const thClose= calib.earBase * kClose;
  const thOpen = calib.earBase * kOpen;

  const kBrow  = sBrow.value/100;                  // p.ej. 1.30
  const thBrow = calib.browBase * kBrow;

  const kMouth = sMouth.value/100;                 // p.ej. 1.60
  const thMouth= calib.mouthBase * kMouth;

  const t = now();

  // ====== Parpadeo (solo si duración dentro del rango) ======
  if (eyeState === 'OPEN' && earAvg < thClose) {
    // cerrar
    if (t - eyeOpenedAt >= EYE_MIN_OPEN_MS) { eyeState = 'CLOSED'; eyeClosedAt = t; }
  } else if (eyeState === 'CLOSED' && earAvg > thOpen) {
    const dur = t - eyeClosedAt;
    if (dur >= BLINK_MIN_MS && dur <= BLINK_MAX_MS && (t - lastBlinkAt) > BLINK_REFRACT_MS) {
      blinkCount++; blinkCountEl.textContent = String(blinkCount);
      lastBlinkAt = t; log(`Parpadeo (${Math.round(dur)} ms)`);
    }
    eyeState = 'OPEN'; eyeOpenedAt = t;
    // limpiar contadores auxiliares
  }

  // ====== Cejas (pico con caída y hold) ======
  if (!browAbove && browDist > thBrow) {
    browAbove = true; browStartAt = t; browPeak = browDist;
  }
  if (browAbove) {
    browPeak = Math.max(browPeak, browDist);
    const dropped = browDist < thBrow * 0.92; // exige caída
    const held = (t - browStartAt) >= BROW_MIN_HOLD_MS;
    if (dropped && held && (t - lastBrowAt) > BROW_REFRACT_MS) {
      browCount++; browCountEl.textContent = String(browCount);
      lastBrowAt = t; log('Cejas ↑');
      browAbove = false; browPeak = 0;
    }
    if (dropped && !held) { browAbove = false; browPeak = 0; } // cancelar falsos picos breves
  }

  // ====== Boca (open ≥300ms -> close ≥250ms) ======
  if (mouthState === 'CLOSED' && mouthOpen > thMouth) {
    mouthState = 'OPEN'; mouthOpenedAt = t; mouthCloseFrames = 0;
  } else if (mouthState === 'OPEN') {
    if (mouthOpen > thMouth) {
      // sigue abierta
    } else {
      // empezó a cerrar
      if (t - mouthOpenedAt >= MOUTH_OPEN_MIN_MS) {
        // cuenta solo cuando además sostiene el cierre mínimo
        if (!mouthClosedAt) mouthClosedAt = t;
        if ((t - mouthClosedAt) >= MOUTH_CLOSE_MIN_MS && (t - lastMouthAt) > MOUTH_REFRACT_MS) {
          mouthCount++; mouthCountEl.textContent = String(mouthCount);
          lastMouthAt = t; log('Boca ↑');
          mouthState = 'CLOSED'; mouthClosedAt = 0;
        }
      } else {
        // fue un “amago” corto → cancelar
        mouthState = 'CLOSED'; mouthClosedAt = 0;
      }
    }
  }
}

// ====== Cámara ======
async function startCamera(){
  if (running) return;

  // reset UI
  blinkCount=browCount=mouthCount=0;
  blinkCountEl.textContent=browCountEl.textContent=mouthCountEl.textContent='0';
  eyeState='OPEN'; eyeOpenedAt=now();
  mouthState='CLOSED'; mouthClosedAt=now();
  browAbove=false; browPeak=0;

  setStatus('solicitando permisos…');
  try{
    const [w,h] = selRes.value.split('x').map(Number);
    const secure = location.protocol==='https:' || ['localhost','127.0.0.1'].includes(location.hostname);
    if(!secure) alert('Sirve el sitio en https:// o localhost.');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width:{ideal:w}, height:{ideal:h}, facingMode:'user' },
      audio:false
    });
  }catch(err){
    setStatus('error de cámara'); log('ERROR cámara: '+err.name+' - '+err.message);
    alert('No se pudo acceder a la cámara: '+err.message); return;
  }

  video.srcObject=stream; video.muted=true; video.playsInline=true;
  await video.play();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;

  // start libs
  try{ if(!faceMesh) setupFaceMesh(); }catch(e){ setStatus('error de procesamiento'); log('ERROR MediaPipe: '+e.message); return; }
  startCalibration();

  running=true; btnStart.disabled=true; btnStop.disabled=false; setStatus('en ejecución'); log('Cámara encendida');

  // bucle de envío a MediaPipe
  const send = async () => { if (!running) return; try { await faceMesh.send({ image: video }); } catch(_){ } rafSend = requestAnimationFrame(send); };
  send();
}

function stopCamera(){
  if(!running && !stream) return;
  try{ cancelAnimationFrame(rafSend); }catch(_){}
  try{ if(stream) stream.getTracks().forEach(t=>t.stop()); }catch(_){}
  try{ video.pause(); video.srcObject=null; }catch(_){}
  try{
    if (frame && !frame.isDeleted()) frame.delete();
    if (gray && !gray.isDeleted()) gray.delete();
    if (clahe) clahe.delete();
  }catch(_){}
  frame=gray=clahe=null; cap=null;

  running=false; stream=null; btnStart.disabled=false; btnStop.disabled=true;
  octx.clearRect(0,0,overlay.width,overlay.height);
  setStatus('detenido'); log('Cámara apagada y recursos liberados');
}

// eventos
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);
