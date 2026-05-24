// frontend/src/App.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { getApiBase, getWsBase } from './api';

async function acquireCameraStream() {
  const attempts = [
    { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
    { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
    { video: true, audio: false },
  ];
  let lastError;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      lastError = e;
      if (e.name === 'NotAllowedError' || e.name === 'NotFoundError') {
        throw e;
      }
    }
  }
  throw lastError;
}

function formatCameraError(e) {
  switch (e?.name) {
    case 'NotAllowedError':
      return 'Camera permission denied. Allow camera access and try again.';
    case 'NotFoundError':
      return 'No camera found. Connect a USB camera and try again.';
    case 'NotReadableError':
      return 'Camera is in use by another application. Close it and try again.';
    case 'OverconstrainedError':
      return 'Camera settings are not supported. A simpler mode will be used on retry.';
    case 'AbortError':
      return 'Camera access was interrupted. Please try again.';
    default:
      return e?.message
        ? `Could not access camera: ${e.message}`
        : 'Could not access camera. Check permissions and that no other app is using it.';
  }
}

function App() {
  const [status, setStatus] = useState('idle');
  const [bees, setBees] = useState([]);
  const [videoUrl, setVideoUrl] = useState(null);
  const [processedFrameUrl, setProcessedFrameUrl] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [targetBee, setTargetBee] = useState('');
  const [alertedBees, setAlertedBees] = useState(new Set());
  const [alertMessage, setAlertMessage] = useState('');
  const [inputSource, setInputSource] = useState('video');
  const [activeInputSource, setActiveInputSource] = useState(null);
  const [cameraError, setCameraError] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const waitingForFrameRef = useRef(false);
  const cameraActiveRef = useRef(false);
  const processedUrlRef = useRef(null);

  const revokeProcessedUrl = useCallback(() => {
    if (processedUrlRef.current) {
      URL.revokeObjectURL(processedUrlRef.current);
      processedUrlRef.current = null;
    }
  }, []);

  const applyBeeResults = useCallback((beeList) => {
    if (!beeList) return;
    setBees(beeList);

    if (!targetBee) return;
    beeList.forEach((bee) => {
      if (
        String(bee.track_id) === String(targetBee) &&
        bee.is_locked &&
        !alertedBees.has(bee.track_id)
      ) {
        try {
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          [0, 0.4, 0.8, 1.2, 1.6].forEach((startTime) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.frequency.value = 880;
            osc.type = 'sine';
            gain.gain.setValueAtTime(0.5, ctx.currentTime + startTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + startTime + 0.3);
            osc.start(ctx.currentTime + startTime);
            osc.stop(ctx.currentTime + startTime + 0.3);
          });
        } catch (e) {
          console.warn('Audio not available:', e);
        }

        setAlertMessage(`🐝 Bee #${targetBee} detected!`);
        setTimeout(() => setAlertMessage(''), 5000);
        setAlertedBees((prev) => new Set(prev).add(bee.track_id));
      }
    });
  }, [targetBee, alertedBees]);

  const stopCameraStream = useCallback(() => {
    cameraActiveRef.current = false;
    waitingForFrameRef.current = false;

    if (wsRef.current) {
      try {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send('stop');
        }
        wsRef.current.close();
      } catch (e) {
        console.warn('WebSocket close:', e);
      }
      wsRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    revokeProcessedUrl();
    setProcessedFrameUrl(null);
  }, [revokeProcessedUrl]);

  const captureAndSendFrame = useCallback(() => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      !cameraActiveRef.current ||
      !ws ||
      ws.readyState !== WebSocket.OPEN ||
      !video ||
      !canvas ||
      waitingForFrameRef.current ||
      video.videoWidth === 0
    ) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob || !cameraActiveRef.current || ws.readyState !== WebSocket.OPEN) {
          return;
        }
        waitingForFrameRef.current = true;
        ws.send(blob);
      },
      'image/jpeg',
      0.72
    );
  }, []);

  const scheduleNextFrame = useCallback(() => {
    if (!cameraActiveRef.current) return;
    requestAnimationFrame(() => captureAndSendFrame());
  }, [captureAndSendFrame]);

  const handleWebSocketMessage = useCallback((event) => {
    waitingForFrameRef.current = false;

    if (event.data instanceof Blob) {
      revokeProcessedUrl();
      const url = URL.createObjectURL(event.data);
      processedUrlRef.current = url;
      setProcessedFrameUrl(url);
    } else if (event.data instanceof ArrayBuffer) {
      revokeProcessedUrl();
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      processedUrlRef.current = url;
      setProcessedFrameUrl(url);
    } else if (typeof event.data === 'string') {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'results') {
          applyBeeResults(msg.bees);
        } else if (msg.type === 'error') {
          setCameraError(msg.message || 'Processing error');
        }
      } catch (e) {
        console.warn('WebSocket message parse error:', e);
      }
    }

    scheduleNextFrame();
  }, [applyBeeResults, revokeProcessedUrl, scheduleNextFrame]);

  useEffect(() => {
    let interval;
    if (status === 'processing' && activeInputSource === 'video') {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${getApiBase()}/get-result`);
          if (res.data.bees) {
            applyBeeResults(res.data.bees);
          }
          if (res.data.video_ended) {
            setStatus('finished');
            clearInterval(interval);
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [status, activeInputSource, applyBeeResults]);

  useEffect(() => {
    return () => {
      stopCameraStream();
    };
  }, [stopCameraStream]);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setVideoUrl(null);
    setBees([]);
    setStatus('idle');
    e.target.value = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
      setStatus('uploading');
      const res = await axios.post(`${getApiBase()}/upload-video`, formData, {
        onUploadProgress: (progressEvent) => {
          const percent = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percent);
        },
      });
      setUploadProgress(0);
      setActiveInputSource('video');
      setStatus('processing');
      setVideoUrl(
        `${getApiBase()}/video-feed?filename=${res.data.filename}&session_id=${res.data.session_id}`
      );
    } catch (e) {
      alert('Upload failed');
    }
  };

  const handleStartCamera = async () => {
    setCameraError('');
    setVideoUrl(null);
    setBees([]);
    setAlertedBees(new Set());
    setAlertMessage('');
    revokeProcessedUrl();
    setProcessedFrameUrl(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      const isSecure =
        window.isSecureContext ||
        location.hostname === 'localhost' ||
        location.hostname === '127.0.0.1';
      setCameraError(
        isSecure
          ? 'Camera access is not supported in this browser.'
          : 'Camera requires HTTPS (or localhost). Open this app via https://… — Chrome blocks cameras on plain http:// sites.'
      );
      return;
    }

    try {
      const stream = await acquireCameraStream();

      mediaStreamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((t) => t.stop());
        setCameraError('Camera UI failed to initialize. Refresh the page and try again.');
        return;
      }

      video.srcObject = stream;
      await video.play();

      try {
        const health = await axios.get(`${getApiBase()}/health`, { timeout: 5000 });
        if (!health.data?.websocket_ready) {
          setCameraError(
            'Backend WebSocket is not active. On the server run: cd ~/bee-project && chmod +x deploy/fix-backend.sh && ./deploy/fix-backend.sh'
          );
          stopCameraStream();
          return;
        }
      } catch {
        setCameraError(
          'Cannot reach the AI backend. On the server: sudo systemctl restart bee-backend && sudo cp ~/bee-project/deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy'
        );
        stopCameraStream();
        return;
      }

      const ws = new WebSocket(`${getWsBase()}/ws/live`);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        cameraActiveRef.current = true;
        setActiveInputSource('camera');
        setStatus('processing');
        scheduleNextFrame();
      };

      ws.onmessage = handleWebSocketMessage;

      ws.onerror = () => {
        setCameraError(
          'WebSocket failed. On the server run: cd ~/bee-project && git pull && sudo systemctl restart bee-backend && sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy'
        );
        stopCameraStream();
        setStatus('idle');
        setActiveInputSource(null);
      };

      ws.onclose = (event) => {
        const wasActive = cameraActiveRef.current;
        cameraActiveRef.current = false;
        if (!wasActive) return;
        if (event.code !== 1000 && event.code !== 1001) {
          setCameraError(
            `AI server connection closed (${event.code}). Check Caddy proxies /ws to port 8000 and bee-backend is running.`
          );
          setStatus('idle');
          setActiveInputSource(null);
          stopCameraStream();
          return;
        }
        setStatus((prev) => (prev === 'processing' ? 'finished' : prev));
      };
    } catch (e) {
      console.error('Camera start error:', e);
      stopCameraStream();
      setCameraError(formatCameraError(e));
      setStatus('idle');
      setActiveInputSource(null);
    }
  };

  const reset = async () => {
    stopCameraStream();
    if (status === 'processing' || status === 'uploading') {
      try {
        await axios.post(`${getApiBase()}/stop-session`);
      } catch (e) {
        console.warn('Stop session:', e);
      }
    }
    setStatus('idle');
    setBees([]);
    setVideoUrl(null);
    setActiveInputSource(null);
    setAlertedBees(new Set());
    setAlertMessage('');
    setCameraError('');
  };

  const handleStopCamera = async () => {
    stopCameraStream();
    try {
      await axios.post(`${getApiBase()}/stop-session`);
    } catch (e) {
      console.warn('Stop session:', e);
    }
    setStatus('finished');
    setVideoUrl(null);
  };

  return (
    <div style={styles.container}>
      <div style={styles.sidebar}>
        <div style={styles.logo}>🐝 Bee<span>Vision</span></div>
        <div style={styles.targetInput}>
          <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px', textTransform: 'uppercase' }}>
            Target Bee ID
          </div>
          <input
            type="number"
            placeholder="e.g. 47"
            value={targetBee}
            onChange={(e) => {
              setTargetBee(e.target.value);
              setAlertedBees(new Set());
            }}
            style={styles.inputField}
          />
        </div>
        <div style={styles.listHeader}>Detections: {bees.length}</div>

        <div style={styles.scrollArea}>
          {bees.length === 0 && status === 'processing' && (
            <div style={styles.statusMsg}>Waiting for detections...</div>
          )}
          {bees.map((bee) => (
            <div
              key={bee.track_id}
              style={{
                ...styles.beeCard,
                borderColor: bee.is_locked ? '#f1c40f' : '#333',
              }}
            >
              <div style={styles.cardHeader}>
                <span>Track ID: {bee.track_id}</span>
                {bee.is_locked && <span style={styles.badge}>LOCKED</span>}
              </div>
              <div style={styles.beeNum}>{bee.number || 'Scanning...'}</div>
              <div style={styles.confidence}>
                Conf: {(bee.confidence * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
        {status !== 'idle' && (
          <button onClick={reset} style={styles.btnReset}>New Scan</button>
        )}
      </div>

      <div style={styles.main}>
        <video ref={videoRef} autoPlay playsInline muted style={styles.hiddenVideo} />
        <canvas ref={canvasRef} style={styles.hiddenVideo} />

        {alertMessage && (
          <div style={styles.alertBanner}>{alertMessage}</div>
        )}
        {status === 'idle' && (
          <div style={styles.upload}>
            <h2 style={{ marginTop: 0 }}>Tag & number Recognition</h2>
            <p style={{ color: '#888' }}>
              {inputSource === 'video'
                ? 'Upload video to identify tags'
                : 'Use your local webcam — frames stream to the server for AI detection'}
            </p>
            <div style={styles.sourceToggle}>
              <button
                type="button"
                onClick={() => setInputSource('video')}
                style={{
                  ...styles.sourceBtn,
                  ...(inputSource === 'video' ? styles.sourceBtnActive : {}),
                }}
              >
                Pre-recorded Video
              </button>
              <button
                type="button"
                onClick={() => setInputSource('camera')}
                style={{
                  ...styles.sourceBtn,
                  ...(inputSource === 'camera' ? styles.sourceBtnActive : {}),
                }}
              >
                Live Camera
              </button>
            </div>
            {inputSource === 'video' ? (
              <>
                <input type="file" id="up" hidden onChange={handleUpload} accept="video/*" />
                <label htmlFor="up" style={styles.btnUpload}>Select Video</label>
              </>
            ) : (
              <button
                type="button"
                onClick={handleStartCamera}
                style={{ ...styles.btnUpload, border: 'none' }}
              >
                Start Live Camera
              </button>
            )}
            {cameraError && (
              <p style={{ color: '#e74c3c', marginTop: '16px', fontSize: '14px' }}>{cameraError}</p>
            )}
          </div>
        )}

        {activeInputSource === 'video' && status !== 'idle' && videoUrl && (
          <div style={styles.videoBox}>
            <img src={videoUrl} alt="Processed video stream" style={styles.img} />
          </div>
        )}

        {activeInputSource === 'camera' && status === 'processing' && (
          <div style={styles.videoBox}>
            {processedFrameUrl ? (
              <img src={processedFrameUrl} alt="Live AI stream" style={styles.img} />
            ) : (
              <div style={styles.cameraPlaceholder}>Connecting camera...</div>
            )}
            <button type="button" onClick={handleStopCamera} style={styles.btnStopCamera}>
              Stop Camera
            </button>
          </div>
        )}

        {status === 'uploading' && (
          <div style={styles.upload}>
            <h2 style={{ marginTop: 0 }}>Uploading video...</h2>
            <div style={styles.progressBar}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${uploadProgress}%`,
                }}
              />
            </div>
            <p style={{ color: '#888' }}>{uploadProgress}%</p>
          </div>
        )}

        {status === 'finished' && (
          <div style={styles.finalOverlay}>
            <div style={styles.finalCard}>
              <div style={styles.finalHeader}>🎥 Session Complete</div>
              <div style={styles.finalSub}>
                Identified {bees.filter((b) => b.number).length} unique tags
              </div>
              <div style={styles.grid}>
                {bees.filter((b) => b.number).map((bee) => (
                  <div key={bee.track_id} style={styles.gridItem}>
                    <div style={styles.gridNum}>{bee.number}</div>
                    <div style={styles.gridId}>ID: {bee.track_id}</div>
                  </div>
                ))}
              </div>
              <button onClick={reset} style={styles.finalBtn}>New Session</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { display: 'flex', height: '100vh', background: '#111', color: '#eee', fontFamily: 'Segoe UI, sans-serif', overflow: 'hidden' },
  sidebar: { width: '300px', background: '#1a1a1a', padding: '20px', borderRight: '1px solid #333', display: 'flex', flexDirection: 'column' },
  logo: { fontSize: '24px', fontWeight: 'bold', marginBottom: '20px', color: '#f1c40f', letterSpacing: '1px' },
  listHeader: { fontSize: '12px', textTransform: 'uppercase', color: '#888', marginBottom: '10px', borderBottom: '1px solid #333', paddingBottom: '5px' },
  scrollArea: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' },
  statusMsg: { color: '#666', fontStyle: 'italic', textAlign: 'center', marginTop: 20 },
  beeCard: { background: '#252525', padding: '15px', borderRadius: '8px', border: '1px solid #333' },
  cardHeader: { display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#888', marginBottom: '5px' },
  badge: { background: '#f1c40f', color: '#000', padding: '2px 5px', borderRadius: '4px', fontWeight: 'bold' },
  beeNum: { fontSize: '28px', fontWeight: 'bold', color: '#fff' },
  confidence: { fontSize: '12px', color: '#aaa' },
  btnReset: { marginTop: '20px', padding: '12px', background: 'transparent', border: '1px solid #444', color: '#eee', cursor: 'pointer', borderRadius: '8px' },
  main: { flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#0a0a0a', position: 'relative' },
  upload: { textAlign: 'center', padding: '60px', background: '#141414', borderRadius: '20px', border: '1px dashed #333' },
  btnUpload: { display: 'inline-block', marginTop: '20px', padding: '12px 30px', background: '#f1c40f', color: '#000', borderRadius: '30px', cursor: 'pointer', fontWeight: 'bold' },
  videoBox: { maxWidth: '90%', maxHeight: '90%', border: '2px solid #333', borderRadius: '10px', overflow: 'hidden' },
  img: { display: 'block', width: '100%', maxHeight: '80vh' },
  hiddenVideo: { position: 'absolute', width: 1, height: 1, opacity: 0, pointerEvents: 'none' },
  cameraPlaceholder: { padding: '80px 40px', textAlign: 'center', color: '#666', fontStyle: 'italic' },
  finalOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', display: 'flex', justifyContent: 'center', alignItems: 'center' },
  finalCard: { background: '#1e1e1e', padding: '40px', borderRadius: '20px', border: '1px solid #f1c40f', width: '500px', textAlign: 'center' },
  finalHeader: { fontSize: '24px', marginBottom: '10px' },
  finalSub: { color: '#888', marginBottom: '20px' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '10px', marginBottom: '20px', maxHeight: '200px', overflowY: 'auto' },
  gridItem: { background: '#252525', padding: '10px', borderRadius: '8px', border: '1px solid #444' },
  gridNum: { fontSize: '20px', fontWeight: 'bold', color: '#f1c40f' },
  gridId: { fontSize: '10px', color: '#666' },
  finalBtn: { width: '100%', padding: '15px', background: '#f1c40f', color: '#000', border: 'none', borderRadius: '10px', fontWeight: 'bold', cursor: 'pointer' },
  progressBar: { width: '300px', height: '8px', background: '#333', borderRadius: '4px', overflow: 'hidden', margin: '20px auto' },
  progressFill: { height: '100%', background: '#f1c40f', borderRadius: '4px', transition: 'width 0.3s ease' },
  targetInput: { marginBottom: '16px', paddingBottom: '16px', borderBottom: '1px solid #333' },
  inputField: { width: '100%', padding: '8px 10px', background: '#252525', border: '1px solid #444', borderRadius: '6px', color: '#fff', fontSize: '16px', boxSizing: 'border-box' },
  alertBanner: { position: 'absolute', top: '20px', left: '50%', transform: 'translateX(-50%)', background: '#f1c40f', color: '#000', padding: '14px 28px', borderRadius: '12px', fontSize: '18px', fontWeight: 'bold', zIndex: 1000, boxShadow: '0 4px 20px rgba(241,196,15,0.5)' },
  sourceToggle: { display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '20px', flexWrap: 'wrap' },
  sourceBtn: { padding: '10px 16px', background: 'transparent', border: '1px solid #444', color: '#aaa', borderRadius: '30px', cursor: 'pointer', fontSize: '13px' },
  sourceBtnActive: { borderColor: '#f1c40f', color: '#f1c40f', background: 'rgba(241,196,15,0.1)' },
  btnStopCamera: { display: 'block', width: '100%', marginTop: '12px', padding: '12px', background: 'transparent', border: '1px solid #444', color: '#eee', cursor: 'pointer', borderRadius: '8px' },
};

export default App;
