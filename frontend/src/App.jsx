// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [status, setStatus] = useState('idle'); // idle, processing, finished
  const [summary, setSummary] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);

  // Polls the backend for results
  useEffect(() => {
    let interval;
    if (status === 'processing') {
      interval = setInterval(async () => {
        try {
          const res = await axios.get("http://localhost:8000/get-result");
          setSummary(res.data);
          
          // Check if video finished playing
          if (res.data.video_ended) {
            setStatus('finished');
            clearInterval(interval);
          }
        } catch (e) { console.error(e); }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [status]);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setVideoUrl(null);
    setSummary(null);
    e.target.value = ""; 

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/upload-video", formData);
      const { filename, session_id } = res.data;
      const streamUrl = `http://localhost:8000/video-feed?filename=${filename}&session_id=${session_id}`;
      
      setVideoUrl(streamUrl);
      setStatus('processing');
    } catch (e) {
      alert("Upload failed.");
    }
  };

  const reset = () => {
    setStatus('idle');
    setSummary(null);
    setVideoUrl(null);
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <div style={styles.sidebar}>
        <div style={styles.logo}>🐝 Bee<span>Vision</span></div>
        
        <div style={styles.stats}>
          <div style={styles.statBox}>
            <label style={styles.label}>Current ID</label>
            <span style={{...styles.val, color: summary?.is_locked ? '#f1c40f' : '#fff'}}>
              {summary?.id || "--"}
            </span>
          </div>

          <div style={styles.statBox}>
            <label style={styles.label}>Status</label>
            <div style={{display: 'flex', alignItems: 'center'}}>
               <span>{status === 'finished' ? "Completed" : (summary?.status || "Idle")}</span>
               {summary?.is_locked && <span style={styles.badge}>LOCKED</span>}
            </div>
          </div>
          
          <div style={styles.statBox}>
            <label style={styles.label}>Confidence</label>
             <span>{(summary?.confidence * 100 || 0).toFixed(1)}%</span>
          </div>
        </div>

        {status !== 'idle' && (
          <button onClick={reset} style={styles.btnReset}>New Scan</button>
        )}
      </div>

      {/* Main Area */}
      <div style={styles.main}>
        {status === 'idle' && (
          <div style={styles.upload}>
            <h2 style={{marginTop: 0}}>Start New Analysis</h2>
            <p style={{color: '#888'}}>Upload a video file to begin detection</p>
            <input type="file" id="up" hidden onChange={handleUpload} accept="video/*" />
            <label htmlFor="up" style={styles.btnUpload}>Select Video</label>
          </div>
        )}

        {status === 'processing' && (
          <div style={styles.videoBox}>
            {videoUrl && <img src={videoUrl} alt="Live Stream" style={styles.img} />}
          </div>
        )}

        {/* --- FINAL REPORT CARD (Pop-up when finished) --- */}
        {status === 'finished' && (
          <div style={styles.finalCard}>
            <div style={styles.finalHeader}>🎥 Analysis Complete</div>
            
            <div style={styles.finalContent}>
              <div style={styles.finalBigNum}>
                {summary?.id || "Unknown"}
              </div>
              <div style={styles.finalLabel}>FINAL IDENTIFIED TAG</div>
              
              <div style={styles.finalStats}>
                <div>Confidence: <strong>{(summary?.confidence * 100 || 0).toFixed(1)}%</strong></div>
                <div>Method: <strong>{summary?.is_locked ? "Majority Vote (Locked)" : "Single Detection"}</strong></div>
              </div>

              <button onClick={reset} style={styles.finalBtn}>Analyze Another Video</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { display: 'flex', height: '100vh', background: '#111', color: '#eee', fontFamily: 'Segoe UI, sans-serif' },
  sidebar: { width: '280px', background: '#1a1a1a', padding: '30px', borderRight: '1px solid #333', display: 'flex', flexDirection: 'column' },
  logo: { fontSize: '24px', fontWeight: 'bold', marginBottom: '40px', color: '#f1c40f', letterSpacing: '1px' },
  stats: { flex: 1 },
  statBox: { background: '#252525', padding: '20px', borderRadius: '10px', marginBottom: '15px', border: '1px solid #333' },
  label: { display: 'block', fontSize: '12px', textTransform: 'uppercase', color: '#888', marginBottom: '8px' },
  val: { fontSize: '32px', fontWeight: 'bold', display: 'block', lineHeight: '1' },
  badge: { background: '#2ecc71', color: '#000', padding: '3px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold', marginLeft: '10px' },
  btnReset: { width: '100%', padding: '12px', background: 'transparent', border: '1px solid #ff4757', color: '#ff4757', cursor: 'pointer', borderRadius: '8px', fontWeight: 'bold', transition: 'all 0.2s' },
  main: { flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#0a0a0a', position: 'relative' },
  
  upload: { textAlign: 'center', padding: '60px', border: '2px dashed #444', borderRadius: '15px', background: '#141414' },
  btnUpload: { display: 'inline-block', marginTop: '20px', padding: '12px 30px', background: '#f1c40f', color: '#000', borderRadius: '30px', cursor: 'pointer', fontWeight: 'bold', fontSize: '16px' },
  
  videoBox: { border: '2px solid #333', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 10px 30px rgba(0,0,0,0.5)', maxWidth: '90%' },
  img: { display: 'block', maxWidth: '100%', maxHeight: '80vh' },

  // Final Card Styles
  finalCard: { background: '#1e1e1e', padding: '40px', borderRadius: '20px', boxShadow: '0 20px 60px rgba(0,0,0,0.8)', textAlign: 'center', border: '1px solid #f1c40f', minWidth: '400px', animation: 'fadeIn 0.5s ease' },
  finalHeader: { fontSize: '18px', color: '#aaa', textTransform: 'uppercase', marginBottom: '20px', letterSpacing: '1px' },
  finalBigNum: { fontSize: '80px', fontWeight: 'bold', color: '#f1c40f', lineHeight: '1', marginBottom: '10px' },
  finalLabel: { fontSize: '14px', color: '#fff', fontWeight: 'bold', marginBottom: '30px' },
  finalStats: { display: 'flex', justifyContent: 'space-between', background: '#252525', padding: '15px', borderRadius: '10px', fontSize: '14px', color: '#ccc', marginBottom: '30px' },
  finalBtn: { width: '100%', padding: '15px', background: '#f1c40f', color: '#000', border: 'none', borderRadius: '10px', fontSize: '16px', fontWeight: 'bold', cursor: 'pointer' }
};

export default App;