// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [status, setStatus] = useState('idle');
  const [bees, setBees] = useState([]); 
  const [videoUrl, setVideoUrl] = useState(null);

  useEffect(() => {
    let interval;
    if (status === 'processing') {
      interval = setInterval(async () => {
        try {
          const res = await axios.get("http://localhost:8000/get-result");
          if (res.data.bees) {
            setBees(res.data.bees);
          }
          if (res.data.video_ended) {
            setStatus('finished');
            clearInterval(interval);
          }
        } catch (e) { console.error("Poll error:", e); }
      }, 1000); 
    }
    return () => clearInterval(interval);
  }, [status]);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setVideoUrl(null);
    setBees([]);
    setStatus('idle');
    e.target.value = ""; 

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/upload-video", formData);
      setVideoUrl(`http://localhost:8000/video-feed?filename=${res.data.filename}&session_id=${res.data.session_id}`);
      setStatus('processing');
    } catch (e) { alert("Upload failed"); }
  };

  const reset = () => {
    setStatus('idle');
    setBees([]);
    setVideoUrl(null);
  };

  return (
    <div style={styles.container}>
      <div style={styles.sidebar}>
        <div style={styles.logo}>🐝 Bee<span>Vision</span></div>
        <div style={styles.listHeader}>Detections: {bees.length}</div>
        
        <div style={styles.scrollArea}>
          {bees.length === 0 && status === 'processing' && (
            <div style={styles.statusMsg}>Waiting for data...</div>
          )}
          {bees.map((bee) => (
            <div key={bee.track_id} style={{
              ...styles.beeCard, 
              borderColor: bee.is_locked ? '#f1c40f' : '#333'
            }}>
              <div style={styles.cardHeader}>
                <span>Track ID: {bee.track_id}</span>
                {bee.is_locked && <span style={styles.badge}>LOCKED</span>}
              </div>
              <div style={styles.beeNum}>{bee.number || 'Scanning...'}</div>
              <div style={styles.confidence}>Conf: {(bee.confidence * 100).toFixed(0)}%</div>
            </div>
          ))}
        </div>
        {status !== 'idle' && (
          <button onClick={reset} style={styles.btnReset}>New Scan</button>
        )}
      </div>

      <div style={styles.main}>
        {status === 'idle' && (
          <div style={styles.upload}>
            <h2 style={{marginTop: 0}}>Tag & number Recognition</h2>
            <p style={{color: '#888'}}>Upload video to identify tags</p>
            <input type="file" id="up" hidden onChange={handleUpload} accept="video/*" />
            <label htmlFor="up" style={styles.btnUpload}>Select Video</label>
          </div>
        )}
        {status !== 'idle' && videoUrl && (
          <div style={styles.videoBox}>
            <img src={videoUrl} alt="Live Stream" style={styles.img} />
          </div>
        )}

        {status === 'finished' && (
          <div style={styles.finalOverlay}>
            <div style={styles.finalCard}>
              <div style={styles.finalHeader}>🎥 Session Complete</div>
              <div style={styles.finalSub}>Identified {bees.filter(b => b.number).length} unique tags</div>
              <div style={styles.grid}>
                {bees.filter(b => b.number).map(bee => (
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
  finalOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.9)', display: 'flex', justifyContent: 'center', alignItems: 'center' },
  finalCard: { background: '#1e1e1e', padding: '40px', borderRadius: '20px', border: '1px solid #f1c40f', width: '500px', textAlign: 'center' },
  finalHeader: { fontSize: '24px', marginBottom: '10px' },
  finalSub: { color: '#888', marginBottom: '20px' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '10px', marginBottom: '20px', maxHeight: '200px', overflowY: 'auto' },
  gridItem: { background: '#252525', padding: '10px', borderRadius: '8px', border: '1px solid #444' },
  gridNum: { fontSize: '20px', fontWeight: 'bold', color: '#f1c40f' },
  gridId: { fontSize: '10px', color: '#666' },
  finalBtn: { width: '100%', padding: '15px', background: '#f1c40f', color: '#000', border: 'none', borderRadius: '10px', fontWeight: 'bold', cursor: 'pointer' }
};

export default App;