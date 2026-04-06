import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';
import './index.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'https://house-value-prediction-app.onrender.com';
const GOOGLE_CLIENT_ID = '474600099307-84dof3mch66i1mrat962jct7dqmc7go0.apps.googleusercontent.com'; // Replace with yours

const MODELS = [
  { value: 'random_forest',     label: 'Random Forest',     color: 'green',  v: 'V4.2', desc: 'High-performance ensemble learning utilizing multiple decision trees for robust valuation output.' },
  { value: 'decision_tree',     label: 'Decision Tree',     color: 'orange', v: 'V3.1', desc: 'Highly interpretable predictive logic mapping asset features to specific value branches.' },
  { value: 'linear_regression', label: 'Linear Regression', color: 'blue',   v: 'V1.4', desc: 'Statistical modeling of linear relationships between market variables and property value.' },
];

/* ─── Helpers ─── */
function parseInputData(raw) {
  try {
    const json = raw.replace(/'/g, '"').replace(/True/g, 'true').replace(/False/g, 'false');
    return JSON.parse(json);
  } catch { return null; }
}

const FIELD_LABELS = {
  longitude: 'Longitude', latitude: 'Latitude', housing_median_age: 'Median Age (yrs)',
  total_rooms: 'Total Rooms', total_bedrooms: 'Total Bedrooms', population: 'Population',
  households: 'Households', median_income: 'Median Income (×$10k)', ocean_proximity: 'Ocean Proximity',
};

/* ─── Components ─── */
function Sidebar({ onLogout }) {
  const { pathname } = useLocation();
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-logo">⌂</div>
        <div><div className="brand-name">Sovereign</div><div className="brand-sub">Intelligence Terminal</div></div>
      </div>
      <nav className="nav-section">
        <div className="nav-label">Main Console</div>
        <Link to="/" className={`nav-item ${pathname === '/' ? 'active' : ''}`}>
          {pathname === '/' && <div className="nav-active-bar" />} Dashboard
        </Link>
        <Link to="/predict" className={`nav-item ${pathname === '/predict' ? 'active' : ''}`}>
          {pathname === '/predict' && <div className="nav-active-bar" />} Valuation Engine
        </Link>
        <Link to="/inferences" className={`nav-item ${pathname === '/inferences' ? 'active' : ''}`}>
          {pathname === '/inferences' && <div className="nav-active-bar" />} History Feed
        </Link>
      </nav>
      <div className="sidebar-footer">
        <Link to="/predict" className="new-btn" style={{textDecoration:'none'}}>+ New Prediction</Link>
        <div className="nav-item" onClick={onLogout} style={{cursor:'pointer'}}>Sign Out</div>
      </div>
    </aside>
  );
}

function Header({ title, user }) {
  const { pathname } = useLocation();
  return (
    <header className="header">
      <div className="header-title">{title}</div>
      <nav className="header-nav">
        <Link to="/" className={`header-link ${pathname === '/' ? 'active' : ''}`}>Dashboard</Link>
        <Link to="/predict" className={`header-link ${pathname === '/predict' ? 'active' : ''}`}>Predictor</Link>
        <Link to="/inferences" className={`header-link ${pathname === '/inferences' ? 'active' : ''}`}>History</Link>
      </nav>
      <div className="header-actions">
        <div style={{display:'flex', alignItems:'center', gap:'1rem'}}>
          <div style={{textAlign:'right'}}>
            <div style={{fontSize:'0.75rem', fontWeight:800}}>{user?.name}</div>
            <div style={{fontSize:'0.6rem', color:'var(--text-dim)'}}>{user?.email}</div>
          </div>
          <img className="avatar" src={user?.picture} alt="profile" style={{width:'32px', height:'32px', border:'1px solid var(--primary)'}} />
        </div>
      </div>
    </header>
  );
}

function DetailModal({ record, onClose }) {
  if (!record) return null;
  const data = parseInputData(record.input_data);
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-header-left">
            <span className="hero-tag" style={{margin:0}}>{record.model_name.replace(/_/g, ' ')}</span>
            <span style={{fontSize:'0.7rem', color:'#888'}}>{new Date(record.timestamp + 'Z').toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour12: true })} IST</span>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-result" style={{padding:'2rem'}}>
          <div className="hero-label">Estimated Market Value</div>
          <div className="hero-value" style={{fontSize:'3rem'}}>{new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(record.prediction)}</div>
        </div>
        <div className="modal-divider" />
        <div className="modal-section-label">Valuer Identity</div>
        <div style={{padding:'0.5rem 1.5rem', display:'flex', alignItems:'center', gap:'0.75rem'}}>
          <div style={{width:'32px', height:'32px', borderRadius:'50%', background:'var(--primary)', color:'black', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.8rem', fontWeight:800}}>
            {record.user_name?.charAt(0) || '?'}
          </div>
          <div>
            <div style={{fontSize:'0.8rem', fontWeight:700}}>{record.user_name || 'Anonymous Valuer'}</div>
            <div style={{fontSize:'0.6rem', color:'var(--text-dim)'}}>{record.user_email || 'No email record'}</div>
          </div>
        </div>
        <div className="modal-divider" />
        <div className="modal-section-label">Asset Core Parameters</div>
        {data && <div className="modal-grid" style={{padding:'1rem 1.5rem'}}>
          {Object.entries(data).map(([k, v]) => (
            <div key={k} className="modal-field">
              <div className="modal-field-label">{FIELD_LABELS[k] ?? k}</div>
              <div className="modal-field-value">{String(v)}</div>
            </div>
          ))}
        </div>}
      </div>
    </div>
  );
}

/* ─── Dashboard Home ─── */
function DashboardPage() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/metrics`).then(r => r.json()).then(setMetrics).catch(() => {});
    fetch(`${API_BASE}/inferences/random_forest`).then(r => r.json()).then(data => setHistory(data.slice(-5).reverse())).catch(() => {});
  }, []);

  return (
    <div className="page-body">
      <div style={{display: 'flex', gap: '2rem'}}>
        <div style={{flex: 1}}>
          <div className="section-title">System Overview</div>
          <div className="section-sub">Consolidated real-time intelligence from the Oracle predictive cluster.</div>
          
          <div className="model-grid">
            {MODELS.map(m => (
              <div key={m.value} className="model-card">
                <div className="model-card-label" style={{color:'var(--primary)'}}>Engine: {m.label}</div>
                <div className="model-card-title">{metrics?.[m.value]?.R2 ? `${(metrics[m.value].R2 * 100).toFixed(1)}%` : '...'} Accuracy</div>
                <div className="model-metrics">
                   <div className="metric-item"><div className="metric-label">RMSE</div><div className="metric-value">{metrics?.[m.value]?.RMSE ? `$${(metrics[m.value].RMSE/1000).toFixed(1)}k` : '...'}</div></div>
                   <div className="metric-item"><div className="metric-label">Status</div><div className="metric-value" style={{color:'var(--primary)'}}>Live</div></div>
                </div>
              </div>
            ))}
          </div>

          <div className="glass-card" style={{marginTop: '2rem'}}>
            <div className="section-title">Recent Network Activity</div>
            <div className="table-container" style={{marginTop:'1.5rem', border:'none'}}>
               <table>
                  <thead><tr><th>Date</th><th>Model</th><th>Valuer</th><th>Prediction</th></tr></thead>
                  <tbody>
                    {history.map(rec => (
                      <tr key={rec.id} className="table-row" onClick={() => setSelected(rec)} style={{cursor:'pointer'}}>
                        <td style={{fontSize:'0.75rem'}}>{new Date(rec.timestamp + 'Z').toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour12: true })}</td>
                        <td><span className="hero-tag" style={{fontSize:'0.6rem', padding:'0.1rem 0.4rem'}}>{rec.model_name.replace(/_/g, ' ')}</span></td>
                        <td>
                          <div style={{fontSize:'0.75rem', fontWeight:600}}>{rec.user_name || 'System'}</div>
                          <div style={{fontSize:'0.6rem', color:'var(--text-dim)'}}>{rec.user_email?.split('@')[0] || ''}</div>
                        </td>
                        <td style={{fontWeight:700, color:'var(--primary)'}}>${(rec.prediction/1000).toFixed(1)}k</td>
                      </tr>
                    ))}
                  </tbody>
               </table>
            </div>
            <Link to="/inferences" style={{display:'block', textAlign:'right', marginTop:'1rem', fontSize:'0.75rem', color:'var(--primary)', textDecoration:'none'}}>View full audit log ›</Link>
          </div>
        </div>

        <aside style={{width: '320px'}}>
           <div className="glass-card">
              <div className="nav-label">QUICK ACTIONS</div>
              <Link to="/predict" className="new-btn" style={{textDecoration:'none', marginBottom:'1.5rem'}}>⚡ Launch Valuation Engine</Link>
              <div className="nav-label">REAL-TIME TELEMETRY</div>
              <div style={{marginTop:'1rem', display:'flex', flexDirection:'column', gap:'0.75rem'}}>
                 <div style={{display:'flex', justifyContent:'space-between'}}><span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>Global History</span> <span style={{fontSize:'0.7rem', fontWeight:700}}>{history.length} Inferences</span></div>
                 <div style={{display:'flex', justifyContent:'space-between'}}><span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>System Status</span> <span style={{fontSize:'0.7rem', fontWeight:700, color:'var(--primary)'}}>ESTABLISHED</span></div>
              </div>
           </div>
        </aside>
      </div>
      <DetailModal record={selected} onClose={() => setSelected(null)} />
    </div>
  );
}

/* ─── Pages ─── */
function PredictPage({ user }) {
  const [form, setForm] = useState({
    longitude: -122.23, latitude: 37.88, housing_median_age: 41,
    total_rooms: 880, total_bedrooms: 129, population: 322,
    households: 126, median_income: 8.3252, ocean_proximity: 'NEAR BAY',
  });
  const [model, setModel] = useState('random_forest');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: name === 'ocean_proximity' ? value : (parseFloat(value) ?? value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); setLoading(true); setResult(null);
    try {
      const payload = { ...form, user_name: user?.name, user_email: user?.email };
      const res = await fetch(`${API_BASE}/predict/${model}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error');
      setResult(data.prediction);
    } catch (err) { alert(err.message); } finally { setLoading(false); }
  };

  const fmt = (v) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v);

  return (
    <div className="page-body">
      <div style={{display: 'flex', gap: '2rem'}}>
        <div style={{flex: 1}}>
          {result ? (
            <div className="result-view">
              <div className="result-hero" style={{backgroundImage: `url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&q=80&w=2000')`}}>
                <div className="result-hero-overlay"></div>
                <div className="result-hero-content">
                  <span className="hero-tag">Verified Model : {MODELS.find(m => m.value === model).label} {MODELS.find(m => m.value === model).v}</span>
                  <div className="hero-label">Estimated Market Value</div>
                  <div className="hero-value">{fmt(result)}</div>
                  <div className="hero-label" style={{fontSize: '0.7rem', marginTop: '1rem'}}>📍 {form.latitude}° N, {form.longitude}° W — Sector 4B Residential</div>
                </div>
              </div>
              <div className="details-grid">
                <div className="detail-card"><div className="detail-label">Median Age</div><div className="detail-value">{form.housing_median_age} <span className="detail-unit">Years</span></div></div>
                <div className="detail-card"><div className="detail-label">Total Rooms</div><div className="detail-value">{form.total_rooms} <span className="detail-unit">Avg</span></div></div>
                <div className="detail-card"><div className="detail-label">Population</div><div className="detail-value">{(form.population/1000).toFixed(1)}k <span className="detail-unit">Density</span></div></div>
                <div className="detail-card"><div className="detail-label">Income</div><div className="detail-value">{form.median_income.toFixed(1)} <span className="detail-unit">Index</span></div></div>
              </div>
              <button className="new-btn" style={{marginTop: '2rem', width: 'auto', padding: '0.75rem 2rem'}} onClick={() => setResult(null)}>Reset Parameters</button>
            </div>
          ) : (
            <div className="predict-view">
              <div className="section-title">Valuation Engine</div>
              <div className="section-sub">Configure the neural parameters and select your institutional-grade model to estimate property valuations with sovereign precision.</div>
              <div className="model-grid">
                {MODELS.map(m => (
                  <div key={m.value} className={`model-card ${model === m.value ? 'active' : ''}`} onClick={() => setModel(m.value)}>
                    <div className="model-card-label">Model: {m.label} {m.v}</div>
                    <div className="model-card-title">{m.label}</div>
                    <div className="model-card-desc">{m.desc}</div>
                  </div>
                ))}
              </div>
              <div className="glass-card" style={{marginTop: '2rem'}}>
                <div className="section-title">Asset Core Parameters</div>
                <form onSubmit={handleSubmit} style={{marginTop: '1.5rem'}}>
                  <div className="input-grid">
                    {[{ name: 'longitude', label: 'Longitude' }, { name: 'latitude', label: 'Latitude' }, { name: 'housing_median_age', label: 'Median House Age' }, { name: 'total_rooms', label: 'Total Rooms' }, { name: 'total_bedrooms', label: 'Total Bedrooms' }, { name: 'population', label: 'Population' }, { name: 'households', label: 'Households' }, { name: 'median_income', label: 'Median Income' }].map(f => (
                      <div className="input-group" key={f.name}><label>{f.label}</label><input type="number" step="any" name={f.name} value={form[f.name]} onChange={handleChange} required /></div>
                    ))}
                    <div className="input-group"><label>Ocean Proximity</label><select name="ocean_proximity" value={form.ocean_proximity} onChange={handleChange}>
                        <option value="NEAR BAY">Near Bay</option><option value="&lt;1H OCEAN">&lt;1H Ocean</option><option value="INLAND">Inland</option><option value="NEAR OCEAN">Near Ocean</option><option value="ISLAND">Island</option>
                      </select></div>
                  </div>
                  <button type="submit" className="new-btn" style={{marginTop: '2rem', height: '56px', fontSize: '1rem'}} disabled={loading}>{loading ? 'CALCULATING SOVEREIGN ESTIMATE...' : `PREDICT WITH ${model.replace(/_/g, ' ').toUpperCase()} ⚡`}</button>
                </form>
              </div>
            </div>
          )}
        </div>
        
        {/* Side Panel */}
        <aside style={{width: '320px', display:'flex', flexDirection:'column', gap: '1.5rem'}}>
          <div className="glass-card">
            <div className="nav-label">SYSTEM CONTEXT</div>
            <div style={{display:'flex', flexDirection:'column', gap:'1rem', marginTop:'1rem'}}>
               <div style={{display:'flex', justifyContent:'space-between'}}><span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>Current Engine</span> <span style={{fontSize:'0.7rem', fontWeight:700, color:'var(--primary)'}}>{MODELS.find(m => m.value === model).label}</span></div>
               <div style={{display:'flex', justifyContent:'space-between'}}><span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>Model Version</span> <span style={{fontSize:'0.7rem', fontWeight:700}}>{MODELS.find(m => m.value === model).v}</span></div>
               <div style={{display:'flex', justifyContent:'space-between'}}><span style={{fontSize:'0.7rem', color:'var(--text-muted)'}}>Input Features</span> <span style={{fontSize:'0.7rem', fontWeight:700}}>9 Parameters</span></div>
            </div>
          </div>
          <div className="glass-card">
            <div className="nav-label">OPERATIONS</div>
            <p style={{fontSize:'0.75rem', color:'var(--text-muted)', marginTop:'0.5rem'}}>Select parameters on the left to generate a real-time housing valuation.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}

function InferencesPage() {
  const [filter, setFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('');
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => { fetchHistory(); const id = setInterval(fetchHistory, 5000); return () => clearInterval(id); }, [filter]);

  const fetchHistory = async () => {
    try {
      let data = [];
      if (filter === 'all') {
        const all = await Promise.all(MODELS.map(m => fetch(`${API_BASE}/inferences/${m.value}`).then(r => r.json())));
        data = all.flat().sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      } else {
        data = await fetch(`${API_BASE}/inferences/${filter}`).then(r => r.json());
        data = [...data].reverse();
      }
      setHistory(data);
    } catch {}
  };

  const filteredHistory = history.filter(rec => {
    if (!dateFilter) return true;
    const recDate = new Date(rec.timestamp + 'Z').toISOString().split('T')[0];
    return recDate === dateFilter;
  });

  const fmt = (v) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v);

  return (
    <div className="page-body">
      <div style={{display: 'flex', gap: '2rem'}}>
        <div style={{flex: 1}}>
          <div className="section-title">Historical Inferences</div>
          <div className="section-sub">Reviewing industrial-grade predictive market valuations generated by the Oracle engine.</div>
          
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-end', marginBottom: '1.5rem', flexWrap:'wrap', gap:'1rem'}}>
            <div className="filter-tabs" style={{display: 'flex', gap: '1rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.2rem'}}>
              {[{value: 'all', label: 'All Models'}, ...MODELS].map(f => (
                <button 
                  key={f.value}
                  onClick={() => setFilter(f.value)}
                  style={{
                    background: 'none', border: 'none', color: filter === f.value ? 'var(--primary)' : 'var(--text-muted)',
                    fontSize: '0.7rem', fontWeight: 700, cursor: 'pointer', padding: '0.5rem 0',
                    borderBottom: filter === f.value ? '2px solid var(--primary)' : '2px solid transparent',
                    textTransform: 'uppercase', letterSpacing: '0.5px', transition: 'all 0.2s'
                  }}
                >
                  {f.label}
                </button>
              ))}
            </div>
            
            <div className="input-group" style={{maxWidth:'200px'}}>
              <label style={{fontSize:'0.6rem', color:'var(--text-dim)', marginBottom:'0.2rem'}}>Filter by Date</label>
              <div style={{display:'flex', gap:'0.5rem'}}>
                <input 
                  type="date" 
                  value={dateFilter} 
                  onChange={e => setDateFilter(e.target.value)} 
                  style={{padding:'0.4rem', fontSize:'0.75rem', background:'var(--card-bg)', border:'1px solid var(--border)', color:'white', borderRadius:'4px'}}
                />
                {dateFilter && <button onClick={() => setDateFilter('')} style={{background:'none', border:'none', color:'#888', cursor:'pointer'}}>✕</button>}
              </div>
            </div>
          </div>
          <div className="table-container">
            <table>
              <thead><tr><th>Inference Date</th><th>Model Engine</th><th>Valuer</th><th>Estimated Value</th><th>Actions</th></tr></thead>
              <tbody>
                {filteredHistory.map(rec => (
                  <tr key={rec.id} className="table-row" onClick={() => setSelected(rec)} style={{cursor: 'pointer'}}>
                    <td>
                      <div style={{fontWeight: 700}}>{new Date(rec.timestamp + 'Z').toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric'})}</div>
                      <div style={{fontSize: '0.7rem', color: '#555'}}>{new Date(rec.timestamp + 'Z').toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true })}</div>
                    </td>
                    <td><span className="hero-tag" style={{fontSize: '0.6rem', padding: '0.1rem 0.4rem'}}>{rec.model_name.replace(/_/g, ' ')}</span></td>
                    <td>
                      <div style={{fontSize:'0.75rem', fontWeight:600}}>{rec.user_name || 'System'}</div>
                      <div style={{fontSize:'0.6rem', color:'var(--text-dim)'}}>{rec.user_email || ''}</div>
                    </td>
                    <td><span className="price-pill">{fmt(rec.prediction)}</span> <span className="change-badge">+2.4%</span></td>
                    <td><button style={{background: 'none', border: 'none', color: '#888', cursor: 'pointer', pointerEvents: 'none'}}>Details ›</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <aside style={{width: '320px'}}>
          <div className="glass-card">
            <div className="nav-label">MARKET ADVISORY</div>
            <p style={{fontSize:'0.8rem', color:'var(--text-muted)', lineHeight:'1.4', marginTop: '1rem'}}>
              Based on the latest data sequence, the {MODELS.find(m => m.value === (filter === 'all' ? 'random_forest' : filter))?.label} shows consistent predictive performance in this sector.
            </p>
            <div style={{display:'flex', gap:'1rem', marginTop:'1.5rem'}}>
               <div><div style={{fontSize:'0.6rem', color:'var(--text-dim)'}}>SYSTEM STATUS</div><div style={{fontSize:'0.9rem', fontWeight:800, color:'var(--primary)'}}>OPERATIONAL</div></div>
               <div><div style={{fontSize:'0.6rem', color:'var(--text-dim)'}}>AUDIT LOGS</div><div style={{fontSize:'0.9rem', fontWeight:800}}>{history.length} RECORDS</div></div>
            </div>
          </div>
        </aside>
      </div>
      <DetailModal record={selected} onClose={() => setSelected(null)} />
    </div>
  );
}

/* ─── Security Perimeter ─── */
function LoginPage({ onLogin }) {
  return (
    <div className="login-screen">
      <div className="login-card glass-card">
         <div className="security-icon">⌂</div>
         <h1 className="section-title" style={{fontSize:'1.5rem', textAlign:'center', marginBottom:'0.5rem'}}>Sovereign Console</h1>
         <p className="section-sub" style={{textAlign:'center', marginBottom:'2rem'}}>Terminal Access Restricted: Authenticate with Google to enter.</p>
         
         <div style={{display:'flex', justifyContent:'center'}}>
            <GoogleLogin
              onSuccess={credentialResponse => {
                const user = jwtDecode(credentialResponse.credential);
                localStorage.setItem('user', JSON.stringify(user));
                onLogin(user);
              }}
              onError={() => console.log('Login Failed')}
              theme="filled_black"
              shape="pill"
            />
         </div>
         
         <div className="login-footer">
            <span className="hero-tag" style={{fontSize: '0.6rem'}}>SECURITY STATUS: ENCRYPTED</span>
         </div>
      </div>
    </div>
  );
}

/* ─── App Root ─── */
export default function App() {
  const [user, setUser] = useState(() => JSON.parse(localStorage.getItem('user')));

  if (!user) {
    return (
      <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
        <LoginPage onLogin={setUser} />
      </GoogleOAuthProvider>
    );
  }

  const handleLogout = () => {
    localStorage.removeItem('user');
    setUser(null);
  };

  const isMobile = window.innerWidth <= 768;

  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <Router>
        <div className="app-container">
          {!isMobile && <Sidebar onLogout={handleLogout} />}
          <main className="main-content">
            <Header title="Sovereign" user={user} />
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/predict" element={<PredictPage user={user} />} />
              <Route path="/inferences" element={<InferencesPage />} />
              <Route path="*" element={<DashboardPage />} />
            </Routes>
            {isMobile && (
              <nav className="mobile-nav">
                <Link to="/" className="mobile-nav-item">Dashboard</Link>
                <Link to="/predict" className="mobile-nav-item">Predict</Link>
                <Link to="/inferences" className="mobile-nav-item">History</Link>
                <div onClick={handleLogout} className="mobile-nav-item">Exit</div>
              </nav>
            )}
          </main>
        </div>
      </Router>
    </GoogleOAuthProvider>
  );
}
