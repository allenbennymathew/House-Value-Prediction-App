# ⌂ Sovereign Intelligence Terminal: SETUP GUIDE

Follow these exact steps to run the application on any new system.

## Prerequisites (Install these first)
1. **Python 3.8+:** [python.org/downloads/](https://www.python.org/downloads/)
2. **Node.js (LTS):** [nodejs.org/](https://nodejs.org/)

---

## Step 1: Brain Setup (Backend)
1. Open a terminal in the `backend/` folder of this project.
2. Create and activate a Virtual Environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install necessary library dependencies:
   ```powershell
   pip install numpy pandas scikit-learn six fastapi uvicorn sqlalchemy pydantic
   ```
4. **Initialize Models & Database:**
   ```powershell
   python build_all.py
   ```

---

## Step 2: Body Setup (Frontend)
1. Open a **second** terminal window in the `frontend/` folder.
2. Install the necessary Node packages (this creates the `node_modules` folder):
   ```powershell
   npm install
   ```

---

## Step 3: Deployment (Running the App)
Both parts of the software must be running simultaneously for the terminal to function.

### Terminal 1: START THE BRAIN
```powershell
cd backend
.\venv\Scripts\activate
python -m uvicorn src.app:app --reload
```

### Terminal 2: START THE BODY
```powershell
cd frontend
npm run dev
```

---

## Access the Interface
Once both terminals are active, the app will be live at:
**[http://localhost:5173](http://localhost:5173)** (or as indicated in the Frontend terminal).
