# Deploy Crop Yield Prediction (Streamlit)

You need the app online in two parts: **code on GitHub**, then **hosting** (easiest: **Streamlit Community Cloud**, free).

## 1) Prepare files for GitHub

These files must be in the repo so the cloud server can load the model:

- `models/best_crop_yield_model.joblib`
- `models/model_metrics.json`

This project’s `.gitignore` is set so those two files **are not ignored** and can be committed.

If you already ran `git add` before, force-add once:

```bash
git add -f models/best_crop_yield_model.joblib models/model_metrics.json
```

Also commit:

- `app/streamlit_app.py`
- `data/raw/sample_crop_yield_data.csv`
- `requirements.txt`
- `.streamlit/config.toml`

## 2) Push to GitHub

1. Create a new repository on GitHub (example name: `crop-yield-prediction`).
2. In your project folder:

```bash
git init
git add .
git commit -m "Crop yield prediction Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Use a folder name **without spaces** on GitHub if you can (e.g. clone into `crop-yield-prediction`).

## 3) Deploy on Streamlit Community Cloud

1. Open [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick your repository and branch (`main`).
3. **Main file path:** `app/streamlit_app.py`
4. **Deploy.**

After a few minutes you get a public URL like `https://YOUR-APP.streamlit.app`.

### If the app says “Model not found”

Train locally, then commit and push again:

```bash
python src/train.py
git add models/best_crop_yield_model.joblib models/model_metrics.json
git commit -m "Add trained model for deployment"
git push
```

---

## Optional: Deploy on Render

1. Create a **Web Service** on [Render](https://render.com), connect the same GitHub repo.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:**

```bash
streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless true
```

4. Use the free tier; first deploy may take several minutes.

If you use the included `render.yaml`, Render can pick settings automatically when you choose “Blueprint”.

---

## Checklist before sharing your link

- [ ] `requirements.txt` installs without errors.
- [ ] `models/best_crop_yield_model.joblib` is in the repo and pushed.
- [ ] Main Streamlit file path is `app/streamlit_app.py` (Streamlit Cloud).
