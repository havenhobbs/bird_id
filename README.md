# bird id platform

a deep-learning platform that identifies bird species from user-uploaded images. the prject utilizes transfer learning with **MobileNetV3Large** via tensorflow, exports the optimized computational graph into **ONNX Runtime**, and serves a clean web interface using **Streamlit**.

the model is trained on the **CUB-200-2011** datasetfeaturing 200 distinct bird species primarily native to North America. 

---

## project architecture

```text
bird_id/
├── app/
│   └── model/
│       └── bird_id.onnx       # optimized production engine
├── data/
│   ├── raw/                   # extracted CUB-200 image files
│   └── processed/             # clean train.csv and val.csv logs
├── models/
│   └── best_bird_model.keras  # native TensorFlow training checkpoints
├── src/
│   ├── download_data.py       # dataset pipeline streaming
│   ├── data_prep.py           # regular-expression dataset builder
│   ├── utils.py               # albumentations multi-threaded loader
│   ├── train.py               # deep-learning training loops
│   └── predict.py             # ONNX structural format converter
├── .gitignore                 # block tracking for heavy file payloads
├── app.py                     # main Streamlit Web Application Interface
└── README.md                  # system documentation guide
```

---

## setup & installation

### 1. clone + navigate to project
```bash 
cd bird_id
```

### 2. configure virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. install core system dependencies
```bash
pip install tensorflow onnxruntime streamlit opencv-python albumentations pandas matplotlib tqdm tf2onnx
```

---

## pipeline execution flow

to download, train, compile, and run the interface from scratch, execute the following commands sequentially from your project root: 

### step 1: download target dataset
streams the compressed archive files from the hosted dataset mirror
```bash
python src/download_data.py
```

### step 2: extract + prepare metadata records
uses string regular expressions to structure image paths and maps 0-indexed categorical numerical mapping
```bash
python -m src.data_prep
```

### step 3: train deep learning neural network
runs transfer learning wiht MobileNetV3Large to fit classification weights over the training dataset matrix
```bash
python -m src.train
```

### step 4: export to ONNX runtime
serializes and encodes the trained keras network graph directly into optimized `.onnx` production formatting
```bash
python -m src.predict
```

### step 5: boot the Streamlit web application ui
spins up a lightweight local development web server to run your browser interface
```bash streamlit run app/app.py
```

---

## dataset scope disclaimer
this platform is strictly trained on the **CUB-200-2011 dataset**, containing 200 specific bird species (e.g., *Northern Cardinal, Blue Jay, Baltimore Oriole, Laysan Albatross*)
* exotic birds outside this taxonomy list (e.g., *Toucans*) or missing birds (*Bluebirds*) will match with the closest visual equivalent
* if a guess yields low classification confidence (<55%), the UI interface flags a native uncertainty warning helper text on-screen
