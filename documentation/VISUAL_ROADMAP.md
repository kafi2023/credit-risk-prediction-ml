# Visual Roadmap for ELTE Thesis: Credit Risk Prediction

## Complete Screenshot Capture Guide

---

## CHAPTER 1: Introduction and Motivation

### ✓ Status: NO IMAGES REQUIRED

Chapter 1 is conceptual/theoretical and does NOT require screenshots. It focuses on background, motivation, and problem framing. No visual artifacts are expected here.

---

## CHAPTER 2: User Documentation

### **Total Images Needed: 8**

---

### **IMAGE 2.1: Python Version & Dependency Check**

**Section:** 2.1 System Requirements  
**Where:** After paragraph ending with "...memory headroom."

**The "WHERE":**

- Exact location: After the sentence "Software prerequisites include Python 3.10 or newer, pip package management, and a modern web browser..."
- This is the first IMAGE PLACEHOLDER in section 2.1

**The "HOW" (CLI Command):**

```bash
# Step 1: Open terminal in the project root
cd "c:\Users\kafis\OneDrive - Eotvos Lorand Tudomanyegyetem\Asztal\Kafi Thesis\credit-risk-prediction-ml"

# Step 2: Run Python version check
python --version

# Step 3: Run pip list to show installed packages (shows Flask, scikit-learn, shap, etc.)
pip list

# Step 4: Check available disk space (Windows)
# Alternative on Windows:
fsutil volume diskfree C:\

# Step 5: Take SCREENSHOT showing:
# - Terminal window
# - Python version output (3.10+)
# - At least 10 rows of pip list output showing: Flask, scikit-learn, xgboost, shap, pandas, numpy
```

**The "WHAT" (Visual Description):**

- Terminal showing "Python 3.11.x" or higher
- `pip list` output clearly showing all required libraries
- Screen should be **1280x720 or larger**, readable text

**Academic Labeling:**

```
Figure 2.1: Python Environment and Dependency Validation
Caption: Terminal output showing Python 3.11 and successfully installed dependencies including Flask, scikit-learn, XGBoost, SHAP, and LIME. This confirms system readiness for deployment.
```

---

### **IMAGE 2.2: Virtual Environment & Installation Sequence**

**Section:** 2.2 Installation and Setup  
**Where:** After first 2 paragraphs, before "Environment variables are optional..."

**The "HOW" (CLI Command):**

```bash
# Step 1: Show environment creation
python -m venv .venv

# Step 2: Activate environment (Windows)
.venv\Scripts\Activate.ps1

# Step 3: Upgrade pip
python -m pip install --upgrade pip

# Step 4: Install requirements
pip install -r requirements.txt

# Step 5: Take SCREENSHOT showing:
# - Terminal with activated virtual environment (note the '(.venv)' prefix)
# - Successfully completed pip install output
# - Final line showing "Successfully installed [X] packages"
```

**The "WHAT" (Visual Description):**

- Terminal window showing:
  - Virtual environment activation (the `(.venv)` prefix visible)
  - pip upgrade command output
  - Partial requirements.txt installation (last 20 lines showing successful completion)

**Academic Labeling:**

```
Figure 2.2: Virtual Environment Setup and Dependency Installation
Caption: Terminal output showing successful creation and activation of isolated Python environment, followed by pip dependency installation from requirements.txt. The (.venv) prefix confirms isolated environment activation.
```

---

### **IMAGE 2.3: Data Preparation & Model Training Output**

**Section:** 2.2 Installation and Setup  
**Where:** Third IMAGE PLACEHOLDER in section 2.2

**The "HOW" (CLI Command):**

```bash
# Step 1: Run preprocessing
cd credit-risk-prediction-ml
python -m src.preprocessing.preprocessor

# Step 2: Wait for output showing:
# "✅ Done!"
# X_train, X_test, y_train, y_test shapes
# Feature count
# Preprocessor saved path

# Step 3: Run model training
python -m src.training.train_models

# Step 4: Take SCREENSHOT showing:
# - All output from preprocessing (shapes, feature count)
# - All output from model training (training progress for 3 models)
# - Final summary showing models saved to models/saved_models/

# Optional: Verify saved files exist
ls models/saved_models/
```

**The "WHAT" (Visual Description):**

- Terminal showing:
  - Preprocessing output: `X_train: (800, 57)`, `X_test: (200, 57)`, feature count
  - Training output: Progress for Logistic Regression, Random Forest, XGBoost
  - Model save confirmations (e.g., "→ Saved logistic_regression.joblib")
  - Final summary: "All models trained and saved"

**Academic Labeling:**

```
Figure 2.3: Data Preprocessing and Model Training Execution
Caption: Terminal output from data preprocessing pipeline (showing train/test shapes and feature engineering) followed by training of three models (Logistic Regression, Random Forest, XGBoost) with persistence confirmation. This demonstrates reproducibility and artifact generation.
```

---

### **IMAGE 2.4: Flask Application Startup & Landing Page**

**Section:** 2.2 Installation and Setup  
**Where:** Fourth IMAGE PLACEHOLDER in section 2.2

**The "HOW" (Web Navigation):**

```bash
# Step 1: Start Flask server (in terminal)
python web/app.py

# Output will show: "Running on http://127.0.0.1:5000"

# Step 2: Open web browser (Chrome, Edge, or Firefox)
# Navigate to: http://localhost:5000

# Step 3: Wait for page to fully load (3-5 seconds)

# Step 4: Take SCREENSHOT showing:
# - Full browser window
# - Page header and title
# - Model selection dropdown (populated with: random_forest, logistic_regression, xgboost)
# - Beginning of the dynamic input form with several fields visible
# - Do NOT submit a prediction yet — just show the landing state
```

**The "WHAT" (Visual Description):**

- Browser window showing:
  - Page title: "Credit Risk Prediction System"
  - Model selector dropdown with 3 options visible
  - Form fields: "Duration (Months)", "Credit Amount", "Installment Rate", etc.
  - Clean, white background, Times New Roman font
  - Left-side form panel only (right result panel not visible yet)

**Academic Labeling:**

```
Figure 2.4: Application Landing Page and User Interface
Caption: Web interface of the credit risk prediction system showing the initial state: model selector dropdown (populated with three trained models) and the dynamically generated input form with numerical and categorical fields. The interface demonstrates schema-driven form generation from the backend.
```

---

### **IMAGE 2.5: Model Dropdown Menu Expanded**

**Section:** 2.3 Interface Overview (or 2.4.1 Module A)  
**Where:** After section 2.4.1 "Module A: Model Selection"

**The "HOW" (Web Navigation):**

```bash
# Step 1: Application should still be running from IMAGE 2.4

# Step 2: In the browser, click on the "Select Model" dropdown

# Step 3: Wait for dropdown to expand (should show 3 options):
# - random_forest
# - logistic_regression
# - xgboost

# Step 4: Take SCREENSHOT showing:
# - Dropdown fully expanded with all 3 model names visible
# - Cursor hovering over one option (e.g., random_forest)
# - Dropdown menu clearly readable
```

**The "WHAT" (Visual Description):**

- Browser showing:
  - Model selector dropdown in expanded state
  - Three model options clearly listed: random_forest, logistic_regression, xgboost
  - Visual highlight/hover effect on one of the options

**Academic Labeling:**

```
Figure 2.5: Model Selection Interface
Caption: Dropdown menu showing the three available trained models: Random Forest, Logistic Regression, and XGBoost. Users can dynamically switch between models to compare predictions and explanations on the same input data.
```

---

### **IMAGE 2.6: Populated Input Form**

**Section:** 2.4.2 Module B: Dynamic Input Form

**The "HOW" (Web Navigation):**

```bash
# Step 1: Application still running

# Step 2: Select "random_forest" from the model dropdown

# Step 3: Fill in the form with sample data. Use these values:
#   Duration (Months): 24
#   Credit Amount: 5000
#   Installment Rate: 2
#   Age: 35
#   Residence Years: 3
#   Num Existing Credits: 1
#   Num Dependents: 2
#   (For categorical fields, select the first valid option in each dropdown)

# Step 4: Scroll down to see all fields filled

# Step 5: Take SCREENSHOT showing:
# - Multiple form fields with values entered
# - At least 3 numerical inputs with values visible
# - At least 2 categorical dropdowns with selections visible
# - The "Predict Credit Risk" button at the bottom
```

**The "WHAT" (Visual Description):**

- Browser showing:
  - Form with mixed input types (number fields and select dropdowns)
  - Sample data clearly visible in fields
  - All required fields filled (no validation errors)
  - "Predict Credit Risk" button visible and ready to click

**Academic Labeling:**

```
Figure 2.6: Input Form with Sample Data
Caption: Populated credit application form showing 7 numerical fields (duration, credit amount, age, etc.) and categorical fields (employment, housing type, etc.). The form is dynamically generated from the backend schema, ensuring consistency between frontend and backend validation rules.
```

---

### **IMAGE 2.7: Prediction Result & SHAP Explanation Chart**

**Section:** 2.4.4 Module D & 2.4.5 Module E

**The "HOW" (Web Navigation):**

```bash
# Step 1: Form from IMAGE 2.6 should still be populated

# Step 2: Click the "Predict Credit Risk" button

# Step 3: Wait for prediction to complete (2-5 seconds)
# You will see the button change to "Predicting..." briefly

# Step 4: Once complete, the right-side panel will appear showing:
# - Risk decision banner (e.g., "HIGH RISK" or "LOW RISK")
# - Confidence percentage
# - A horizontal bar chart with SHAP feature contributions

# Step 5: Take SCREENSHOT showing:
# - Both the left form panel AND the right result panel
# - Risk banner clearly visible
# - SHAP chart with at least 5 feature names and colored bars
# - The full page width so both panels are visible
```

**The "WHAT" (Visual Description):**

- Browser showing:
  - Left panel: The submitted form (unchanged)
  - Right panel (newly visible):
    - Risk decision banner with color (e.g., green for LOW, red for HIGH)
    - Confidence percentage (e.g., 75%)
    - Horizontal bar chart showing feature contributions
    - Feature names readable on Y-axis
    - Positive (red) and negative (blue) bars distinguished

**Academic Labeling:**

```
Figure 2.7: Prediction Result with SHAP Explanation
Caption: Output of credit risk prediction showing (left) the submitted applicant data and (right) the decision result and SHAP feature importance chart. The chart displays top contributing factors with color-coded impact direction: positive contributions increase predicted risk (red), negative contributions decrease risk (blue). This enables stakeholders to understand and validate model decisions.
```

---

### **IMAGE 2.8: Validation Error Message**

**Section:** 2.4.6 Module F: Error Handling & Validation Feedback

**The "HOW" (Web Navigation):**

```bash
# Step 1: Refresh the page (Ctrl+R)
# The form should return to empty state

# Step 2: Fill in ONLY some fields (intentionally leave required field empty):
#   Duration: 24
#   (Leave "Credit Amount" EMPTY)
#   Age: 35
#   (Leave all other fields empty)

# Step 3: Click "Predict Credit Risk" button

# Step 4: An error alert/message should appear at the top saying:
# "Error: Missing required field: 'credit_amount'"
# or similar validation message

# Step 5: Take SCREENSHOT showing:
# - The error message/alert banner clearly visible
# - The form with partially filled data
# - Red or warning-style formatting on the error
```

**The "WHAT" (Visual Description):**

- Browser showing:
  - Error alert banner (red or orange background)
  - Clear error message listing missing or invalid fields
  - Form still visible in background
  - Error message readable and professional-looking

**Academic Labeling:**

```
Figure 2.8: Input Validation and Error Feedback
Caption: User interface response to incomplete form submission. The system provides structured validation feedback identifying the missing required field ("credit_amount") and prevents prediction execution, ensuring data quality at the input stage.
```

---

## CHAPTER 3: Developer Documentation

### **Total Images Needed: 2**

---

### **IMAGE 3.1: System Architecture Diagram**

**Section:** 3.1 System Architecture  
**Where:** After paragraph ending with "...avoids route-level code duplication."

**The "HOW" (Create a Diagram):**
You have two options:

**Option A: Draw in PowerPoint/Draw.io (Recommended for Academic)**

```
Elements to include:
1. Frontend Layer: "HTML/CSS/JavaScript Form" box
2. Flask Routes Layer: Boxes for "/" "/schema" "/models" "/predict" "/health"
3. Prediction Service: "CreditRiskPredictor" box with methods
4. Preprocessing: "ColumnTransformer" + "fit_preprocessor.joblib"
5. Models: Three boxes for LogisticRegression, RandomForest, XGBoost
6. Explainability: "SHAP Explainer" and "LIME Explainer"
7. Data Flow arrows showing request → response → explanation

Color scheme: Light blue for UI, green for business logic, yellow for models, orange for explanations

Size: 1000x700 pixels minimum
```

**Option B: Use Python matplotlib to generate (Advanced)**

```bash
# Create a file: documentation/generate_architecture_diagram.py
# Use matplotlib or graphviz to auto-generate the diagram
# Then screenshot or export as PNG

# For now, create a professional diagram using:
# - PowerPoint (File > Save As > PNG)
# - draw.io (File > Export as > PNG)
# - Lucidchart or similar

# Save as: documentation/architecture_diagram.png
```

**The "WHAT" (Visual Description):**

- Diagram showing:
  - Layered architecture (Frontend → Routes → Services → Models → Data)
  - Component boxes with clear labels
  - Arrows showing data flow (request → processing → response)
  - Color coding for different logical layers
  - Explicit singleton pattern for model caching
  - SHAP/LIME modules clearly visible

**Academic Labeling:**

```
Figure 3.1: System Architecture and Data Flow
Caption: Layered architecture diagram of the credit risk prediction system. The frontend (HTML/CSS/JavaScript) communicates with Flask routes, which delegate to the CreditRiskPredictor service layer. This service coordinates input validation, preprocessing via ColumnTransformer, model inference, and explainability computation via SHAP. Models and preprocessor are persisted as joblib artifacts for training-inference consistency. Arrows indicate request flow (solid) and response flow (dashed).
```

---

### **IMAGE 3.2: Data Transformation Pipeline Diagram**

**Section:** 3.3 Data Model and Structures  
**Where:** After paragraph ending with "...typed transformation graph from user payload to interpreted decision output."

**The "HOW" (Create a Diagram):**

```
Elements to show:
1. Raw Input: Box "Raw JSON Input" with example field names
2. Validation: "Input Validator" box
3. DataFrame: "Pandas DataFrame" with 20 columns
4. Preprocessing:
   - Numerical Pipeline: "Imputer → StandardScaler"
   - Categorical Pipeline: "Imputer → OneHotEncoder"
5. Transformed Matrix: "Transformed Feature Matrix (N, 57)"
6. Model: "ML Model (RF/LR/XGBoost)"
7. Output: "Prediction + Probability"
8. Explanation: "SHAP Values" box

Show parallel processing (numerical vs categorical) with separate paths that merge before model input.

Size: 1000x600 pixels
```

**The "WHAT" (Visual Description):**

- Diagram showing:
  - Left side: Raw 20 input features (7 numerical, 13 categorical)
  - Middle: Preprocessing branches (numerical imputation & scaling, categorical encoding)
  - Right side: Transformed feature matrix (57 dimensions after one-hot encoding)
  - Below: Model inference and SHAP explanation
  - Clear labels on each transformation step
  - Dimensionality changes clearly marked

**Academic Labeling:**

```
Figure 3.2: Data Transformation Pipeline
Caption: End-to-end data transformation from raw user input to model prediction and explanation. Raw 20-feature input splits into parallel pipelines: numerical features (7) undergo median imputation and standardization; categorical features (13) undergo constant imputation and one-hot encoding. These merge into a 57-dimensional feature matrix suitable for model inference. Output includes prediction, probability, and SHAP-based explanation values for transparency.
```

---

## CHAPTER 4: Conclusion

### **Status: NO IMAGES REQUIRED**

Chapter 4 is evaluative and reflective. No screenshots or technical diagrams are expected. The text stands on its own.

---

## SUPPLEMENTARY GUIDELINES FOR ACADEMIC IMAGE INSERTION

### **Figure Numbering Scheme (ELTE Standard):**

```
Chapter 2: Figure 2.1, Figure 2.2, ..., Figure 2.8
Chapter 3: Figure 3.1, Figure 3.2
Chapter 4: (No figures)
```

### **Where to Place Images in Word Document:**

1. **Before insertion:** Right-click at the end of the relevant paragraph
2. **Select:** Insert → Picture → Choose your screenshot/diagram file
3. **Resize:** Make image width = page width minus margins (approximately 5.5 inches)
4. **Anchor:** Right-click image → Wrap Text → "Square" (allows text to continue after)
5. **Caption:** Right-click image → Insert Caption
   - Format: `Figure 2.1: [Title from list above]`
   - Position: Below image
   - Font: Times New Roman, 11pt, bold for figure number, normal for title

### **Caption Format (ELTE Standard):**

```
**Format:**
Figure X.Y: [Descriptive Title]
[1-2 sentences explaining what the screenshot shows and its relevance to the thesis]

**Example:**
Figure 2.7: Prediction Result with SHAP Explanation
Output of credit risk prediction showing the decision result (HIGH/LOW risk with confidence) and SHAP feature importance chart. The chart displays top contributing factors with color-coded impact direction: positive contributions increase predicted risk (red), negative contributions decrease risk (blue).

**Style:**
- Font: Times New Roman, 10pt
- Alignment: Centered below image
- Single-spaced
- Italicized optional but not required
```

### **Image Quality Requirements (ELTE Standard):**

- **Resolution:** Minimum 150 DPI for printing, 96 DPI for screen
- **Size:** 1000x700 pixels or larger (for readability in document)
- **Format:** PNG or JPEG (PNG preferred for diagrams, JPEG for screenshots)
- **Text in screenshots:** Must be clearly readable at document size
- **Color:** Full color acceptable for all images

### **Ordering Checklist:**

```
Before you start taking screenshots, prepare:

□ Terminal ready (repo cloned, venv created, dependencies installed)
□ Flask application ready to start
□ Web browser ready (Chrome/Edge/Firefox at 1280x720+)
□ Screenshot tool ready (Windows Snip & Sketch, ShareX, or built-in)
□ Diagram tool ready (PowerPoint, draw.io, or similar)

Capture order:
1. Ch 2.1 - Python version & pip list (CLI)
2. Ch 2.2 - Virtual env setup (CLI)
3. Ch 2.3 - Data prep & training (CLI)
4. Ch 2.4 - Flask landing page (Web)
5. Ch 2.5 - Model dropdown (Web)
6. Ch 2.6 - Populated form (Web)
7. Ch 2.7 - Prediction result (Web) ← MOST IMPORTANT
8. Ch 2.8 - Validation error (Web)
9. Ch 3.1 - Architecture diagram (Draw)
10. Ch 3.2 - Data pipeline diagram (Draw)
```

### **Tips for Professional Screenshots:**

- Maximize contrast (dark text on light background)
- Crop out unnecessary UI elements (taskbars, notifications)
- Use 1280x720 minimum resolution
- Make sure command output is complete (show success messages)
- For web: Use developer zoom to adjust if needed (Ctrl+Plus/Minus)
- Test images in the Word document before final submission

---

## NEXT STEPS:

1. Use this roadmap to capture each screenshot in order
2. Save images in: `documentation/screenshots/` folder
3. Insert each image into the Word document at the exact location specified
4. Add captions using the format provided above
5. Update Figure references in text if needed (e.g., "see Figure 2.7")

**Questions?** Refer back to the specific "HOW" section for step-by-step CLI commands or web navigation instructions.

---

**Document Version:** 1.0  
**Created:** April 29, 2026  
**Status:** Ready for screenshot capture
