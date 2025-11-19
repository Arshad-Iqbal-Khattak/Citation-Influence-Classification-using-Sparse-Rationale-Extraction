# 🧠 Interpretable Citation Influence Classification

### Sparse, Faithful Rationale Extraction using SciBERT + HardKuma-L₀

Understanding how influential a citation truly is requires more than
simply counting citations. While neural models classify citation intent
accurately, most operate as **black boxes**, limiting interpretability
in bibliometric studies that require transparency and explainability.

This project introduces an **interpretable-by-design deep learning
framework** for classifying whether an in-text citation is *influential*
or *non-influential*. The model integrates **SciBERT** with a
**HardKuma--L₀ sparsity-controlled selector**, which identifies the
*smallest possible set of informative words* within the citation
context---without any additional supervision. These selected tokens form
**faithful, human-readable rationales** that directly determine the
model's final decision.

Experiments on **ACL-ARC** and **SciCite** datasets achieve
**state-of-the-art F1 scores: 0.95 and 0.92**,
respectively---statistically significant improvements over baseline
models *(p \< 0.01, McNemar's test)*. Using the **ERASER benchmark**, we
further demonstrate that our rationales are more sufficient,
comprehensive, and faithful compared to attention-based explanations,
LIME, or SHAP.

The repository also includes an **interactive Streamlit dashboard**,
enabling researchers to explore predictions, view token-level
rationales, and adjust sparsity parameters in a human-in-the-loop
environment.

## 🔍 Key Features

-   **Interpretable by design** --- HardKuma-L₀ sparse rationale
    extraction\
-   **State-of-the-art citation influence performance**\
-   **Human-readable explanations** without extra annotations\
-   **Rationale faithfulness evaluated** on the ERASER benchmark\
-   **Interactive Streamlit interface** for easy exploration\
-   **Ablation studies** validating sparsity and structural cues\
-   **Modular, research-friendly codebase**

## 🧱 Model Architecture

    Citation Context Text
            │
            ▼
        SciBERT Encoder
            │
            ├──► HardKuma-L₀ Selector (Sparse Token Mask)
            │
            ▼
      Masked Representation
            │
            ▼
      Classification Layer
            │
            ▼
    Influential / Non-Influential
    + Token-Level Rationales



## 🚀 Getting Started

### 1. Clone Repository

``` bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
```

### 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3. Download SciBERT

``` bash
huggingface-cli download allenai/scibert_scivocab_uncased
```

### 4. Train the Model

``` bash
python training/train.py     --dataset acl_arc     --model scibert     --sparsity 0.07     --batch_size 8
```

### 5. Evaluate Performance

``` bash
python evaluation/eval_f1.py --dataset acl_arc
python evaluation/eval_rationale.py --dataset eraser
```

## 🧪 Ablation Experiments

``` bash
# Train without sparsity constraints
python training/train.py --disable_sparsity

# Train without structural text features
python training/train.py --disable_structure
```

## 🖥️ Streamlit Demo

``` bash
streamlit run demo/app.py
```

## 📈 Example Output

**Input Context:**\
\> "According to \[Smith et al., 2019\], the attention mechanism fails
when inputs are noisy."

**Prediction:** **Influential**\
**Extracted Rationales:**\
- **fails**\
- **noisy**\
- **attention mechanism**

## 📝 Citation

    @article{your2025citation,
      title={Interpretable Sparse Rationale Extraction for Citation Influence Classification},
      author={Your Name},
      year={2025}
    }

## 🤝 Contributions

Pull requests and suggestions are welcome.


