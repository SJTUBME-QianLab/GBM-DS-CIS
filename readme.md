# GBM-DS-CIS

This repository will hold the PyTorch code of our paper **Diversified Synthesis with Causal-Intervened Separation for Glioblastoma Progression Diagnosis**.

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Abstract

Post-treatment patients with glioblastoma (GBM) often develop pseudoprogression (PsP), a condition that visually resembles true tumor progression (TTP) yet necessitates a distinct therapeutic approach. Consequently, it is crucial to construct an automated model to distinguish between these two types of GBM progression. However, the reliability of models is suboptimal due to limited availability of patient data and presence of non-causal features, such as tumor-adjacent regions with lesion-like textures and tumor-internal areas with less discrimination. Therefore, we propose the Diversified Synthesis with Causal-Intervened Separation (DS-CIS) method for accurate and reliable GBM progression diagnosis. Specifically, first, a diversified synthesis strategy is proposed to enhance data diversity and extract discriminative features in lesion regions through multi-parametric spatial transformations and consistency constraints. Subsequently, a causal-intervened separation mechanism is designed, which utilizes constraints to separate causal and non-causal features, thereby enabling the GBM progression diagnosis based solely on causal features. Extensive validation on GBM clinical data demonstrates that the DS-CIS outperforms existing models and the visualization of its causal features aligns with the clinical foundations. The concepts of texture diversification and causal separation used in this method offer a valuable paradigm for effective medical imaging analysis. The code will be released at https://github.com/SJTUBME-QianLab/GBM-DS-CIS.


# 📂 Files

- **sig_cim.py**  
  Main Python script. Provides functionality for loading medical image dataset, training models, and performing inference using PyTorch.

- **sig_cnn.py**  
  Defines CNN model architectures (based on DS-CIS) and related utility functions. Includes modules implemented with PyTorch and some visualization tools.

- **sig_cnn.c**  
  C code generated by Cython from `sig_cnn.py`, improving computational efficiency. This file is typically generated by the build process and not meant for manual modification.

- **sig_cnn.cpython-37m-x86_64-linux-gnu.so**  
  Precompiled Python C extension module (for Linux, Python 3.7). It speeds up some computations.

- **util.py**  
  A collection of utility functions, including evaluation metrics (precision, recall, F1-score) and image/data processing utilities.

- **temp.linux-x86_64-3.7/**  
  Temporary build directory for compiled files. Usually created during packaging or extension compilation.


# ⚙️ Dependencies

Recommended Python version: **3.7 or higher**. Required Python packages include:

- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `Pillow`

Install them using:

```bash
pip install torch numpy scikit-learn matplotlib tqdm Pillow
```

# ✉ Contact

For any question, feel free to contact

```
Qiang Li : lq_1929@sjtu.edu.cn
```
