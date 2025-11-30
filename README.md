```text


██████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗       █████╗ ███╗   ██╗ ██████╗██╗███████╗███╗   ██╗████████╗
██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║      ██╔══██╗████╗  ██║██╔════╝██║██╔════╝████╗  ██║╚══██╔══╝
██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║█████╗███████║██╔██╗ ██║██║     ██║█████╗  ██╔██╗ ██║   ██║   
██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║╚════╝██╔══██║██║╚██╗██║██║     ██║██╔══╝  ██║╚██╗██║   ██║   
██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║      ██║  ██║██║ ╚████║╚██████╗██║███████╗██║ ╚████║   ██║   
╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝      ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
                                                                                                                   
██╗      █████╗ ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗  ██████╗ ███████╗███████╗                                      
██║     ██╔══██╗████╗  ██║██╔════╝ ██║   ██║██╔══██╗██╔════╝ ██╔════╝██╔════╝                                      
██║     ███████║██╔██╗ ██║██║  ███╗██║   ██║███████║██║  ███╗█████╗  ███████╗                                      
██║     ██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔══██║██║   ██║██╔══╝  ╚════██║                                      
███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║╚██████╔╝███████╗███████║                                      
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝                                      


```

# PythonAncientLanguages

> 📌 **Project status update (Nov 30, 2025)**  
> Development on this project will resume soon. Expect refactoring, dataset updates, and new model experiments.

> Deep Learning for the recognition and interpretation of Ancient Egyptian Hieroglyphs, Sumerian Cuneiform, and other ancient scripts.

This repository explores the application of **Deep Learning (Computer Vision + NLP)** to detect, recognize and ultimately translate ancient written languages such as:

- Egyptian Hieroglyphs  
- Sumerian Cuneiform  
- Mayan Glyphs *(future work)*  

The primary focus is **Egyptian hieroglyphics**, with the broader goal of building a general framework for ancient script recognition.

The project uses:
- Python 3  
- TensorFlow / Keras  
- Convolutional Neural Networks (CNNs)  
- Traditional Computer Vision techniques where applicable  

---

## 🔍 Overview

This project is inspired by the academic work of:

- Morris Franken (University of Amsterdam, 2013)
- His repository: https://github.com/morrisfranken/glyphreader

Franken’s work remains one of the strongest baselines for glyph recognition, but is limited by:
- small datasets
- lack of scalable annotation environments
- incomplete end-to-end pipelines

Other notable prior research includes:

- F. Gimbert’s project: https://github.com/fgimbert/Hieroglyphs *(inactive)*
- Barucci et al. (2021): High accuracy single-glyph classifiers  
- Ubisoft / Google Arts “Fabricius” project  
  - https://artsexperiments.withgoogle.com/fabricius  
  - Dataset not publicly released

While classification accuracy for individual glyphs reaches 96% in recent work, **real translation** remains unsolved due to:

- glyph segmentation complexity
- directionality
- grammar rules
- missing corpora for supervised training
- few publicly available labeled datasets

This project aims to move beyond recognition into:
- segmentation
- classification
- interpretation
- transliteration
- (eventually) translation

---

## 🚀 Quickstart

Currently implemented scripts:

### Feature extraction
```bash
python image_processing.py
```

### CNN training
```bash
python train_model.py
```

### Requirements
```bash
pip install -r requirements.txt
```

**Python 3.x** required.

---

## 🧭 Project Status

This repository is under active development again.

Current state:
- ✅ Feature extraction tool
- ✅ Basic CNN classification
- 🔄 Dataset curation ongoing
- 🔄 Model improvements planned
- 🔄 Refactor and modernization underway

Next milestones:
- Multi-glyph detection in real images
- Automatic segmentation
- Hieroglyph line parsing
- NLP pipeline integration
- Translation pipeline
- Annotator interface
- Cross-script generalization

---

## 🎯 Goals

- Glyph detection in real photographs
- Dataset tooling and augmentation
- Improved CNN / Transformer models
- Character-level recognition
- Sequence modeling
- Corpus integration
- Offline-first design (no cloud requirement)
- Public dataset building tools
- Open annotation environment

---

## 📦 Datasets

Public data is sparse and fragmented.

Known sources:

- Fayrose Middle Egyptian Dataset  
  https://github.com/fayrose/MiddleEgyptianDictionaryWebsite

- Gimbert dataset  
  https://github.com/fgimbert/Hieroglyphs/tree/master/hieroglyphs

- Google Arts Fabricius sample images  
  https://github.com/googleartsculture/workbench/tree/main/src/assets/images

- JSesh database  
  https://github.com/rosmord/jsesh

Note:  
Google’s training dataset (~50,000 traced glyphs) is not released.

More info:
See `public_datasets.md`.

---

## 🧪 Research References

- Franken et al. (2013)  
- Domingo et al. (2017)  
- Talwar (2017)  
- Wiesenbach & Riezler (2019)  
- Elnabawy et al. (2021)  
- Barucci et al. (2021)

*(Original list preserved)*

---

## 🤝 Contributing

This is a research-driven repository.

Contributions welcome:
- datasets
- annotation tools
- segmentation ideas
- model architecture experiments
- OCR pipelines
- NLP approaches
- domain expertise

---

## ⚠️ Disclaimer

This project is experimental research.

No claims of correctness or historical interpretation accuracy are made.

This is **not a commercial project**.

---

## 📚 Resources

- Transfer Learning in CNNs  
  https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

---

> “We decode the past using the tools of the future.”
