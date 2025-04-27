# Auxiliary future state generation framework using diffusion model


> 

---

## 📖 Table of Contents
- [🖼 Architecture](#-architecture)  
- [✨ Features](#-features)  
- [⚙️ Requirements](#️-requirements)  
- [🚦 Quick Start](#-quick-start)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  
- [📚 Citation](#-citation)

---

## 🖼 Architecture



---

## ⚙️ Requirements

Install dependencies:

1. use one conda env to install habitat-sim
2. use another one conda env to install unik3d


---

## 🚦 Quick Start



```bash
cd axground3d

# use habitat-sim env
python ./run_gen_real.py

# use unik3d env
python ./run_gen_ply.py

# use habitat-sim env
python ./run_gen_fake.py

# Evaluation
python ./fid_cal.py
```

---

## 🛠 Configuration

All hyperparameters live in `.default.yaml`. Make one `.local.yaml` first. All hyperparameters live in `.local.yaml`. Key sections:

``` yaml
mp3d_habitat_scene_dataset_path: "<your path>/mp3d/"
vlm_model_path: "./model/Janus-Pro-1B"
number_of_node_per_scene: 1
```
---

## 🤝 Contributing

1. Fork → Clone → Create feature branch  
2. Add tests for new modules  
3. Submit PR → Review → Merge  

---

## 📄 License

This project is MIT Licensed. See `LICENSE` for details.

---

## 📚 Citation

If you find our work helpful, feel free to give us a cite:
```
@misc{axground3D,
    title = {Auxiliary future state generation framework using diffusion model},
    url = {https://github.com/YichengDuan/axground3D},
    author = {Yicheng Duan, Duo Chen},
    month = {April},
    year = {2025}
}
```


