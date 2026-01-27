# [AAAI 2026] UniMo: Unified Motion Generation and Understanding with Chain of Thought
<p align="center">
  <a href="https://arxiv.org/abs/2601.12126"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://aaai.org/conference/aaai/aaai-26/"><img src="https://img.shields.io/badge/AAAI-2026-4b44ce.svg" alt="Conference"></a>
</p>
<p align="center">
  <b>Guocun Wang<sup>1*</sup>, Kenkun Liu<sup>2*</sup>, Jing Lin<sup>3</sup>, Guorui Song<sup>1</sup>, Jian Li<sup>1†</sup>, Xiaoguang Han<sup>2†</sup></b><br>
  <sup>1</sup>Tsinghua University, <sup>2</sup>CUHK-Shenzhen, <sup>3</sup>Nanyang Technological University
</p>
UniMo is a novel unified framework for 3D human motion generation and understanding that leverages the powerful reasoning capabilities of large language models (LLMs) enhanced by Chain of Thought (CoT) and Group Relative Policy Optimization (GRPO) based reinforcement learning. UniMo bridges the gap between natural language and human motion, achieving state-of-the-art performance on both text-to-motion (T2M) and motion-to-text (M2T) tasks.

<p align="center">
  <img src="assets/UniMo.jpg" alt="UniMo Framework" width="95%"/>
</p>

---
## 📋 TODO

- [x] 🚀 Release training code
- [x] 🔧 Release inference code
- [ ] 📦 Release pre-trained model weights
      
## Installation

1. **Create the Conda environment:**

   ```bash
   conda env create -f environment.yaml
   ```

---

## Pretrained Weight Preparation

Download the required pretrained models following the steps below:

1. **Download VQ-VAE, Glove, and extractor weights**
   
   Refer to [Motion-Agent repository](https://github.com/szqwu/Motion-Agent.git) for download links, or simply run:

   ```bash
   bash prepare/download_ckpt.sh
   bash prepare/download_glove.sh
   bash prepare/download_extractor.sh
   ```

2. **Download Qwen2.5-3B-Instruct**
   
   Place the official [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model in `./Qwen2.5-3B-Instruct`.

---

## Dataset Preparation

1. **HumanML3D Dataset**
   
   Download the [HumanML3D dataset](https://github.com/EricGuo5513/HumanML3D.git) and place it in:

   ```
   ./dataset/HumanML3D
   ```

2. **Text Reasoning Data**
   
   Unzip `./dataset/texts_think_qwen.zip`, and move the extracted `texts_think_qwen` directory to:

   ```
   ./dataset/HumanML3D/texts_think_qwen
   ```

---

## Training & Evaluation Pipeline

### SFT Stage

1. **Encode Motions with VQ-VAE:**

   ```bash
   python get_motion_code.py
   ```

   The encoded motion tokens will be saved to `./dataset/HumanML3D/motionagent_vqvae`.

2. **Train the SFT Model:**

   ```bash
   python train_sft.py
   ```

### GRPO Stage

1. **Preprocess Training Data:**

   ```bash
   python get_train_data.py
   python pkl2json.py
   ```

   PKL files are saved in `./dataset/HumanML3D/tmp_think`; the converted JSON will be at `./motion_train.json`.

2. **GRPO Training:**

   ```bash
   deepspeed --num_gpus=4 train_grpo.py --deepspeed_config ds_config.json
   ```

   Ensure that the `model_name` in the code is set to the checkpoint from the SFT stage.

---

### Evaluation

* **T2M Evaluation:**

  ```bash
  python eval_mllm_think_t2m.py
  python eval_mllm_think_t2m_multimodality.py
  ```

  Set `args.llm_backbone` in these scripts to your final checkpoint.

* **M2T Evaluation:**

  ```bash
  python eval_mllm_think_m2t.py
  python create_m2t_gt.py
  python eval_m2t.py
  ```

  Similarly, set `args.llm_backbone` to your trained checkpoint.

---

### Demo

To run a quick demo, simply use:

```bash
python demo.py
```
Similarly, set `args.llm_backbone` to your trained checkpoint.

---


## Acknowledgements

* [Motion-Agent](https://github.com/szqwu/Motion-Agent)
* [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
* [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## Citation
If you find UniMo useful for your research, please cite our paper:

```bibtex
@article{wang2026unimo,
  title={UniMo: Unified Motion Generation and Understanding with Chain of Thought},
  author={Wang, Guocun and Liu, Kenkun and Lin, Jing and Song, Guorui and Li, Jian and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2601.12126},
  year={2026}
}
```

