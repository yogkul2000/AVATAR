<h2 align="center"> <a href="https://arxiv.org/abs/2508.03100">AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video
</a></h2>
<div align="center">


<br>


<a href='https://arxiv.org/abs/2508.03100'><img src='https://img.shields.io/badge/arXiv-2508.03100-b31b1b.svg'></a> &nbsp;
 <a href='https://people-robots.github.io/AVATAR/'><img src='https://img.shields.io/badge/Project-Website-blue'></a>&nbsp;
 <a href='https://huggingface.co/yogkul2000/AVATAR'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 

 
 </div>

## Abstract
Multimodal reasoning over long-horizon video is challenging due to the need for precise spatiotemporal fusion and alignment across modalities. While recent methods such as Group Relative Policy Optimization (GRPO) have shown promise in this domain, they suffer from three key limitations: (1) data inefficiency from their on-policy design, (2) a vanishing advantage problem, where identical or near-identical rewards within a group eliminate the learning signal by producing zero-valued advantages, and (3) uniform credit assignment that fails to emphasize critical reasoning steps.

We introduce AVATAR (Audio-Video Agent for Alignment and Reasoning), a framework that addresses these limitations through two core components: (1) an off-policy training architecture that improves sample efficiency and resolves vanishing advantages by reusing past experiences with greater reward diversity, and (2) Temporal Advantage Shaping (TAS), a novel credit assignment strategy that upweights key reasoning phases during learning.

AVATAR achieves strong performance across various benchmarks, outperforming the Qwen2.5-Omni baseline by +5.4 on MMVU, +4.9 on OmniBench, and +4.5 on Video-Holmes, while demonstrating over 35% higher sample efficiency. These results demonstrate that targeted RL improvements, rather than massive architectural changes, effectively address core multimodal reasoning challenges.

<table class="center">
    <tr>
    <td><img src="assets/avatar.png" alt="AVATAR Overview Diagram"></td>
    </tr>
    <tr>
    <td align="center"><em>Overview of the AVATAR.</em></td>
    </tr>
</table>

## 🧰 TODO
- [x] Release Paper.
- [x] Release AVATAR reasoning model fine-tuned model weights.
- [x] Release Inference Code.
- [x] Release eval scripts for all benchmarks.
- [x] Stage wise training data.
- [x] Release GRPO Trainer with TAS.

## 📦 Install

### Environment Setup

```bash
conda create -n avatar python=3.10
conda activate avatar

# Install PyTorch with CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install flash attention (if facing issues, use the command below)
pip install flash-attn==2.7.4.post1

# If flash-attn installation fails, try:
pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install transformers==4.54.1

# Install other dependencies
pip install decord opencv-python pillow numpy
pip install qwen-omni-utils[decord] -U
```

### Individual Benchmark Evaluation
All eval for benchmarks in eval folder. Paths hardcoded for now.

## 📝 Citation
If you find AVATAR useful for your research, please cite our paper:
```bib
@article{kulkarni2025avatar,
  title={AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video},
  author={Kulkarni, Yogesh and Fazli, Pooyan},
  journal={arXiv preprint arXiv:2508.03100},
  year={2025}
}
```

## 📪 Contact
For questions about the paper, please contact Yogesh Kulkarni at `ykulka10@asu.edu`. You can also open an issue in this GitHub repository for bugs or specific questions related to the code.

