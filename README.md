# The Brain Watching Itself: Identifying Brain Tumors with Spiking Neural Networks

A research project investigating the application of Spiking Neural Networks (SNNs) for MRI brain tumor classification, exploring their potential as an efficient, biologically-inspired alternative to traditional convolutional neural networks.

## Overview

This project implements and evaluates Convolutional Spiking Neural Networks (CSNNs) for brain tumor detection from MRI images. The research demonstrates that SNNs can achieve competitive performance with traditional CNNs while using significantly less computational resources and memory.

**Key Findings:**
- Temporal encoding CSNN achieved **73.10% accuracy** 
- **73% memory savings** compared to traditional CNNs
- **13× faster inference** than conventional architectures
- Demonstrates biological plausibility for neural computation

## Dataset

The project uses the Brain Tumor MRI dataset with 3,264 human brain MRI images categorized into four classes:
- Glioma
- Meningioma  
- No tumor
- Pituitary tumor

Images are preprocessed from 512×512×3 to 150×150×3 dimensions with normalization to [0,1] range.

## Architecture

### Convolutional Spiking Neural Network (CSNN)
- **Convolutional Layers**: Two conv layers (32 and 64 filters) with LIF neuron activations
- **Pooling**: Average pooling after each convolutional layer
- **Fully Connected**: 128-neuron FC layer with dropout for regularization
- **Neuron Model**: Leaky Integrate-and-Fire (LIF) neurons
- **Time Window**: 100 timesteps

### Encoding Schemes

**Rate Encoding**: Information encoded by spike frequency relative to pixel intensity
```python
S_ij(t) = 1 if t < floor(I_ij * M), else 0
```

**Temporal Encoding**: Information encoded by precise spike timing (brighter pixels spike earlier)
```python
t_k = floor((1 - I_ij) * T) + epsilon_k
```
### Comparing with CNN Baselines

The project includes implementations of several CNN architectures for comparison:
- EfficientNetB0
- TumorDetNet  
- AlexNet
- DenseNet201


## Installation

```bash
git clone https://github.com/namita-ach/TheBrainWatchingItself.git
cd TheBrainWatchingItself

# Install required packages
pip install torch torchvision
pip install snntorch==0.9.4
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
```

## Results

### Performance Comparison

| Model | Accuracy | Memory (MB) | Inference Time (s) | CPU Usage |
|-------|----------|-------------|-------------------|-----------|
| **Temporal CSNN** | **73.10%** | **996.72** | **0.00721** | **8.8%** |
| Rate CSNN | 65.44% | 1123.78 | 0.00699 | 4.9% |
| DenseNet201 | 78.68% | 6892.12 | 0.45777 | 48.0% |
| AlexNet | 72.84% | 3676.35 | 0.09688 | 23.6% |
| EfficientNetB0 | 62.84% | 4310.96 | 0.16683 | 2.5% |

### Key Insights

1. **Temporal encoding significantly outperforms rate encoding** (73.10% vs 65.44%)
2. **Massive resource efficiency**: Up to 73% memory savings and 13× faster inference
3. **Competitive accuracy** with traditional CNNs while being more energy-efficient
4. **Biological plausibility** through event-driven spike-based processing

## Requirements

- Python 3.8+
- PyTorch 1.12+
- snntorch 0.9.4
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn

## Technical Details

### LIF Neuron Model
The Leaky Integrate-and-Fire neuron follows the differential equation:
```
τ_m * dV(t)/dt = -(V(t) - V_rest) + R*I(t)
```

Where the neuron spikes when `V(t) >= V_th` and resets to `V_reset`.

## Future Work

- Validation across multiple medical imaging datasets (BraTS, TCIA)
- Optimization of hybrid encoding techniques
- Transition to dedicated neuromorphic hardware platforms
- Integration into portable diagnostic devices

## Authors

- **Namita Achyuthan** - Dept. of CSE(AIML), PES University
- **Bhaskarjyoti Das** - Dept. of CSE(AIML), PES University


## Acknowledgments

- Brain Tumor MRI dataset from Sartaj Bhuvaji's repository
- snntorch library for SNN implementation
- PES University Department of Computer Science & Engineering (AI & ML)
