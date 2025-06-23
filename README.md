# Poetry VAE: A Variational Autoencoder for Diverse Poetry Generation

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A deep learning implementation of a Variational Autoencoder (VAE) for generating diverse and creative poetry using PyTorch. This project leverages LSTM-based encoder-decoder architecture with self-attention mechanisms to learn the latent representations of poetic text and generate novel poems.

## 🎯 Overview

This project implements a sophisticated Poetry VAE that can:
- Learn latent representations of poetry from the Poetry Foundation dataset
- Generate diverse, novel poems with controllable creativity
- Visualize the latent space of learned poetry representations
- Support various generation techniques including temperature sampling and beam search

## ✨ Features

- **LSTM-based VAE Architecture**: Bidirectional LSTM encoder with unidirectional LSTM decoder
- **Self-Attention Mechanism**: Enhanced context understanding through self-attention layers
- **Advanced Generation**: Multiple sampling strategies (temperature, top-k, nucleus sampling)
- **Latent Space Visualization**: t-SNE and PCA visualizations of learned poetry embeddings
- **Comprehensive Training**: KL annealing, gradient clipping, and teacher forcing
- **Evaluation Metrics**: Perplexity calculation and generation quality assessment

## 🏗️ Architecture

The Poetry VAE consists of:

1. **Encoder**: Bidirectional LSTM + Self-Attention → Latent space (μ, σ)
2. **Latent Space**: 128-dimensional continuous representation
3. **Decoder**: LSTM with latent context injection → Poetry generation
4. **Loss Function**: Reconstruction loss + β-weighted KL divergence

```
Input Poetry → Encoder → Latent Space (μ, σ) → Reparameterization → Decoder → Generated Poetry
```

## 📊 Dataset

- **Source**: Poetry Foundation dataset from Kaggle
- **Size**: Thousands of poems with metadata (title, author, tags)
- **Preprocessing**: Tokenization, vocabulary building, sequence padding
- **Split**: 60% training, 20% validation, 20% testing

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn nltk tqdm
pip install kagglehub
```

### Running on Google Colab

This project is optimized for **Google Colab with T4 GPU**. Simply upload the notebook and run all cells:

1. Open the notebook in Google Colab
2. Ensure GPU runtime is selected (Runtime → Change runtime type → GPU → T4)
3. Run all cells to download data, train the model, and generate poetry

### Local Setup

```bash
# Clone the repository
git clone https://github.com/LaibaBatool01/Poetry_VAE_A_Variational_Autoencoder_for_Diverse_Poetry_Generation_-genAi_Proj-.git
cd Poetry_VAE_A_Variational_Autoencoder_for_Diverse_Poetry_Generation_-genAi_Proj-


# Run the notebook
jupyter notebook Poetry_VAE_A_Variational_Autoencoder_for_Diverse_Poetry_Generation_(genAi_Proj).ipynb
```

## 🔧 Configuration

Key hyperparameters (configurable in the notebook):

```python
EMBEDDING_DIM = 256        # Word embedding dimension
HIDDEN_DIM = 512          # LSTM hidden dimension
LATENT_DIM = 128          # Latent space dimension
NUM_LAYERS = 2            # Number of LSTM layers
DROPOUT = 0.5             # Dropout rate
BATCH_SIZE = 32           # Training batch size
EPOCHS = 20               # Training epochs
LEARNING_RATE = 3e-4      # Learning rate
MAX_SEQUENCE_LENGTH = 150 # Maximum poem length
BETA = 1.0                # KL divergence weight
```

## 📈 Training

The model uses advanced training techniques:

- **KL Annealing**: Gradual increase of β parameter for stable training
- **Teacher Forcing**: Probabilistic use of ground truth during training
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Based on validation perplexity

### Training Progress Visualization

The notebook includes real-time plotting of:
- Training and validation loss
- Reconstruction loss vs KL divergence
- Perplexity metrics
- Generated sample quality over epochs

## 🎨 Generation

Multiple generation strategies are supported:

### Temperature Sampling
```python
# Generate with different creativity levels
conservative_poem = model.sample(temperature=0.7)  # More conservative
creative_poem = model.sample(temperature=1.2)      # More creative
```

### Conditional Generation
```python
# Generate poems with specific characteristics
samples = generate_poems(model, idx2word, num_samples=5, temperature=1.0)
```

## 📊 Evaluation & Visualization

### Latent Space Analysis
- **t-SNE Visualization**: 2D projection of learned poetry embeddings
- **PCA Analysis**: Principal component analysis of latent representations
- **Clustering**: Automatic discovery of poetry themes and styles

### Metrics
- **Perplexity**: Measures model's uncertainty in predictions
- **BLEU Score**: Evaluation against reference poems (if available)
- **Diversity Metrics**: Vocabulary richness and n-gram diversity

## 🛠️ Hardware Requirements

- **Recommended**: Google Colab T4 GPU (free tier)
- **Minimum**: 8GB RAM, modern CPU
- **Training Time**: ~2-4 hours on T4 GPU for 20 epochs

## 📁 Project Structure

```
poetry-vae/
├── Poetry_VAE_A_Variational_Autoencoder_for_Diverse_Poetry_Generation_(genAi_Proj).ipynb
├── Poetry VAE (genAi).pdf
├── README.md
├── requirements.txt (if created)
├── data/               # Dataset storage
├── models/             # Saved model checkpoints
├── outputs/            # Generated poems
└── visualizations/     # Plots and analysis
```

## 🔬 Research Methodology & Process

This project follows a comprehensive research methodology:

### 📖 Phase 1: Literature Review
- Conducted extensive research on existing VAE applications for text generation
- Identified key research papers suggesting VAE architectures for poetry generation
- Analyzed state-of-the-art approaches and their limitations

### 💻 Phase 2: Implementation
- Developed a custom PyTorch implementation based on research findings
- Implemented advanced features like self-attention mechanisms and KL annealing
- Optimized for Google Colab T4 GPU training environment
- Extensive experimentation with hyperparameters and architecture choices

### 📝 Phase 3: Research Paper
- Documented findings and methodology in a comprehensive research paper
- Analyzed results, performance metrics, and generated poetry quality
- Contributed novel insights to the field of neural poetry generation

### 🎯 Key Results

This implementation demonstrates:
- Successful learning of poetry structure and style
- Generation of coherent, creative poems
- Meaningful latent space organization by poetic themes
- Controllable generation through temperature and sampling strategies
- Novel contributions to VAE-based text generation research

See `Poetry VAE (genAi).pdf` for detailed research findings, experimental results, and comprehensive analysis.


## 🙏 Acknowledgments

- Poetry Foundation for the dataset
- Kaggle for data hosting
- Google Colab for providing free GPU access
- PyTorch team for the deep learning framework

## 📚 References

- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes
- Bowman, S. R., et al. (2016). Generating Sentences from a Continuous Space
- Zhang, Y., et al. (2019). Learning to Generate Poetry with Mixed Neural Models


---

**Generated with ❤️ using PyTorch and creativity** 
