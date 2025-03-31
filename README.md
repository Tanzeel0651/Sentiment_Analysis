# Sentiment Analysis with BiLSTM and GloVe Embeddings

This project demonstrates a sentiment classification model built using Keras and TensorFlow. It uses a Bidirectional LSTM architecture and pre-trained GloVe embeddings to analyze and classify tweets as **positive**, **neutral**, or **negative**.

## 💡 Features
- Tokenization and padding using Keras
- GloVe-based embedding matrix for transfer learning
- Bidirectional LSTM model with Global Max Pooling
- ROC-AUC evaluation and training visualization
- CleanTweet dataset (sample of 500 entries)

## 📦 Tech Stack
- Python, Pandas, NumPy
- Keras with TensorFlow backend
- GloVe Word Embeddings
- Matplotlib (for training visualization)
- scikit-learn (ROC-AUC)

## 🚀 How to Run
1. Place the GloVe vector file in the `/glove_vector/` directory.
2. Load and preprocess data from `CleanTweet.csv`.
3. Run the notebook or script to train the BiLSTM model.
4. Visualize training metrics and evaluate AUC.

## 📈 Example Output
Model achieves strong AUC on small test samples with minimal tuning.

---

Feel free to fork or contribute!  
