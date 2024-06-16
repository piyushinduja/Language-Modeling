# Character-Level Language Model using LSTM and N-gram Models
This repository contains implementations of character-level language models using LSTM and N-gram approaches. The models are designed to predict sequences of characters based on input sequences, demonstrating their capabilities in language modeling tasks.

## Overview
Language modeling is essential for predicting the probability of sequences of tokens (characters, words) in natural language processing. This project explores two approaches:

### LSTM Language Model:

Built an LSTM-based recurrent neural network.
Predicts the next character in a sequence given previous characters.
Achieved a perplexity of 9.43 on evaluation, indicating strong predictive performance.

### N-gram Model with Laplace Smoothing:

Implemented an N-gram model with Laplace (add-one) smoothing.
Estimates probabilities of character sequences based on frequency counts.
Achieved a perplexity of 17.85, demonstrating robust performance on unseen sequences.

## Conclusion
This project highlights the effectiveness of LSTM and N-gram models in character-level language modeling tasks. The LSTM model demonstrates superior performance with a lower perplexity, whereas the N-gram model showcases the importance of smoothing techniques in handling unseen data.
