# inference.py

import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import your custom modules here
# from your_module import (
#     Transformer,
#     MCTS,
#     State,
#     ActionEncoder,
#     RepresentationNetwork,
#     DynamicsNetwork,
#     PredictionNetwork,
#     PPOAgent,
#     MCTSNode
# )

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the tokenizer
model_path = 'path_to_your_model'  # Replace with your actual model path
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure that the pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define the necessary model hyperparameters
input_dim = tokenizer.vocab_size
output_dim = tokenizer.vocab_size
d_model = 512  # Model dimensionality
num_heads = 8  # Number of attention heads
num_layers = 6  # Number of transformer layers
d_ff = 2048  # Dimension of feedforward network
num_experts = 4  # Number of experts in MoE
dropout = 0.1
top_k = 2

# Initialize the Transformer model
transformer = Transformer(
    input_dim=input_dim,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    num_experts=num_experts,
    output_dim=output_dim,
    dropout=dropout,
    top_k=top_k
).to(device)

# Load the pretrained model weights
model_weights_path = r"C:\Users\Admin\lightbulb\distilled_model\epoch_20.pt"
if os.path.exists(model_weights_path):
    try:
        transformer.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Loaded pretrained model weights from '{model_weights_path}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Model weights file not found at '{model_weights_path}'")
    exit(1)

# Initialize other components
action_vocab_size = len(tokenizer)  # Assuming actions correspond to tokenizer vocabulary
action_encoder = ActionEncoder(action_vocab_size, d_model).to(device)
state_dim = d_model
value_dim = 1  # Assuming scalar value output

# Initialize representation, dynamics, and prediction networks
representation_network = RepresentationNetwork(output_dim, d_model, state_dim).to(device)
dynamics_network = DynamicsNetwork(state_dim, d_model, d_ff).to(device)
prediction_network = PredictionNetwork(state_dim, action_vocab_size, value_dim).to(device)

# Load pretrained weights for other components if available
# For example:
# representation_network.load_state_dict(torch.load('path_to_representation_weights.pth'))
# dynamics_network.load_state_dict(torch.load('path_to_dynamics_weights.pth'))
# prediction_network.load_state_dict(torch.load('path_to_prediction_weights.pth'))

# Initialize MCTS
action_to_index = {token: idx for idx, token in tokenizer.get_vocab().items()}
mcts = MCTS(
    prediction_network=prediction_network,
    dynamics_network=dynamics_network,
    action_encoder=action_encoder,
    action_to_index=action_to_index,
    num_iterations=10,
    exploration_constant=math.sqrt(2),
    beam_size=5,
    n_tokens_predict=3
)

# Vectorizer for computing attention weights
vectorizer = TfidfVectorizer()

# Helper function to move tensors to device
def to_device(tensor):
    return tensor.to(device)

# Function to augment the input prompt
def augment_prompt(prompt, previous_outputs):
    # Compute attention weights based on similarity
    attention_weights = compute_attention_weights(prompt, previous_outputs, vectorizer)
    # Include weighted previous outputs
    if len(attention_weights) > 0:
        weighted_outputs = []
        for weight, output in zip(attention_weights, previous_outputs):
            weighted_output = f"[Weight: {weight:.2f}] {output}"
            weighted_outputs.append(weighted_output)
        augmented_prompt = prompt + "\n" + "\n".join(weighted_outputs)
    else:
        augmented_prompt = prompt
    return augmented_prompt

# Function to compute attention weights based on similarity
def compute_attention_weights(current_step_desc, previous_outputs, vectorizer):
    if not previous_outputs:
        return np.array([])
    documents = [current_step_desc] + previous_outputs
    tfidf_matrix = vectorizer.fit_transform(documents)
    current_vector = tfidf_matrix[0:1]
    previous_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(current_vector, previous_vectors)[0]
    # Normalize similarities to sum to 1
    if similarities.sum() == 0:
        weights = np.array([1.0 / len(similarities)] * len(similarities))
    else:
        weights = similarities / similarities.sum()
    return weights

# Function to generate text using the Transformer model
def generate_text(prompt, max_length=50):
    transformer.eval()
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)['input_ids']
    input_ids = to_device(input_ids)
    with torch.no_grad():
        outputs = transformer.generate_with_beam_search(
            src=input_ids,
            tokenizer=tokenizer,
            beam_size=5,
            max_length=max_length,
            n_tokens_predict=3,
            temperature=1.0
        )
    # Decode the generated sequences
    generated_texts = []
    for seq, score in outputs:
        text = tokenizer.decode(seq.squeeze(), skip_special_tokens=True)
        generated_texts.append((text, score))
    return generated_texts

# Main inference function
def main():
    # Load your prompt or data seed
    data_seed = "Your input prompt or data seed goes here."

    # Example steps (you can modify these as needed)
    steps = {
        1: {
            'description': "Analyze all the relevant input data about {data_seed}, in whatever forms it is in.",
            'dependencies': [],
            'weight': 1.0
        },
        2: {
            'description': "Summarize the data about {data_seed}, utilizing output of step (1).",
            'dependencies': [1],
            'weight': 0.9
        },
        # Add more steps if necessary
    }

    previous_steps_outputs = []
    for step_num in range(1, len(steps) + 1):
        step_info = steps[step_num]
        # Format the step with data_seed
        step_text = step_info['description'].format(data_seed=data_seed)
        # Augment the prompt
        augmented_prompt = augment_prompt(step_text, previous_steps_outputs)
        # Generate text
        generated_texts = generate_text(augmented_prompt)
        # Select the best generated text based on score
        if generated_texts:
            best_text, best_score = max(generated_texts, key=lambda x: x[1])
            print(f"Step {step_num} Output:\n{best_text}\n")
            # Store the output for the next step
            previous_steps_outputs.append(best_text)
        else:
            print(f"No output generated for Step {step_num}.")
            break

if __name__ == "__main__":
    main()
