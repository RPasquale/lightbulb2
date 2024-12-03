import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Tuple
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train or Inference with World Model and Tree of Thought.')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pretrained model name or path')

    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name from HuggingFace Datasets')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset configuration name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--mcts_iterations', type=int, default=3, help='Number of MCTS Iterations')
    parser.add_argument('--mcts_exploration_constant', type=float, default=1.414, help='Exploration constant for MCTS')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--alpha', type=float, default=0.1, help='Entropy regularization weight')
    parser.add_argument('--beta', type=float, default=0.1, help='Variance regularization weight')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save the models')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for entropy and variance')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train', help='Mode: train or inference')
    parser.add_argument('--inference_mode', type=str, choices=['world_model', 'without_world_model', 'world_model_tree_of_thought'], default='world_model_tree_of_thought', help='Inference mode')
    parser.add_argument('--query', type=str, default='', help='Input query for inference')
    parser.add_argument('--train_mode', type=str, choices=['world_model', 'language_model'], default='language_model', help='Train world model or language model only')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--n_tokens_predict', type=int, default=3, help='Number of tokens to predict at each step')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load saved model. If not provided, a new model will be initialized.')

    parser.add_argument('--use_custom_data', action='store_true', help='Use custom data for training')

    # Determine the base directory
    if hasattr(sys, 'frozen') and hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_dir = sys._MEIPASS
    elif '__file__' in globals():
        # Running as a script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # Running in an interactive environment (e.g., Jupyter, Colab)
        base_dir = os.getcwd()

    default_paths = [
        '/content/drive/MyDrive/lightbulb/knowledge_base.json',
        '/content/drive/MyDrive/lightbulb/rag_cache.json',
        '/content/drive/MyDrive/lightbulb/llm_training_data/llm_training_data.jsonl'
    ]

    parser.add_argument('--custom_data_paths', nargs='+', default=default_paths,
                        help='Paths to custom data files (relative to the script location or current working directory)')

    args, unknown = parser.parse_known_args()

    # Convert relative paths to absolute paths
    args.custom_data_paths = [os.path.abspath(os.path.join(base_dir, path)) for path in args.custom_data_paths]

    return args

import json
import jsonlines

def load_custom_data_from_files(file_paths):
    custom_data = []
    for file_path in file_paths:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    custom_data.extend(data)
                else:
                    custom_data.append(data)
        elif file_path.endswith('.jsonl'):
            with jsonlines.open(file_path) as reader:
                custom_data.extend(reader)
    return custom_data

def preprocess_custom_data(data_list):
    processed_data = []
    for item in data_list:
        # Check if the item is a string (JSON)
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {item[:100]}...")  # Print first 100 chars for debugging
                continue  # Skip this item if it's not valid JSON

        # Process query and content
        query = item.get('query', '')
        content = item.get('content', '')
        if content == "RAG response generation failed.":
            content = ""

        # Combine query and content
        combined_text = f"Query: {query} Content: {content}"

        # Process numerical data (assuming these are available in the item dict)
        episode_reward = item.get('episode_reward', 0)
        loss = item.get('loss', 0)
        cosine_similarity = item.get('cosine_similarity', 0)
        rag_performance = item.get('rag_performance', 0)
        ranking_model_performance = item.get('ranking_model_performance', 0)

        # Create a dictionary with processed data
        processed_item = {
            'text': combined_text,
            'episode_reward': episode_reward,
            'loss': loss,
            'cosine_similarity': cosine_similarity,
            'rag_performance': rag_performance,
            'ranking_model_performance': ranking_model_performance
        }

        processed_data.append(processed_item)

    return processed_data

def load_custom_data(args, tokenizer, custom_data):
    # Preprocess the custom data
    processed_data = preprocess_custom_data(custom_data)

    # Create a custom dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            encoded = self.tokenizer.encode_plus(
                item['text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'episode_reward': torch.tensor(item['episode_reward'], dtype=torch.float),
                'loss': torch.tensor(item['loss'], dtype=torch.float),
                'cosine_similarity': torch.tensor(item['cosine_similarity'], dtype=torch.float),
                'rag_performance': torch.tensor(item['rag_performance'], dtype=torch.float),
                'ranking_model_performance': torch.tensor(item['ranking_model_performance'], dtype=torch.float)
            }

    # Create dataset and dataloader
    dataset = CustomDataset(processed_data, tokenizer, args.max_length)

    # Split the dataset into train and eval
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, eval_loader



def load_data(args, tokenizer):
    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.max_length)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset['train'].column_names,
    )

    # Build inputs and labels for language modeling
    block_size = args.max_length

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples['input_ids'])
        # We drop the small remainder
        total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
    )

    # Create DataLoader
    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation'] if 'validation' in lm_datasets else lm_datasets['test']

    def data_collator(data):
        return {
            'input_ids': torch.tensor([f['input_ids'] for f in data], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in data], dtype=torch.long)
        }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True,  # Speeds up transfer to GPU
        num_workers=4
    )
    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, eval_loader

def save_all_models(transformer_model, representation_network, dynamics_network, prediction_network, action_encoder, save_dir, epoch):
    """
    Save all models to the specified directory.

    Args:
        transformer_model (nn.Module): Transformer model.
        representation_network (nn.Module): Representation network.
        dynamics_network (nn.Module): Dynamics network.
        prediction_network (nn.Module): Prediction network.
        action_encoder (nn.Module): Action encoder.
        save_dir (str): Directory to save the models.
        epoch (int): Current epoch number.
    """
    os.makedirs(save_dir, exist_ok=True)

    torch.save(transformer_model.state_dict(), os.path.join(save_dir, f'transformer_model_epoch_{epoch}.pt'))
    torch.save(representation_network.state_dict(), os.path.join(save_dir, f'representation_network_epoch_{epoch}.pt'))
    torch.save(dynamics_network.state_dict(), os.path.join(save_dir, f'dynamics_network_epoch_{epoch}.pt'))
    torch.save(prediction_network.state_dict(), os.path.join(save_dir, f'prediction_network_epoch_{epoch}.pt'))
    torch.save(action_encoder.state_dict(), os.path.join(save_dir, f'action_encoder_epoch_{epoch}.pt'))

    print(f"All models saved for epoch {epoch}.")




# Objective Functions

class InfoNCE_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE_Loss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Args:
            z_i (torch.Tensor): Flattened representations from view i, shape (2n, embed_dim)
            z_j (torch.Tensor): Flattened representations from view j, shape (2n, embed_dim)

        Returns:
            torch.Tensor: InfoNCE loss
        """
        n = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2n, embed_dim)

        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T)  # Shape: (2n, 2n)

        # Create a mask to exclude self-similarity
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e4)  # Use a manageable negative value

        # Create labels for contrastive learning
        labels = torch.arange(n, device=z.device)
        labels = torch.cat([labels + n, labels], dim=0)  # Shape: (2n,)

        # Apply temperature scaling
        similarity_matrix /= self.temperature

        # Compute cross-entropy loss
        loss = self.cross_entropy(similarity_matrix, labels)
        return loss

class CovarianceRegularization(nn.Module):
    def __init__(self, lambda_reg=1e-3):
        super(CovarianceRegularization, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, embeddings):
        """
        Args:
            embeddings (torch.Tensor): Embedding tensor, shape (batch_size, embed_dim)

        Returns:
            torch.Tensor: Covariance regularization loss
        """
        batch_size, embed_dim = embeddings.size()
        mean = embeddings.mean(dim=0)
        embeddings_centered = embeddings - mean
        cov = (embeddings_centered.T @ embeddings_centered) / (batch_size - 1)
        cov_loss = torch.sum(cov ** 2) - torch.sum(torch.diag(cov) ** 2)
        return self.lambda_reg * cov_loss

class DynamicsPerformanceLoss(nn.Module):
    def __init__(self, lambda_var=1e-3):
        super(DynamicsPerformanceLoss, self).__init__()
        self.lambda_var = lambda_var

    def forward(self, true_next_state, predicted_next_state):
        """
        Args:
            true_next_state (torch.Tensor): Ground truth next state, shape (batch_size, state_dim)
            predicted_next_state (torch.Tensor): Predicted next state, shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Dynamics performance loss
        """
        mse_loss = F.mse_loss(predicted_next_state, true_next_state)
        variance_loss = torch.var(predicted_next_state, dim=0).mean()
        return mse_loss + self.lambda_var * variance_loss

class ThoughtConsistencyLoss(nn.Module):
    def __init__(self):
        super(ThoughtConsistencyLoss, self).__init__()

    def forward(self, true_next_state, perturbed_next_state):
        """
        Args:
            true_next_state (torch.Tensor): Ground truth next state, shape (batch_size, state_dim)
            perturbed_next_state (torch.Tensor): Perturbed next state, shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Thought-consistency loss
        """
        return F.mse_loss(true_next_state, perturbed_next_state)

class PolicyValueJointLoss(nn.Module):
    def __init__(self, lambda_value=0.5):
        super(PolicyValueJointLoss, self).__init__()
        self.lambda_value = lambda_value
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, policy_logits, true_policy, value_pred, true_value):
        """
        Args:
            policy_logits (torch.Tensor): Logits from the policy network, shape (batch_size * seq_len, num_actions)
            true_policy (torch.Tensor): Ground truth policy, shape (batch_size * seq_len, num_actions)
            value_pred (torch.Tensor): Predicted values, shape (batch_size * seq_len)
            true_value (torch.Tensor): Ground truth values, shape (batch_size * seq_len)

        Returns:
            torch.Tensor: Combined policy and value loss
        """
        policy_logits = policy_logits.reshape(-1, policy_logits.size(-1))
        true_policy = true_policy.reshape(-1, true_policy.size(-1))
        value_pred = value_pred.reshape(-1)
        true_value = true_value.reshape(-1)


        policy_loss = self.cross_entropy(policy_logits, true_policy.argmax(dim=1))
        value_loss = self.mse_loss(value_pred, true_value)
        return policy_loss + self.lambda_value * value_loss

class ActionDiversityReward(nn.Module):
    def __init__(self, lambda_div=1e-3):
        super(ActionDiversityReward, self).__init__()
        self.lambda_div = lambda_div

    def forward(self, action_embeddings):
        """
        Args:
            action_embeddings (torch.Tensor): Embeddings of actions, shape (batch_size, embed_dim)

        Returns:
            torch.Tensor: Action diversity loss
        """
        similarity_matrix = F.cosine_similarity(action_embeddings.unsqueeze(1), action_embeddings.unsqueeze(0), dim=2)
        # Zero out self-similarity
        similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(action_embeddings.device)
        diversity_loss = torch.sum(similarity_matrix ** 2)
        return self.lambda_div * diversity_loss

class ExpectedThoughtValueLoss(nn.Module):
    def __init__(self):
        super(ExpectedThoughtValueLoss, self).__init__()

    def forward(self, mcts_best_values):
        """
        Args:
            mcts_best_values (torch.Tensor): Best values from MCTS, shape (batch_size)

        Returns:
            torch.Tensor: ETV loss
        """
        return -mcts_best_values.mean()

class ExplorationRegularization(nn.Module):
    def __init__(self, lambda_expl=1e-3):
        super(ExplorationRegularization, self).__init__()
        self.lambda_expl = lambda_expl

    def forward(self, visit_counts):
        """
        Args:
            visit_counts (torch.Tensor): Visit counts for actions, shape (batch_size, num_actions)

        Returns:
            torch.Tensor: Exploration regularization loss
        """
        reward = torch.sum(1.0 / (visit_counts + 1), dim=-1)
        return self.lambda_expl * reward.mean()

class KL_DivergenceLoss(nn.Module):
    def __init__(self):
        super(KL_DivergenceLoss, self).__init__()

    def forward(self, old_policy, new_policy):
        """
        Args:
            old_policy (torch.Tensor): Old policy probabilities, shape (batch_size, num_actions)
            new_policy (torch.Tensor): New policy probabilities, shape (batch_size, num_actions)

        Returns:
            torch.Tensor: KL divergence loss
        """
        kl_div = F.kl_div(new_policy.log(), old_policy, reduction='batchmean')
        return kl_div


# Language Model Components

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEncoding, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len, batch_size, _ = x.size()
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        sin = sinusoid_inp.sin().unsqueeze(1)  # (seq_len, 1, d_model/2)
        cos = sinusoid_inp.cos().unsqueeze(1)  # (seq_len, 1, d_model/2)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Apply rotation
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x1 * cos - x2 * sin
        x_rotated[..., 1::2] = x1 * sin + x2 * cos

        return x_rotated

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        query = self.linear_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask shape: (batch_size, seq_len)
            # Expand mask to (batch_size, 1, 1, seq_len) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        return self.linear_out(output)


class MoE(nn.Module):
    def __init__(self, d_model, num_experts, d_ff, top_k=2, dropout=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU() if i % 2 == 0 else nn.SiLU(),
                nn.Linear(d_ff, d_model)
            )
            for i in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        # Compute gating scores
        gate_scores = self.gate(x)  # (batch_size, seq_len, num_experts)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (batch_size, seq_len, top_k)
        top_k_scores = F.softmax(top_k_scores, dim=-1)  # (batch_size, seq_len, top_k)

        # Initialize output
        output = torch.zeros_like(x)

        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        output_flat = output.view(-1, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (batch_size * seq_len, top_k)
        top_k_scores_flat = top_k_scores.view(-1, self.top_k)  # (batch_size * seq_len, top_k)

        for k in range(self.top_k):
            expert_idx_flat = top_k_indices_flat[:, k]  # (batch_size * seq_len)
            expert_scores_flat = top_k_scores_flat[:, k]  # (batch_size * seq_len)
            for e in range(self.num_experts):
                mask = (expert_idx_flat == e)  # Boolean mask
                if mask.any():
                    x_masked = x_flat[mask]  # Select tokens for expert e
                    expert_output = self.experts[e](x_masked)  # Apply expert e
                    output_flat[mask] += expert_scores_flat[mask].unsqueeze(-1) * expert_output

        output = output_flat.view(batch_size, seq_len, d_model)
        return self.dropout(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, dropout=0.1, top_k=2):
        super(TransformerBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, num_experts, d_ff, top_k, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, enc_output=None, enc_mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        # Cross-attention (only in decoder)
        if enc_output is not None:
            cross_attn_output = self.cross_attention(x, enc_output, enc_output, enc_mask)
            x = self.norm2(x + cross_attn_output)
        # Feedforward/MoE
        moe_output = self.moe(x)
        return self.norm3(x + moe_output)

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, num_experts, output_dim, dropout=0.1, top_k=2):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model, padding_idx=input_dim - 1)
        self.rotary_positional_encoding = RotaryPositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, num_experts, dropout, top_k) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, num_experts, dropout, top_k) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.rotary_positional_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        src_output = src_emb
        for layer in self.encoder_layers:
            src_output = layer(src_output, mask=src_mask)

        # Decoder
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.rotary_positional_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        tgt_output = tgt_emb
        for layer in self.decoder_layers:
            tgt_output = layer(tgt_output, mask=tgt_mask, enc_output=src_output, enc_mask=src_mask)
        output = self.output_layer(tgt_output)
        return output


    def generate_with_beam_search(self, src, tokenizer, beam_size=5, max_length=20, n_tokens_predict=3, temperature=1.0):
        """
        Generate sequences using beam search with multi-token prediction.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, seq_len)
            tokenizer: Tokenizer to access special tokens
            beam_size (int): Size of the beam for beam search
            max_length (int): Maximum length of the generated sequence
            n_tokens_predict (int): Number of tokens to predict at each step
            temperature (float): Temperature parameter for softmax

        Returns:
            List[Tuple[torch.Tensor, float]]: List of (sequence, score) tuples
        """
        batch_size = src.size(0)
        device = src.device
        vocab_size = self.output_layer.out_features

        # Encode the source
        src_enc = self.encode(src)

        # Initialize beam
        beam = [(torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device),
                 0.0,  # log probability
                 torch.zeros(batch_size, device=device),  # cumulative entropy
                 torch.zeros(batch_size, device=device))]  # cumulative variance

        for _ in range(max_length // n_tokens_predict):
            all_candidates = []
            for seq, score, cum_entropy, cum_variance in beam:
                if seq[:, -1].item() == tokenizer.eos_token_id:
                    all_candidates.append((seq, score, cum_entropy, cum_variance))
                    continue

                # Predict next n tokens
                logits = self.predict_next_n_tokens(src_enc, seq, n_tokens_predict)

                # Calculate probabilities, entropy, and variance
                probs = F.softmax(logits / temperature, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                variance = torch.var(probs, dim=-1)

                # Sample top-k tokens for each position
                topk_probs, topk_indices = torch.topk(probs, k=beam_size, dim=-1)

                # Generate all possible continuations
                for i in range(beam_size ** n_tokens_predict):
                    indices = [i // (beam_size ** j) % beam_size for j in range(n_tokens_predict)]
                    new_tokens = topk_indices[:, range(n_tokens_predict), indices]
                    new_seq = torch.cat([seq, new_tokens], dim=-1)
                    new_score = score + torch.sum(torch.log(topk_probs[:, range(n_tokens_predict), indices]))
                    new_entropy = cum_entropy + torch.sum(entropy[:, indices])
                    new_variance = cum_variance + torch.sum(variance[:, indices])

                    all_candidates.append((new_seq, new_score, new_entropy, new_variance))

            # Select top beam_size candidates
            beam = sorted(all_candidates, key=lambda x: x[1] - 0.1 * x[2] + 0.05 * x[3], reverse=True)[:beam_size]

            # Stop if all beams have ended
            if all(seq[:, -1].item() == tokenizer.eos_token_id for seq, _, _, _ in beam):
                break

        return [(seq, score) for seq, score, _, _ in beam]

    def encode(self, src):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb.transpose(0, 1)
        src_emb = self.rotary_positional_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)
        src_enc = src_emb
        for layer in self.encoder_layers:
            src_enc = layer(src_enc)
        return src_enc

    def predict_next_n_tokens(self, src_enc, tgt_seq, n_tokens):
        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.rotary_positional_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_dec = tgt_emb
        for layer in self.decoder_layers:
            tgt_dec = layer(tgt_dec, None, src_enc, None)
        output = self.output_layer(tgt_dec[:, -1:])
        return output.repeat(1, n_tokens, 1)



# MuZero Components

class ActionEncoder(nn.Module):
    def __init__(self, action_vocab_size, embed_dim):
        super(ActionEncoder, self).__init__()
        self.embedding = nn.Embedding(action_vocab_size, embed_dim)

    def forward(self, action_indices):
        """
        Args:
            action_indices (torch.Tensor): Tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Encoded actions of shape (batch_size, seq_len, embed_dim)
        """
        return self.embedding(action_indices)

class RepresentationNetwork(nn.Module):
    def __init__(self, vocab_dim, d_model, state_dim):
        super(RepresentationNetwork, self).__init__()
        self.proj = nn.Linear(vocab_dim, d_model)  # Project from vocab_dim to d_model
        self.linear = nn.Linear(d_model, state_dim)  # Project from d_model to state_dim
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, transformer_output):
        """
        Args:
            transformer_output (torch.Tensor): Shape (batch_size, seq_len, vocab_dim)

        Returns:
            torch.Tensor: Encoded state of shape (batch_size, seq_len, state_dim)
        """
        #print(f"[RepresentationNetwork] transformer_output shape: {transformer_output.shape}")
        
        # Project down from vocab_dim to d_model
        projected_output = self.proj(transformer_output)  # Shape: (batch_size, seq_len, d_model)
        #print(f"[RepresentationNetwork] projected_output shape after proj layer: {projected_output.shape}")
        
        # Project down from d_model to state_dim
        state = self.linear(projected_output)  # Shape: (batch_size, seq_len, state_dim)
        #print(f"[RepresentationNetwork] state shape after linear layer: {state.shape}")
        
        # Apply layer normalization
        state = self.norm(state)  # Shape: (batch_size, seq_len, state_dim)
        #print(f"[RepresentationNetwork] state shape after layer norm: {state.shape}")
        
        return state

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DynamicsNetwork, self).__init__()
        self.rms_norm = nn.LayerNorm(state_dim)
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        """
        Args:
            state (torch.Tensor): Current state, shape (batch_size, seq_len, state_dim)
            action (torch.Tensor): Action embedding, shape (batch_size, seq_len, action_dim)

        Returns:
            torch.Tensor: Predicted next state, shape (batch_size, seq_len, state_dim)
        """
        #print(f"[DynamicsNetwork] state shape before norm: {state.shape}")
        #print(f"[DynamicsNetwork] action shape: {action.shape}")
        
        # Normalize state
        norm_state = self.rms_norm(state)
        #print(f"[DynamicsNetwork] norm_state shape after layer norm: {norm_state.shape}")
        
        # Concatenate state and action
        combined = torch.cat([norm_state, action], dim=-1)
        #print(f"[DynamicsNetwork] combined shape after concatenation: {combined.shape}")
        
        # Pass through first fully connected layer and apply activation
        hidden = self.activation(self.fc1(combined))
        #print(f"[DynamicsNetwork] hidden shape after fc1 and activation: {hidden.shape}")
        
        # Project back to state_dim
        next_state = self.fc2(hidden)
        #print(f"[DynamicsNetwork] next_state shape after fc2: {next_state.shape}")
        
        return next_state

class PredictionNetwork(nn.Module):
    def __init__(self, state_dim, action_vocab_size, value_dim):
        super(PredictionNetwork, self).__init__()
        self.state_dim = state_dim
        self.rms_norm = nn.LayerNorm(state_dim)
        self.policy_head = nn.Linear(state_dim, action_vocab_size)  # Output size is action_vocab_size
        self.value_head = nn.Linear(state_dim, value_dim)

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): State representation, shape (batch_size, state_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy logits and value estimates
        """
        print(f"[PredictionNetwork] state shape before norm: {state.shape}")
        
        # Normalize state
        norm_state = self.rms_norm(state)
        #print(f"[PredictionNetwork] norm_state shape after layer norm: {norm_state.shape}")
        
        # Compute policy logits
        policy_logits = self.policy_head(norm_state)  # Shape: (batch_size, action_vocab_size)
        #print(f"[PredictionNetwork] policy_logits shape after policy head: {policy_logits.shape}")
        
        # Compute value estimates
        value_estimates = self.value_head(norm_state).squeeze(-1)  # Shape: (batch_size)
        #print(f"[PredictionNetwork] value_estimates shape after value head: {value_estimates.shape}")
        
        return policy_logits, value_estimates

class MCTSNode:
    __slots__ = [
        'state',
        'parent',
        'action',
        'children',
        'visit_count',
        'value_sum',
        'prior',
        'cached_policy',
        'cached_value',
        'thought_node',
        'entropy',
        'variance'
    ]

    def __init__(self, state, thought_node, parent=None, action=None):
        self.state = state
        self.thought_node = thought_node
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.cached_policy = None
        self.cached_value = None
        self.entropy = 0.0
        self.variance = 0.0

    def expand(self, priors):
        for child_thought_node in self.thought_node.children:
            action = child_thought_node.name
            if action not in self.children:
                child_state = self.state.apply_action(action)
                child_node = MCTSNode(
                    state=child_state,
                    thought_node=child_thought_node,
                    parent=self,
                    action=action
                )
                child_node.prior = priors.get(action, 1.0 / len(self.thought_node.children))
                self.children[action] = child_node

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, total_visits, exploration_constant=math.sqrt(2)):
        if self.visit_count == 0:
            return float('inf')  # Ensure unvisited nodes are selected first
        avg_value = self.value_sum / self.visit_count
        exploration_term = exploration_constant * self.prior * math.sqrt(total_visits) / (1 + self.visit_count)
        entropy_term = -0.1 * self.entropy  # Slightly prefer lower entropy
        variance_term = 0.05 * self.variance  # Slightly prefer higher variance
        return avg_value + exploration_term + entropy_term + variance_term

class MCTS:
    def __init__(self, prediction_network, dynamics_network, action_encoder, action_to_index, num_iterations=10, exploration_constant=math.sqrt(2), beam_size=5, n_tokens_predict=3):
        self.prediction_network = prediction_network
        self.dynamics_network = dynamics_network
        self.action_encoder = action_encoder
        self.action_to_index = action_to_index  # Store action_to_index
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.beam_size = beam_size
        self.n_tokens_predict = n_tokens_predict
        self.cache = {}

    def search_with_beam(self, root_state):
        root_node = MCTSNode(state=root_state, thought_node=root_state.thought_node)

        # Evaluate the root node and backpropagate
        value_estimate = self.evaluate(root_node)  # Evaluate and expand root_node
        self.backpropagate(root_node, value_estimate)  # Backpropagate the value

        beam = [(root_node, 0.0, 0.0, 0.0, [])]  # (node, score, cum_entropy, cum_variance, action_sequence)

        for iteration in range(self.num_iterations):
            all_candidates = []
            for node, score, cum_entropy, cum_variance, action_sequence in beam:
                if node.is_leaf():
                    value_estimate = self.evaluate(node)
                    self.backpropagate(node, value_estimate)  # Backpropagate after evaluation
                if len(node.children) == 0:
                    continue  # No children to expand

                total_visits = sum(child.visit_count for child in node.children.values())
                # Select top actions based on UCB score
                sorted_children = sorted(
                    node.children.items(),
                    key=lambda item: item[1].ucb_score(total_visits, self.exploration_constant),
                    reverse=True
                )[:self.beam_size]

                for selected_action, selected_node in sorted_children:
                    current_node = selected_node
                    current_sequence = action_sequence + [selected_action]
                    current_score = score
                    current_entropy = cum_entropy + selected_node.entropy
                    current_variance = cum_variance + selected_node.variance

                    # Predict n_tokens_predict actions
                    for _ in range(self.n_tokens_predict):
                        if current_node.is_leaf():
                            value_estimate = self.evaluate(current_node)
                            self.backpropagate(current_node, value_estimate)  # Backpropagate after evaluation
                        if len(current_node.children) == 0:
                            break  # No more actions
                        total_visits = sum(child.visit_count for child in current_node.children.values())
                        next_action, next_node = max(
                            current_node.children.items(),
                            key=lambda item: item[1].ucb_score(total_visits, self.exploration_constant)
                        )
                        current_sequence.append(next_action)

                        # Prevent division by zero by ensuring visit_count > 0
                        if next_node.visit_count > 0:
                            current_score += next_node.value_sum / next_node.visit_count
                        else:
                            # Assign a default value or handle the zero division case
                            current_score += 0.0  # Alternatively, use a small epsilon or skip

                        current_entropy += next_node.entropy
                        current_variance += next_node.variance
                        current_node = next_node

                    all_candidates.append((current_node, current_score, current_entropy, current_variance, current_sequence))

            if not all_candidates:
                break  # No more candidates to expand

            # Select top beam_size candidates
            beam = sorted(all_candidates, key=lambda x: x[1] - 0.1 * x[2] + 0.05 * x[3], reverse=True)[:self.beam_size]
            #print(f"Iteration {iteration + 1}: Beam size after sorting: {len(beam)}")  # Debug

        if beam:
            best_sequence = beam[0][4]
            return best_sequence
        else:
            return []



    def search(self, root_state):
        root_node = MCTSNode(state=root_state, thought_node=root_state.thought_node)

        for _ in range(self.num_iterations):
            node = self.select(root_node)
            value = self.evaluate(node)
            self.backpropagate(node, value)

        return self.best_action_sequence(root_node)

    def select(self, node):
        while not node.is_leaf():
            total_visits = sum(child.visit_count for child in node.children.values())
            _, node = max(
                node.children.items(),
                key=lambda item: item[1].ucb_score(total_visits, self.exploration_constant)
            )
        return node

    def evaluate(self, node):
        # Extract the last time step
        state_representation = node.state.representation[:, -1, :]  # Shape: (batch_size=1, state_dim)
        #print(f"Evaluating node with state_representation shape: {state_representation.shape}")  # Debug
        policy_logits, value_estimate = self.prediction_network(state_representation)
        #print(f"Policy logits shape: {policy_logits.shape}, Value estimate shape: {value_estimate.shape}")  # Debug
        value_estimate = value_estimate.item()  # Now safe as batch_size=1

        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)  # Shape: (action_vocab_size,)
        #print(f"Policy probabilities shape: {policy_probs.shape}")  # Debug

        priors = {}
        for child in node.thought_node.children:
            action_name = child.name
            action_idx = self.action_to_index.get(action_name, None)
            if action_idx is not None and action_idx < policy_probs.size(0):
                priors[action_name] = policy_probs[action_idx].item()
            else:
                priors[action_name] = 1.0 / len(node.thought_node.children)

        node.expand(priors)

        # Calculate entropy and variance
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-9))
        variance = torch.var(policy_probs)
        node.entropy = entropy.item()
        node.variance = variance.item()

        #print(f"Node entropy: {node.entropy}, variance: {node.variance}")  # Debug

        return value_estimate  # Return the value estimate for backpropagation


    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def best_action_sequence(self, root_node):
        sequences = []
        self._generate_sequences(root_node, [], sequences)

        # Score sequences based on visit counts, entropy, and variance
        scored_sequences = []
        for seq in sequences:
            score = sum(node.visit_count for node in seq)
            entropy = sum(node.entropy for node in seq)
            variance = sum(node.variance for node in seq)
            adjusted_score = score - 0.1 * entropy + 0.05 * variance
            scored_sequences.append((seq, adjusted_score))

        # Sort sequences by adjusted score and select top beam_size
        best_sequences = sorted(scored_sequences, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # Return the actions of the best sequence
        best_sequence = best_sequences[0][0]
        return [node.action for node in best_sequence[1:self.n_tokens_predict+1]]  # Exclude root node

    def _generate_sequences(self, node, current_sequence, sequences):
        current_sequence.append(node)
        if len(current_sequence) > self.n_tokens_predict or not node.children:
            sequences.append(current_sequence)
        else:
            for child in node.children.values():
                self._generate_sequences(child, current_sequence.copy(), sequences)

class State:
    def __init__(self, representation, dynamics_network, action_encoder, action_to_index, thought_node):
        self.representation = representation
        self.dynamics_network = dynamics_network
        self.action_encoder = action_encoder
        self.action_to_index = action_to_index  # Store action_to_index
        self.thought_node = thought_node

    def apply_action(self, action):
        next_thought_node = None
        for child in self.thought_node.children:
            if child.name == action:
                next_thought_node = child
                break
        if next_thought_node is None:
            raise ValueError(f"Action '{action}' is not valid from the current thought node.")

        # Adjust action_index and action_embedding shapes
        action_index = torch.tensor([self.action_to_index[action]], device=self.representation.device)
        action_embedding = self.action_encoder(action_index)
        # Extract the last time step of the state
        state = self.representation[:, -1, :]  # Shape: (batch_size, state_dim)

        # Ensure action_embedding matches the state dimension
        next_state_representation = self.dynamics_network(state, action_embedding)  # Shape: (batch_size, state_dim)

        # Append the new state to the representation history
        new_representation = torch.cat([self.representation, next_state_representation.unsqueeze(1)], dim=1)  # Shape: (batch_size, seq_len+1, state_dim)

        return State(
            representation=new_representation,
            dynamics_network=self.dynamics_network,
            action_encoder=self.action_encoder,
            action_to_index=self.action_to_index,  # Pass action_to_index
            thought_node=next_thought_node
        )

class PPOAgent:
    def __init__(self, policy_network, optimizer, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def compute_loss(self, states, old_log_probs, actions, returns, advantages):
        # Get policy logits and value estimates
        policy_logits, value_estimates = self.policy_network(states)

        # Flatten all tensors
        policy_logits = policy_logits.reshape(-1, policy_logits.size(-1))
        value_estimates = value_estimates.reshape(-1)
        actions = actions.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        # Ensure all tensors have the same first dimension
        assert policy_logits.size(0) == value_estimates.size(0) == actions.size(0) == old_log_probs.size(0) == returns.size(0) == advantages.size(0), "Tensor sizes mismatch"

        # Compute new log probabilities
        new_log_probs_all = F.log_softmax(policy_logits, dim=-1)
        new_log_probs = new_log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute ratios
        ratios = torch.exp(new_log_probs - old_log_probs)

        # PPO surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(value_estimates, returns)

        # Entropy loss
        entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total_loss


# Tree of Thought Components

class ThoughtNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children is not None else []  # Initialize children if provided
        self.parent = None

        # Set parent references for each child
        for child in self.children:
            child.parent = self

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)    

# Function to build the Tree of Thought from your detailed structure
def build_tree_of_thought():
    # Create the root node
    root = ThoughtNode('Problem-Solving Process')

    # Level 1 nodes
    problem_identification = ThoughtNode('Problem Identification')
    problem_analysis = ThoughtNode('Problem Analysis')
    solution_generation = ThoughtNode('Solution Generation')
    implementation = ThoughtNode('Implementation')
    evaluation_adjustment = ThoughtNode('Evaluation and Adjustment')

    root.add_child(problem_identification)
    root.add_child(problem_analysis)
    root.add_child(solution_generation)
    root.add_child(implementation)
    root.add_child(evaluation_adjustment)

    # Problem Identification children
    B1 = ThoughtNode('Define the Problem')
    B2 = ThoughtNode('Identify Stakeholders')
    B3 = ThoughtNode('Determine Constraints')
    B4 = ThoughtNode('Recognize Problem Type')
    B5 = ThoughtNode('Historical Context')
    problem_identification.add_child(B1)
    problem_identification.add_child(B2)
    problem_identification.add_child(B3)
    problem_identification.add_child(B4)
    problem_identification.add_child(B5)

    # Define the Problem children
    B1a = ThoughtNode('Problem Statement Formulation')
    B1b = ThoughtNode('Scope Definition')
    B1c = ThoughtNode('Objective Setting')
    B1.add_child(B1a)
    B1.add_child(B1b)
    B1.add_child(B1c)

    # Identify Stakeholders children
    B2a = ThoughtNode('Stakeholder Mapping')
    B2b = ThoughtNode('Interest and Influence Analysis')
    B2c = ThoughtNode('Engagement Strategy')
    B2.add_child(B2a)
    B2.add_child(B2b)
    B2.add_child(B2c)

    # Determine Constraints children
    B3a = ThoughtNode('Resource Limitations')
    B3b = ThoughtNode('Time Constraints')
    B3c = ThoughtNode('Legal and Regulatory Constraints')
    B3.add_child(B3a)
    B3.add_child(B3b)
    B3.add_child(B3c)

    # Recognize Problem Type children
    B4a = ThoughtNode('Simple vs Complex')
    B4b = ThoughtNode('Known vs Unknown')
    B4c = ThoughtNode('Tame vs Wicked Problems')
    B4.add_child(B4a)
    B4.add_child(B4b)
    B4.add_child(B4c)

    # Historical Context children
    B5a = ThoughtNode('Previous Attempts')
    B5b = ThoughtNode('Lessons Learned')
    B5c = ThoughtNode('Environmental Factors')
    B5.add_child(B5a)
    B5.add_child(B5b)
    B5.add_child(B5c)

    # Problem Analysis children
    C1 = ThoughtNode('Root Cause Analysis')
    C2 = ThoughtNode('System Mapping')
    C3 = ThoughtNode('Data Collection')
    C4 = ThoughtNode('Impact Assessment')
    C5 = ThoughtNode('Theoretical Framework')
    problem_analysis.add_child(C1)
    problem_analysis.add_child(C2)
    problem_analysis.add_child(C3)
    problem_analysis.add_child(C4)
    problem_analysis.add_child(C5)

    # Root Cause Analysis children
    C1a = ThoughtNode('5 Whys Technique')
    C1b = ThoughtNode('Fishbone Diagram')
    C1c = ThoughtNode('Pareto Analysis')
    C1.add_child(C1a)
    C1.add_child(C1b)
    C1.add_child(C1c)

    # System Mapping children
    C2a = ThoughtNode('Causal Loop Diagrams')
    C2b = ThoughtNode('Stock and Flow Models')
    C2c = ThoughtNode('Network Analysis')
    C2.add_child(C2a)
    C2.add_child(C2b)
    C2.add_child(C2c)

    # Data Collection children
    C3a = ThoughtNode('Quantitative Data')
    C3b = ThoughtNode('Qualitative Data')
    C3c = ThoughtNode('Data Validation')
    C3.add_child(C3a)
    C3.add_child(C3b)
    C3.add_child(C3c)

    # Quantitative Data children
    C3a1 = ThoughtNode('Surveys and Questionnaires')
    C3a2 = ThoughtNode('Experimental Data')
    C3a3 = ThoughtNode('Big Data Analytics')
    C3a.add_child(C3a1)
    C3a.add_child(C3a2)
    C3a.add_child(C3a3)

    # Qualitative Data children
    C3b1 = ThoughtNode('Interviews')
    C3b2 = ThoughtNode('Focus Groups')
    C3b3 = ThoughtNode('Observational Studies')
    C3b.add_child(C3b1)
    C3b.add_child(C3b2)
    C3b.add_child(C3b3)

    # Data Validation children
    C3c1 = ThoughtNode('Statistical Validation')
    C3c2 = ThoughtNode('Cross-Validation')
    C3c3 = ThoughtNode('Expert Review')
    C3c.add_child(C3c1)
    C3c.add_child(C3c2)
    C3c.add_child(C3c3)

    # Impact Assessment children
    C4a = ThoughtNode('Environmental Impact')
    C4b = ThoughtNode('Social Impact')
    C4c = ThoughtNode('Economic Impact')
    C4.add_child(C4a)
    C4.add_child(C4b)
    C4.add_child(C4c)

    # Theoretical Framework children
    C5a = ThoughtNode('Literature Review')
    C5b = ThoughtNode('Conceptual Modeling')
    C5c = ThoughtNode('Hypothesis Formation')
    C5.add_child(C5a)
    C5.add_child(C5b)
    C5.add_child(C5c)

    # Solution Generation children
    D1 = ThoughtNode('Creative Problem Solving')
    D2 = ThoughtNode('Analytical Approach')
    D3 = ThoughtNode('Mathematical Computation')
    D4 = ThoughtNode('Decision Making')
    solution_generation.add_child(D1)
    solution_generation.add_child(D2)
    solution_generation.add_child(D3)
    solution_generation.add_child(D4)

    # Action Planning, Resource Allocation, Change Management children (implementation phase)
    E1 = ThoughtNode('Action Planning')
    E2 = ThoughtNode('Resource Allocation')
    E3 = ThoughtNode('Change Management')
    implementation.add_child(E1)
    implementation.add_child(E2)
    implementation.add_child(E3)

    # Verification, Performance Metrics, Feedback Loops, Continuous Improvement children (evaluation phase)
    F1 = ThoughtNode('Verification')
    F2 = ThoughtNode('Performance Metrics')
    F3 = ThoughtNode('Feedback Loops')
    F4 = ThoughtNode('Continuous Improvement')
    evaluation_adjustment.add_child(F1)
    evaluation_adjustment.add_child(F2)
    evaluation_adjustment.add_child(F3)
    evaluation_adjustment.add_child(F4)

    # Cross-Cutting Considerations children
    G = ThoughtNode('Cross-Cutting Considerations')
    root.add_child(G)

    # Cross-Cutting Considerations children
    G1 = ThoughtNode('Ethical Framework')
    G2 = ThoughtNode('Stakeholder Management')
    G3 = ThoughtNode('Interdisciplinary Connections')
    G4 = ThoughtNode('Technological Integration')
    G5 = ThoughtNode('Emotional Intelligence')
    G6 = ThoughtNode('Collaborative Problem Solving')
    G7 = ThoughtNode('Computational Considerations')  # Assuming H was intended as G7
    G8 = ThoughtNode('Order of Operations')  # Assuming I was intended as G8
    G9 = ThoughtNode('Critical Thinking')  # Assuming J was intended as G9
    G10 = ThoughtNode('Future Perspective')  # Assuming K was intended as G10
    G11 = ThoughtNode('Learning and Adaptation')  # Assuming L was intended as G11
    G.add_child(G1)
    G.add_child(G2)
    G.add_child(G3)
    G.add_child(G4)
    G.add_child(G5)
    G.add_child(G6)
    G.add_child(G7)
    G.add_child(G8)
    G.add_child(G9)
    G.add_child(G10)
    G.add_child(G11)

    # Ethical Framework children
    G1a = ThoughtNode('Value-based Decision Making')
    G1b = ThoughtNode('Long-term Consequences')
    G1.add_child(G1a)
    G1.add_child(G1b)

    # Value-based Decision Making children
    G1a1 = ThoughtNode('Ethical Theories Application')
    G1a2 = ThoughtNode('Moral Dilemma Resolution')
    G1a.add_child(G1a1)
    G1a.add_child(G1a2)

    # Long-term Consequences children
    G1b1 = ThoughtNode('Sustainability Assessment')
    G1b2 = ThoughtNode('Intergenerational Impact')
    G1b.add_child(G1b1)
    G1b.add_child(G1b2)

    # Stakeholder Management children
    G2a = ThoughtNode('Direct Stakeholders')
    G2b = ThoughtNode('Indirect Stakeholders')
    G2c = ThoughtNode('Conflicting Interests')
    G2.add_child(G2a)
    G2.add_child(G2b)
    G2.add_child(G2c)

    # Conflicting Interests children
    G2c1 = ThoughtNode('Negotiation Strategies')
    G2c2 = ThoughtNode('Conflict Resolution Techniques')
    G2c.add_child(G2c1)
    G2c.add_child(G2c2)

    # Interdisciplinary Connections children
    G3a = ThoughtNode('Related Fields')
    G3b = ThoughtNode('Cross-disciplinary Impact')
    G3.add_child(G3a)
    G3.add_child(G3b)

    # Related Fields children
    G3a1 = ThoughtNode('Cross-domain Knowledge Transfer')
    G3a2 = ThoughtNode('Interdisciplinary Collaboration')
    G3a.add_child(G3a1)
    G3a.add_child(G3a2)

    # Cross-disciplinary Impact children
    G3b1 = ThoughtNode('Synergy Identification')
    G3b2 = ThoughtNode('Holistic Impact Assessment')
    G3b.add_child(G3b1)
    G3b.add_child(G3b2)

    # Technological Integration children
    G4a = ThoughtNode('AI-assisted Problem Solving')
    G4b = ThoughtNode('Data-driven Insights')
    G4c = ThoughtNode('Digital Collaboration Tools')
    G4.add_child(G4a)
    G4.add_child(G4b)
    G4.add_child(G4c)

    # AI-assisted Problem Solving children
    G4a1 = ThoughtNode('Machine Learning Models')
    G4a2 = ThoughtNode('Natural Language Processing')
    G4a.add_child(G4a1)
    G4a.add_child(G4a2)

    # Data-driven Insights children
    G4b1 = ThoughtNode('Big Data Analytics')
    G4b2 = ThoughtNode('Predictive Modeling')
    G4b.add_child(G4b1)
    G4b.add_child(G4b2)

    # Digital Collaboration Tools children
    G4c1 = ThoughtNode('Project Management Platforms')
    G4c2 = ThoughtNode('Virtual Reality Collaboration')
    G4c.add_child(G4c1)
    G4c.add_child(G4c2)

    # Emotional Intelligence children
    G5a = ThoughtNode('Self-Awareness')
    G5b = ThoughtNode('Empathy')
    G5c = ThoughtNode('Stress Management')
    G5.add_child(G5a)
    G5.add_child(G5b)
    G5.add_child(G5c)

    # Self-Awareness children
    G5a1 = ThoughtNode('Emotional Recognition')
    G5a2 = ThoughtNode('Personal Bias Identification')
    G5a.add_child(G5a1)
    G5a.add_child(G5a2)

    # Empathy children
    G5b1 = ThoughtNode('Perspective Taking')
    G5b2 = ThoughtNode('Active Listening')
    G5b.add_child(G5b1)
    G5b.add_child(G5b2)

    # Stress Management children
    G5c1 = ThoughtNode('Mindfulness Techniques')
    G5c2 = ThoughtNode('Resilience Building')
    G5c.add_child(G5c1)
    G5c.add_child(G5c2)

    # Collaborative Problem Solving children
    G6a = ThoughtNode('Team Dynamics')
    G6b = ThoughtNode('Communication Strategies')
    G6c = ThoughtNode('Conflict Resolution')
    G6.add_child(G6a)
    G6.add_child(G6b)
    G6.add_child(G6c)

    # Team Dynamics children
    G6a1 = ThoughtNode('Team Formation Strategies')
    G6a2 = ThoughtNode('Role Assignment')
    G6a.add_child(G6a1)
    G6a.add_child(G6a2)

    # Communication Strategies children
    G6b1 = ThoughtNode('Clear Messaging')
    G6b2 = ThoughtNode('Feedback Mechanisms')
    G6b.add_child(G6b1)
    G6b.add_child(G6b2)

    # Conflict Resolution children
    G6c1 = ThoughtNode('Mediation Techniques')
    G6c2 = ThoughtNode('Consensus Building')
    G6c.add_child(G6c1)
    G6c.add_child(G6c2)

    # Computational Considerations children
    G7a = ThoughtNode('CPU Operations')
    G7b = ThoughtNode('GPU Parallelization')
    G7c = ThoughtNode('Floating-Point Precision')
    G7.add_child(G7a)
    G7.add_child(G7b)
    G7.add_child(G7c)

    # CPU Operations children
    G7a1 = ThoughtNode('Instruction Set Architecture')
    G7a2 = ThoughtNode('Pipelining and Parallelism')
    G7a.add_child(G7a1)
    G7a.add_child(G7a2)

    # GPU Parallelization children
    G7b1 = ThoughtNode('CUDA Programming')
    G7b2 = ThoughtNode('OpenCL Framework')
    G7b.add_child(G7b1)
    G7b.add_child(G7b2)

    # Floating-Point Precision children
    G7c1 = ThoughtNode('IEEE 754 Standard')
    G7c2 = ThoughtNode('Error Propagation Analysis')
    G7c.add_child(G7c1)
    G7c.add_child(G7c2)

    # Order of Operations children
    G8a = ThoughtNode('Parentheses')
    G8b = ThoughtNode('Exponents')
    G8c = ThoughtNode('Multiplication and Division')
    G8d = ThoughtNode('Addition and Subtraction')
    G8.add_child(G8a)
    G8.add_child(G8b)
    G8.add_child(G8c)
    G8.add_child(G8d)

    # Critical Thinking children
    G9a = ThoughtNode('Assumptions Questioning')
    G9b = ThoughtNode('Bias Recognition')
    G9.add_child(G9a)
    G9.add_child(G9b)

    # Assumptions Questioning children
    G9a1 = ThoughtNode('Socratic Questioning')
    G9a2 = ThoughtNode('Devil\'s Advocate Approach')
    G9a.add_child(G9a1)
    G9a.add_child(G9a2)

    # Bias Recognition children
    G9b1 = ThoughtNode('Cognitive Bias Identification')
    G9b2 = ThoughtNode('Debiasing Techniques')
    G9b.add_child(G9b1)
    G9b.add_child(G9b2)

    # Future Perspective children
    G10a = ThoughtNode('Short-term Projections')
    G10b = ThoughtNode('Long-term Scenarios')
    G10c = ThoughtNode('Potential Impacts')
    G10.add_child(G10a)
    G10.add_child(G10b)
    G10.add_child(G10c)

    # Short-term Projections children
    G10a1 = ThoughtNode('Trend Analysis')
    G10a2 = ThoughtNode('Scenario Planning')
    G10a.add_child(G10a1)
    G10a.add_child(G10a2)

    # Long-term Scenarios children
    G10b1 = ThoughtNode('Futures Wheel')
    G10b2 = ThoughtNode('Backcasting')
    G10b.add_child(G10b1)
    G10b.add_child(G10b2)

    # Potential Impacts children
    G10c1 = ThoughtNode('Risk Assessment')
    G10c2 = ThoughtNode('Opportunity Identification')
    G10c.add_child(G10c1)
    G10c.add_child(G10c2)

    # Learning and Adaptation children
    G11a = ThoughtNode('Reflective Practice')
    G11b = ThoughtNode('Knowledge Transfer')
    G11c = ThoughtNode('Adaptive Problem Solving')
    G11.add_child(G11a)
    G11.add_child(G11b)
    G11.add_child(G11c)

    # Reflective Practice children
    G11a1 = ThoughtNode('After Action Review')
    G11a2 = ThoughtNode('Learning Journals')
    G11a.add_child(G11a1)
    G11a.add_child(G11a2)

    # Knowledge Transfer children
    G11b1 = ThoughtNode('Best Practice Documentation')
    G11b2 = ThoughtNode('Mentoring Programs')
    G11b.add_child(G11b1)
    G11b.add_child(G11b2)

    # Adaptive Problem Solving children
    G11c1 = ThoughtNode('Iterative Approaches')
    G11c2 = ThoughtNode('Flexibility in Methodology')
    G11c.add_child(G11c1)
    G11c.add_child(G11c2)

    return root

def traverse_tree(node, action_list):
    if node.name not in action_list:
        action_list.append(node.name)
    for child in node.children:
        traverse_tree(child, action_list)



def infer(query, world_model_components, root_thought_node, tokenizer, max_length=2000, inference_mode='world_model', beam_size=5, n_tokens_predict=3, mcts_iterations=10, exploration_constant=1.414):


    """
    Perform inference given a query, utilizing the Tree of Thought and MCTS with multi-token beam search.

    Args:
        query (str): The input query or prompt.
        world_model_components (tuple): Tuple containing the model components.
        root_thought_node (ThoughtNode): The root node of the Tree of Thought.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used.
        max_length (int): Maximum length for the generated sequence.
        inference_mode (str): Inference mode ('world_model', 'without_world_model', 'world_model_tree_of_thought')
        beam_size (int): Size of the beam for beam search
        n_tokens_predict (int): Number of tokens to predict at each step

    Returns:
        List[str] or str: The sequence of actions (thoughts) selected or generated text.
    """
    representation_network, dynamics_network, prediction_network, action_encoder, ppo_agent, model_transformer = world_model_components

    # Tokenize and encode the query
    input_ids = tokenizer.encode(query, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    if inference_mode == 'without_world_model':
        # Directly use the transformer model to generate text with beam search
        with torch.no_grad():
            generated_sequences = model_transformer.generate_with_beam_search(
                src=input_ids,
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_length=max_length,
                n_tokens_predict=n_tokens_predict,
                temperature=args.temperature
            )
        best_sequence, best_score = generated_sequences[0]
        generated_text = tokenizer.decode(best_sequence[0], skip_special_tokens=True)
        return generated_text

    else:
        # Use the world model components
        with torch.no_grad():
            transformer_output = model_transformer(input_ids, input_ids)
            # Get the initial state representation
            initial_representation = representation_network(transformer_output)  # Shape: (batch_size=1, seq_len, state_dim)
            initial_representation = initial_representation[:, -1, :].unsqueeze(1)  # Shape: (batch_size=1, 1, state_dim)
            initial_state = State(
                representation=initial_representation,
                dynamics_network=dynamics_network,
                action_encoder=action_encoder,
                thought_node=root_thought_node
            )
            if inference_mode == 'world_model_tree_of_thought':
                # Use MCTS with Tree of Thought and multi-token beam search
                mcts = MCTS(prediction_network, dynamics_network, action_encoder, num_iterations=mcts_iterations, exploration_constant=exploration_constant)

                current_state = initial_state
                thought_sequence = []

                for _ in range(max_length // n_tokens_predict):
                    best_actions = mcts.search_with_beam(current_state)

                    thought_sequence.extend(best_actions)

                    # Apply the best actions to get the next state
                    for action in best_actions:
                        current_state = current_state.apply_action(action)

                    # Check if we've reached a leaf node (no further actions)
                    if len(current_state.thought_node.children) == 0:
                        break

                return thought_sequence
            else:
                # Use the world model without Tree of Thought, but with multi-token beam search
                beam = [(initial_state, 0.0, torch.zeros(1, device=device), torch.zeros(1, device=device))]  # (state, score, cum_entropy, cum_variance)

                for _ in range(max_length // n_tokens_predict):
                    all_candidates = []
                    for state, score, cum_entropy, cum_variance in beam:
                        policy_logits, _ = prediction_network(state.representation)
                        probs = F.softmax(policy_logits / args.temperature, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                        variance = torch.var(probs, dim=-1)

                        topk_probs, topk_indices = torch.topk(probs, k=beam_size, dim=-1)

                        for i in range(beam_size ** n_tokens_predict):
                            indices = [i // (beam_size ** j) % beam_size for j in range(n_tokens_predict)]
                            new_actions = [index_to_action[topk_indices[0, j, indices[j]].item()] for j in range(n_tokens_predict)]
                            new_score = score + torch.sum(torch.log(topk_probs[0, range(n_tokens_predict), indices]))
                            new_entropy = cum_entropy + torch.sum(entropy[0, indices])
                            new_variance = cum_variance + torch.sum(variance[0, indices])

                            new_state = state
                            for action in new_actions:
                                new_state = new_state.apply_action(action)

                            all_candidates.append((new_state, new_score, new_entropy, new_variance, new_actions))

                    # Select top beam_size candidates
                    beam = sorted(all_candidates, key=lambda x: x[1] - 0.1 * x[2] + 0.05 * x[3], reverse=True)[:beam_size]

                    # Accumulate actions
                    if not thought_sequence:
                        thought_sequence = [b[4] for b in beam]
                    else:
                        for i, b in enumerate(beam):
                            thought_sequence[i].extend(b[4])

                # Return the top sequence
                return thought_sequence[0]


def train_epoch_world_model(world_model_components, train_loader, optimizer, scheduler, scaler, args, model_transformer, state_dim, embed_dim, input_dim):
    representation_network, dynamics_network, prediction_network, action_encoder, ppo_agent, _ = world_model_components
    representation_network.train()
    dynamics_network.train()
    prediction_network.train()
    action_encoder.train()
    ppo_agent.policy_network.train()

    total_loss = 0.0
    optimizer.zero_grad()
    print(f"Starting World Model training epoch with {len(train_loader)} batches...")

    for i, batch in enumerate(train_loader):
        print(f"Processing batch {i+1}/{len(train_loader)}...")

        # Move batches to the device
        src_batch = batch['input_ids'].to(device)
        tgt_batch = batch['labels'].to(device)

        with torch.amp.autocast(device_type='cuda'):
            print("Forward pass through Transformer (frozen)...")
            with torch.no_grad():
                transformer_output = model_transformer(src_batch, tgt_batch[:, :-1])

            # World Model - Representation
            state_representation = representation_network(transformer_output)

            # For simplicity, let's assume true actions are provided (e.g., next tokens)
            true_actions = tgt_batch[:, :-1]
            action_sequences = true_actions

            # Get action embeddings
            action_embeddings = action_encoder(action_sequences)

            # Apply dynamics network
            predicted_next_state_batch = dynamics_network(state_representation, action_embeddings)

            # Prediction Network - Policy logits and value
            policy_logits, value_estimates = prediction_network(predicted_next_state_batch)

            # Define true_policy and true_value as placeholders on the GPU
            true_policy = F.one_hot(true_actions, num_classes=input_dim).float()
            true_value = torch.zeros_like(value_estimates).to(device)

            # Compute individual losses
            ppo_loss = ppo_agent.compute_loss(
                state_representation,
                torch.zeros_like(true_actions, dtype=torch.float32).to(device),
                true_actions,
                torch.zeros_like(value_estimates, dtype=torch.float32).to(device),
                torch.zeros_like(value_estimates, dtype=torch.float32).to(device)
            )

            info_nce = InfoNCE_Loss()(
                state_representation.reshape(-1, state_dim),
                F.dropout(state_representation.reshape(-1, state_dim), p=0.1, training=True)
            )


            covariance = CovarianceRegularization()(predicted_next_state_batch.view(-1, predicted_next_state_batch.size(-1)))
            dynamics_loss = DynamicsPerformanceLoss()(state_representation, predicted_next_state_batch)

            perturbed_next_state = predicted_next_state_batch + torch.randn_like(predicted_next_state_batch) * 0.01
            thought_loss = ThoughtConsistencyLoss()(predicted_next_state_batch, perturbed_next_state)

            pv_loss = PolicyValueJointLoss()(policy_logits, true_policy, value_estimates.squeeze(-1), true_value.squeeze(-1))
            action_diversity = ActionDiversityReward()(action_embeddings.view(-1, embed_dim))

            mcts_best_values = torch.zeros(true_actions.size(0)).to(device)
            etv = ExpectedThoughtValueLoss()(mcts_best_values)

            visit_counts = torch.ones(true_actions.size(0), policy_logits.size(-1)).to(device)
            exploration = ExplorationRegularization()(visit_counts)

            old_policy = F.softmax(policy_logits.detach(), dim=-1)
            new_policy = F.softmax(policy_logits, dim=-1)
            kl_loss = KL_DivergenceLoss()(old_policy, new_policy)

            # Total Loss
            loss = (
                ppo_loss +
                info_nce +
                covariance +
                dynamics_loss +
                thought_loss +
                pv_loss +
                action_diversity +
                etv +
                exploration +
                kl_loss
            )
            loss = loss / args.accumulation_steps

        print("Backward pass...")
        scaler.scale(loss).backward()

        if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
            print("Gradient clipping...")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for group in optimizer.param_groups for param in group['params']],
                args.max_grad_norm
            )

            print("Optimizer step...")
            scaler.step(optimizer)
            scaler.update()

            print("Zeroing gradients...")
            optimizer.zero_grad()

            print("Updating learning rate...")
            scheduler.step()

        total_loss += loss.item() * args.accumulation_steps

        # Print individual losses and total loss for this batch
        print(f"Batch {i+1} completed. Losses:")
        print(f"  PPO Loss: {ppo_loss.item():.4f}")
        print(f"  InfoNCE Loss: {info_nce.item():.4f}")
        print(f"  Covariance Loss: {covariance.item():.4f}")
        print(f"  Dynamics Loss: {dynamics_loss.item():.4f}")
        print(f"  Thought Consistency Loss: {thought_loss.item():.4f}")
        print(f"  Policy-Value Loss: {pv_loss.item():.4f}")
        print(f"  Action Diversity Loss: {action_diversity.item():.4f}")
        print(f"  Expected Thought Value Loss: {etv.item():.4f}")
        print(f"  Exploration Loss: {exploration.item():.4f}")
        print(f"  KL Divergence Loss: {kl_loss.item():.4f}")
        print(f"  Total Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"World Model training epoch completed. Average loss: {avg_loss:.4f}")
    return avg_loss



def train_epoch_language_model(model, train_loader, optimizer, scheduler, scaler, args):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    print(f"Starting Language Model training epoch with {len(train_loader)} batches...")

    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast():
            outputs = model(input_ids, input_ids)
            logits = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=model.embedding.padding_idx)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for group in optimizer.param_groups for param in group['params']],
                args.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * args.accumulation_steps
        print(f"Batch {i + 1} completed. Current loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Language Model training epoch completed. Average loss: {avg_loss:.4f}")
    return avg_loss


def train_custom_data_epoch_world_model(world_model_components, train_loader, optimizer, scheduler, scaler, args, model_transformer, state_dim, embed_dim, input_dim):
    representation_network, dynamics_network, prediction_network, action_encoder, ppo_agent, _ = world_model_components
    representation_network.train()
    dynamics_network.train()
    prediction_network.train()
    action_encoder.train()
    ppo_agent.policy_network.train()

    total_loss = 0.0
    optimizer.zero_grad()
    print(f"Starting World Model training epoch with {len(train_loader)} batches...")

    for i, batch in enumerate(train_loader):
        print(f"Processing batch {i+1}/{len(train_loader)}...")

        # Move batches to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        episode_reward = batch['episode_reward'].to(device)
        loss_value = batch['loss'].to(device)
        cosine_similarity = batch['cosine_similarity'].to(device)
        rag_performance = batch['rag_performance'].to(device)
        ranking_model_performance = batch['ranking_model_performance'].to(device)

        with torch.amp.autocast(device_type='cuda'):
            print("Forward pass through Transformer (frozen)...")
            with torch.no_grad():
                transformer_output = model_transformer(input_ids, input_ids)

            # World Model - Representation
            state_representation = representation_network(transformer_output)
            print(f"State representation shape: {state_representation.shape}")

            # For simplicity, let's assume true actions are provided (e.g., next tokens)
            true_actions = input_ids[:, 1:]  # Shift input_ids by 1 to get next tokens
            print(f"True actions shape: {true_actions.shape}")
            action_sequences = true_actions

            # Get action embeddings
            action_embeddings = action_encoder(action_sequences)
            print(f"Action embeddings shape: {action_embeddings.shape}")

            # Ensure state_representation and action_embeddings have the same sequence length
            min_seq_len = min(state_representation.size(1), action_embeddings.size(1))
            state_representation = state_representation[:, :min_seq_len, :]
            action_embeddings = action_embeddings[:, :min_seq_len, :]

            print(f"Adjusted state representation shape: {state_representation.shape}")
            print(f"Adjusted action embeddings shape: {action_embeddings.shape}")

            # Apply dynamics network
            predicted_next_state_batch = dynamics_network(state_representation, action_embeddings)
            print(f"Predicted next state batch shape: {predicted_next_state_batch.shape}")

            # Prediction Network - Policy logits and value
            policy_logits, value_estimates = prediction_network(predicted_next_state_batch)

            # Adjust true_actions to match the sequence length
            true_actions = true_actions[:, :min_seq_len]

            # Define true_policy and true_value
            true_policy = F.one_hot(true_actions, num_classes=input_dim).float()
            true_value = episode_reward.unsqueeze(1).expand(-1, min_seq_len)  # Expand to match sequence length

            # Compute individual losses
            info_nce = InfoNCE_Loss()(
                state_representation.reshape(-1, state_dim),
                F.dropout(state_representation.reshape(-1, state_dim), p=0.1, training=True)
            )

            covariance = CovarianceRegularization()(predicted_next_state_batch.view(-1, predicted_next_state_batch.size(-1)))
            dynamics_loss = DynamicsPerformanceLoss()(state_representation, predicted_next_state_batch)

            perturbed_next_state = predicted_next_state_batch + torch.randn_like(predicted_next_state_batch) * 0.01
            thought_loss = ThoughtConsistencyLoss()(predicted_next_state_batch, perturbed_next_state)

            pv_loss = PolicyValueJointLoss()(policy_logits, true_policy, value_estimates.squeeze(-1), true_value.squeeze(-1))
            action_diversity = ActionDiversityReward()(action_embeddings.view(-1, embed_dim))

            mcts_best_values = torch.zeros(true_actions.size(0)).to(device)
            etv = ExpectedThoughtValueLoss()(mcts_best_values)

            visit_counts = torch.ones(true_actions.size(0), policy_logits.size(-1)).to(device)
            exploration = ExplorationRegularization()(visit_counts)

            old_policy = F.softmax(policy_logits.detach(), dim=-1)
            new_policy = F.softmax(policy_logits, dim=-1)
            kl_loss = KL_DivergenceLoss()(old_policy, new_policy)

            # Compute mean value estimates over the sequence length
            value_estimates_mean = value_estimates.squeeze(-1).mean(dim=1)  # Shape: [batch_size]

            # Add new loss components
            rag_loss = F.mse_loss(value_estimates_mean, rag_performance)
            ranking_loss = F.mse_loss(value_estimates_mean, ranking_model_performance)
            cosine_similarity_loss = 1 - cosine_similarity.mean()  # Maximize cosine similarity

            # Total Loss
            loss = (
                info_nce +
                covariance +
                dynamics_loss +
                thought_loss +
                pv_loss +
                action_diversity +
                etv +
                exploration +
                kl_loss +
                rag_loss +
                ranking_loss +
                cosine_similarity_loss
            )
            loss = loss / args.accumulation_steps

            print("Backward pass...")
            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                print("Gradient clipping...")
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for group in optimizer.param_groups for param in group['params']],
                    args.max_grad_norm
                )

                print("Optimizer step...")
                scaler.step(optimizer)
                scaler.update()

                print("Zeroing gradients...")
                optimizer.zero_grad()

                print("Updating learning rate...")
                scheduler.step()

        # Print individual losses and total loss for this batch
        print(f"Batch {i+1} completed. Losses:")
        print(f"  InfoNCE Loss: {info_nce.item():.4f}")
        print(f"  Covariance Loss: {covariance.item():.4f}")
        print(f"  Dynamics Loss: {dynamics_loss.item():.4f}")
        print(f"  Thought Consistency Loss: {thought_loss.item():.4f}")
        print(f"  Policy-Value Loss: {pv_loss.item():.4f}")
        print(f"  Action Diversity Loss: {action_diversity.item():.4f}")
        print(f"  Expected Thought Value Loss: {etv.item():.4f}")
        print(f"  Exploration Loss: {exploration.item():.4f}")
        print(f"  KL Divergence Loss: {kl_loss.item():.4f}")
        print(f"  RAG Loss: {rag_loss.item():.4f}")
        print(f"  Ranking Loss: {ranking_loss.item():.4f}")
        print(f"  Cosine Similarity Loss: {cosine_similarity_loss.item():.4f}")
        print(f"  Total Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"World Model training epoch completed. Average loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    print("Arguments parsed successfully.")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Save directory created: {args.save_dir}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # Define padding_idx and input dimension based on tokenizer
    padding_idx = tokenizer.pad_token_id
    input_dim = len(tokenizer)


    # Initialize the Transformer model on GPU
    print("Initializing Transformer model...")
    model_transformer = Transformer(
        input_dim=input_dim,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        num_experts=2,
        output_dim=input_dim,
        dropout=0.1,
        top_k=2
    ).to(device)
    model_transformer.train()
    print("Transformer model initialized on device.")

    # Define model parameters (adjusted for speed)
    d_model = 32
    state_dim = 32
    action_dim = d_model
    hidden_dim = 64
    vocab_dim = input_dim
    embed_dim = d_model

    # Define World Model components
    representation_network = RepresentationNetwork(vocab_dim, d_model, state_dim).to(device)
    dynamics_network = DynamicsNetwork(state_dim, action_dim, hidden_dim).to(device)
    prediction_network = PredictionNetwork(state_dim, input_dim, 1).to(device)
    action_encoder = ActionEncoder(input_dim, action_dim).to(device)

    # Initialize PPO Agent
    ppo_agent = PPOAgent(
        policy_network=prediction_network,
        optimizer=optim.AdamW(prediction_network.parameters(), lr=args.learning_rate),
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5
    )

    # Bundle World Model components
    world_model_components = (representation_network, dynamics_network, prediction_network, action_encoder, ppo_agent, model_transformer)

    print(f"Current mode: {args.mode}")
    if args.mode == 'train':
        print("Loading and preprocessing data...")
        if args.use_custom_data:
            custom_data = load_custom_data_from_files(args.custom_data_paths)
            processed_data = preprocess_custom_data(custom_data)
            train_loader, eval_loader = load_custom_data(args, tokenizer, processed_data)
            print("Custom data loaded and preprocessed successfully.")
        else:
            train_loader, eval_loader = load_data(args, tokenizer)
            print("Default data loaded and preprocessed successfully.")

        # Optimizer and Scheduler
        optimizer = optim.AdamW(
            list(representation_network.parameters()) +
            list(dynamics_network.parameters()) +
            list(prediction_network.parameters()) +
            list(action_encoder.parameters()),
            lr=args.learning_rate, weight_decay=args.weight_decay
        ) if args.train_mode == 'world_model' else optim.AdamW(model_transformer.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        scaler = GradScaler()

        print(f"Starting {args.train_mode} training...")

        for epoch in range(args.num_epochs):
            if args.train_mode == 'world_model':
                if args.use_custom_data:
                    avg_loss = train_custom_data_epoch_world_model(
                        world_model_components,
                        train_loader,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        model_transformer,
                        state_dim,
                        embed_dim,
                        input_dim
                    )
                else:
                    avg_loss = train_epoch_world_model(
                        world_model_components,
                        train_loader,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        model_transformer,
                        state_dim,
                        embed_dim,
                        input_dim
                    )
            else:
                avg_loss = train_epoch_language_model(
                    model_transformer,
                    train_loader,
                    optimizer,
                    scheduler,
                    scaler,
                    args
                )

            print(f"{args.train_mode.capitalize()} training epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

            # Save models
            if args.train_mode == 'world_model':
                save_all_models(model_transformer, representation_network, dynamics_network, prediction_network, action_encoder, args.save_dir, epoch + 1)
                print(f"Models saved for epoch {epoch + 1}")
            else:
                torch.save(model_transformer.state_dict(), os.path.join(args.save_dir, f'language_model_epoch_{epoch + 1}.pt'))
                print(f"Language model saved for epoch {epoch + 1}")

        print("Training completed.")

    elif args.mode == 'inference':
        print("Entering inference mode...")
        # Build Tree of Thought if needed
        print("Building Tree of Thought...")
        tree_root = build_tree_of_thought()
        print("Tree of Thought built successfully.")

        # Generate action list
        print("Generating action list...")
        action_list = []
        traverse_tree(tree_root, action_list)
        print(f"Action list generated. Total actions: {len(action_list)}")

        # Create mappings
        global action_to_index, index_to_action
        action_to_index = {action: idx for idx, action in enumerate(action_list)}
        index_to_action = {idx: action for action, idx in action_to_index.items()}
        action_vocab_size = len(action_list)
        print(f"Action mappings created. Vocabulary size: {action_vocab_size}")

        # Initialize or load models based on the load_model argument
        if args.load_model:
            print(f"Loading saved model from {args.load_model}")
            # Load the saved models
            model_transformer.load_state_dict(torch.load(os.path.join(args.load_model, 'transformer_model.pt')))
            representation_network.load_state_dict(torch.load(os.path.join(args.load_model, 'representation_network.pt')))
            dynamics_network.load_state_dict(torch.load(os.path.join(args.load_model, 'dynamics_network.pt')))

            # Load prediction network and adjust its size if necessary
            saved_state_dict = torch.load(os.path.join(args.load_model, 'prediction_network.pt'))
            saved_vocab_size = saved_state_dict['policy_head.weight'].size(0)
            if saved_vocab_size != action_vocab_size:
                print(f"Adjusting prediction network size from {saved_vocab_size} to {action_vocab_size}")
                prediction_network = PredictionNetwork(state_dim, saved_vocab_size, 1).to(device)
                prediction_network.load_state_dict(saved_state_dict)
                prediction_network.policy_head = nn.Linear(prediction_network.state_dim, action_vocab_size).to(device)
            else:
                prediction_network = PredictionNetwork(state_dim, action_vocab_size, 1).to(device)
                prediction_network.load_state_dict(saved_state_dict)

            action_encoder.load_state_dict(torch.load(os.path.join(args.load_model, 'action_encoder.pt')))
        else:
            print("Using newly initialized models")

        # Prepare the components
        world_model_components = (representation_network, dynamics_network, prediction_network, action_encoder, ppo_agent, model_transformer)

        print("Starting inference loop...")
        while True:
            if args.query:
                query = args.query
                args.query = None  # Reset query for next iteration
            else:
                query = input("Please enter your query (or type 'exit' to quit): ")
                if query.lower() == 'exit':
                    break

            print(f"Processing query: {query}")
            result = infer(query, world_model_components, tree_root, tokenizer,
                          max_length=args.max_length,
                          inference_mode=args.inference_mode,
                          beam_size=args.beam_size,
                          n_tokens_predict=args.n_tokens_predict,
                          mcts_iterations=args.mcts_iterations,
                          exploration_constant=args.mcts_exploration_constant)


            if args.inference_mode == 'without_world_model':
                print("Generated Text:")
                print(result)
            else:
                print("Generated Thought Sequence:")
                for thought in result:
                    print(thought)

            print("\n")  # Add a newline for better readability between queries

        print("Inference completed.")

    else:
        print(f"Invalid mode: {args.mode}. Please choose 'train' or 'inference'.")
if __name__ == '__main__':
    sys.argv = [
        'lightbulb_2.py',
        '--mode', 'inference',
        '--train_mode', 'world_model',  # Set 'world_model' or 'language_model' depending on the training mode
        '--dataset_name', 'wikitext',   # Specify the Hugging Face dataset (e.g., 'wikitext')
        '--dataset_config', 'wikitext-2-raw-v1',  # Use if you need a specific config of the dataset
        '--num_epochs', '10',
        '--batch_size', '4',
        '--accumulation_steps', '1',
        '--max_grad_norm', '1.0',
        '--weight_decay', '0.01',
        '--learning_rate', '1e-4',
        '--max_length', '512',
        '--save_dir', './trained_models',
        # Uncomment the following line to use custom data instead of a Hugging Face dataset
        #'--use_custom_data',
        '--custom_data_paths', '/content/drive/MyDrive/lightbulb/knowledge_base.json',
        '--custom_data_paths', '/content/drive/MyDrive/lightbulb/rag_cache.json',
        '--custom_data_paths', '/content/drive/MyDrive/lightbulb/llm_training_data/llm_training_data.jsonl'
    ]

    # Parse the arguments and run the main training function
    args = parse_args()

    # Check which data source to use
    if args.use_custom_data:
        print("Training with custom data from paths:")
        for path in args.custom_data_paths:
            print(f"  - {path}")
    else:
        print(f"Training with dataset '{args.dataset_name}' from Hugging Face Datasets")

    main()

