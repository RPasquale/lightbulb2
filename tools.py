import os
import sys
import random
import math
import argparse
import json
import jsonlines
import csv
from io import StringIO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Helper function to move tensors to the correct device
def to_device(tensor):
    return tensor.to(device)

# Custom Components
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        
    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)

class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, top_k):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_experts)])
        
    def forward(self, x):
        gate_output = self.gate(x)
        top_k_gates, top_k_indices = torch.topk(gate_output, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            mask = top_k_indices == i
            output += expert_output * (top_k_gates * mask.float()).sum(dim=-1, keepdim=True)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, num_experts, output_dim, dropout, top_k):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = RotaryPositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.moe = MoE(d_model, d_model, num_experts, top_k)
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        embedded_src = self.embedding(src) + self.positional_encoding(src)
        embedded_tgt = self.embedding(tgt) + self.positional_encoding(tgt)
        
        x = embedded_src
        for block in self.transformer_blocks:
            x = block(x, src_mask)
        
        x = self.moe(x)
        output = self.output_layer(x)
        return output

# Custom Loss Functions and Components
class InfoNCE_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t().contiguous()) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool).fill_diagonal_(0)
        negative_samples = sim[mask].reshape(2 * batch_size, -1)
        labels = torch.zeros(2 * batch_size).to(positive_samples.device).long()
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss

class CovarianceRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size, num_features = x.size()
        mean = torch.mean(x, dim=0)
        x_centered = x - mean.unsqueeze(0)
        cov = torch.matmul(x_centered.t(), x_centered) / (batch_size - 1)
        off_diagonal = cov - torch.diag(torch.diag(cov))
        return torch.sum(off_diagonal ** 2)

class DynamicsPerformanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted_state, true_state):
        return F.mse_loss(predicted_state, true_state)

class ThoughtConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, thought1, thought2):
        return 1 - F.cosine_similarity(thought1, thought2).mean()

class PolicyValueJointLoss(nn.Module):
    def __init__(self, value_loss_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def forward(self, logits, true_policy, value, true_value):
        policy_loss = F.cross_entropy(logits, true_policy)
        value_loss = F.mse_loss(value, true_value)
        entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        return policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

class ActionDiversityReward(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, action_embeddings):
        similarities = F.cosine_similarity(action_embeddings.unsqueeze(1), action_embeddings.unsqueeze(0), dim=-1)
        diversity = 1 - similarities.mean()
        return diversity

class ExpectedThoughtValueLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mcts_values):
        return -torch.mean(mcts_values)

class ExplorationRegularization(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        
    def forward(self, visit_counts):
        probs = visit_counts / visit_counts.sum(dim=-1, keepdim=True)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return -self.c * entropy.mean()

class KL_DivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, p, q):
        return F.kl_div(p.log(), q, reduction='batchmean')

class ActionEncoder(nn.Module):
    def __init__(self, action_vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(action_vocab_size, embed_dim)
        
    def forward(self, action_indices):
        return self.embedding(action_indices)

class RepresentationNetwork(nn.Module):
    def __init__(self, vocab_dim, d_model, state_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, d_model)
        self.lstm = nn.LSTM(d_model, state_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        return h_n.squeeze(0)

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class PredictionNetwork(nn.Module):
    def __init__(self, state_dim, action_vocab_size, value_dim):
        super().__init__()
        self.policy_head = nn.Linear(state_dim, action_vocab_size)
        self.value_head = nn.Linear(state_dim, value_dim)
        
    def forward(self, state):
        policy_logits = self.policy_head(state)
        value = self.value_head(state)
        return policy_logits, value

class ThoughtNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children is not None else []
        self.visits = 0
        self.value = 0
        
    def add_child(self, child):
        self.children.append(child)

class MCTS:
    def __init__(self, prediction_network, dynamics_network, action_encoder, action_to_index, num_iterations, exploration_constant, beam_size, n_tokens_predict):
        self.prediction_network = prediction_network
        self.dynamics_network = dynamics_network
        self.action_encoder = action_encoder
        self.action_to_index = action_to_index
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.beam_size = beam_size
        self.n_tokens_predict = n_tokens_predict
        
    def search_with_beam(self, state):
        root = ThoughtNode("root")
        for _ in range(self.num_iterations):
            leaf = self.select(root)
            value = self.expand_and_evaluate(leaf, state)
            self.backpropagate(leaf, value)
        
        best_child = max(root.children, key=lambda c: c.visits)
        return [best_child.name]
    
    def select(self, node):
        while node.children:
            if not all(child.visits > 0 for child in node.children):
                return self.expand(node)
            node = self.get_best_child(node)
        return node
    
    def expand(self, node):
        actions = list(self.action_to_index.keys())
        for action in actions:
            if action not in [child.name for child in node.children]:
                new_node = ThoughtNode(action)
                node.add_child(new_node)
                return new_node
        return random.choice(node.children)
    
    def expand_and_evaluate(self, node, state):
        action_index = self.action_to_index[node.name]
        action_embedding = self.action_encoder(torch.tensor([action_index], device=state.representation.device))
        next_state = self.dynamics_network(state.representation, action_embedding)
        policy_logits, value = self.prediction_network(next_state)
        return value.item()
    
    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def get_best_child(self, node):
        return max(node.children, key=lambda c: c.value / c.visits + self.exploration_constant * math.sqrt(math.log(node.visits) / c.visits))

class State:
    def __init__(self, representation, dynamics_network, action_encoder, action_to_index, thought_node):
        self.representation = representation
        self.dynamics_network = dynamics_network
        self.action_encoder = action_encoder
        self.action_to_index = action_to_index
        self.thought_node = thought_node

class PPOAgent:
    def __init__(self, policy_network, optimizer, clip_epsilon, entropy_coef, value_coef):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
    
    def compute_loss(self, states, old_log_probs, actions, returns, advantages):
        policy_logits, values = self.policy_network(states)
        new_log_probs = F.log_softmax(policy_logits, dim=-1).gather(1, actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        entropy = -(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)).sum(-1).mean()
        
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total_loss
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

# Utility Functions
def split_into_chunks(text, max_words=256):
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def compute_attention_weights(current_step_desc, previous_outputs, vectorizer):
    if not previous_outputs:
        return np.array([])
    documents = [current_step_desc] + previous_outputs
    tfidf_matrix = vectorizer.fit_transform(documents)
    current_vector = tfidf_matrix[0:1]
    previous_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(current_vector, previous_vectors)[0]
    if similarities.sum() == 0:
        weights = np.array([1.0 / len(similarities)] * len(similarities))
    else:
        weights = similarities / similarities.sum()
    return weights

def generate_options_for_step(prompt, tokenizer, model, device, num_options=3, max_new_tokens=150):
    options = []
    input_encoding = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = to_device(input_encoding['input_ids'])
    attention_mask = to_device(input_encoding['attention_mask'])
    
    try:
        for idx in range(num_options):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            option_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            option = option_text[len(prompt):].strip()
            option_label = f"{chr(65 + idx)})"
            if option.startswith(('A)', 'B)', 'C)', 'a)', 'b)', 'c)')):
                option = option.split(')', 1)[1].strip()
            option = f"{option_label} {option}"
            options.append(option)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
    return options

def evaluate_options(current_step_desc, options, vectorizer):
    rewards = []
    for option in options:
        documents = [current_step_desc, option]
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        reward = similarity
        rewards.append(reward)
    return rewards

def generate_instruction(answer_text, tokenizer, model, device):
    prompt = "Write an instruction that would lead to the following answer:\n" + answer_text
    input_encoding = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = to_device(input_encoding['input_ids'])
    attention_mask = to_device(input_encoding['attention_mask'])
    
    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        instruction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.info(f"Generated Instruction: {instruction}")
    except Exception as e:
        logger.error(f"Error during instruction generation: {e}")
        instruction = ""
    return instruction

# Data Loading Functions
def get_data_seeds_from_pdfs():
    logger.info("You have selected PDFs.")
    pdf_paths_input = input("Enter the paths to your PDF files, separated by commas: ").strip()
    if not pdf_paths_input:
        logger.info("No PDF paths provided. Skipping PDF data source.")
        return []
    f = StringIO(pdf_paths_input)
    reader = csv.reader(f, skipinitialspace=True)
    pdf_paths = next(reader, [])
    data_seeds = []
    for pdf_path in pdf_paths:
        pdf_path = pdf_path.strip('"').strip("'")
        if not os.path.isfile(pdf_path):
            logger.warning(f"File not found: {pdf_path}. Skipping.")
            continue
        try:
            with open(pdf_path, 'rb') as f_pdf:
                reader_pdf = PyPDF2.PdfReader(f_pdf)
                text = ''
                for page_num, page in enumerate(reader_pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + ' '
                    else:
                        logger.warning(f"No text found on page {page_num} of {pdf_path}.")
                sentences = text.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        chunks = split_into_chunks(sentence)
                        data_seeds.extend(chunks)
                logger.info(f"Extracted {len(data_seeds)} data seeds from {pdf_path}.")
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            continue
    logger.info(f"Total extracted data seeds from PDFs: {len(data_seeds)}")
    return data_seeds

def get_data_seeds_from_huggingface_dataset():
    logger.info("You have selected Hugging Face Dataset.")
    dataset_name = input("Enter the name of the Hugging Face dataset (e.g., 'squad'): ").strip()
    if not dataset_name:
        logger.warning("Dataset name cannot be empty. Skipping Hugging Face dataset.")
        return []
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        error_message = str(e)
        if 'trust_remote_code' in error_message or 'custom code' in error_message:
            logger.error(f"Error loading dataset '{dataset_name}': {e}")
            trust = input("This dataset requires 'trust_remote_code=True'. Do you want to proceed? (y/n): ").strip().lower()
            if trust == 'y':
                try:
                    dataset = load_dataset(dataset_name, trust_remote_code=True)
                except Exception as e2:
                    logger.error(f"Failed to load dataset with 'trust_remote_code=True': {e2}")
                    return []
            else:
                logger.info("Skipping Hugging Face dataset.")
                return []
        else:
            logger.error(f"Error loading dataset '{dataset_name}': {e}")
            return []
    try:
        if 'train' in dataset:
            split = 'train'
        else:
            split = list(dataset.keys())[0]
            logger.info(f"No 'train' split found. Using '{split}' split.")
        data_seeds = []
        for example in dataset[split]:
            if 'text' in example:
                chunks = split_into_chunks(str(example['text']))
                data_seeds.extend(chunks)
            elif 'content' in example:
                chunks = split_into_chunks(str(example['content']))
                data_seeds.extend(chunks)
            elif 'question' in example:
                chunks = split_into_chunks(str(example['question']))
                data_seeds.extend(chunks)
            else:
                for value in example.values():
                    if isinstance(value, str):
                        chunks = split_into_chunks(value)
                        data_seeds.extend(chunks)
        logger.info(f"Extracted {len(data_seeds)} data seeds from dataset '{dataset_name}'.")
        return data_seeds
    except Exception as e:
        logger.error(f"Error processing dataset '{dataset_name}': {e}")
        return []

def get_data_seeds_from_npy_files():
    logger.info("You have selected .npy Files.")
    npy_paths_input = input("Enter the paths to your .npy files, separated by commas: ").strip()
    if not npy_paths_input:
        logger.info("No .npy paths provided. Skipping .npy data source.")
        return []
    f = StringIO(npy_paths_input)
    reader = csv.reader(f, skipinitialspace=True)
    npy_paths = next(reader, [])
    data_seeds = []
    for npy_path in npy_paths:
        npy_path = npy_path.strip('"').strip("'")
        if not os.path.isfile(npy_path):
            logger.warning(f"File not found: {npy_path}. Skipping.")
            continue
        try:
            data = np.load(npy_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                data = data.flatten()
            for item in data:
                if isinstance(item, (str, np.str_)):
                    chunks = split_into_chunks(item)
                    data_seeds.extend(chunks)
            logger.info(f"Extracted {len(data_seeds)} data seeds from {npy_path}.")
        except Exception as e:
            logger.error(f"Error reading .npy file {npy_path}: {e}")
            continue
    logger.info(f"Total extracted data seeds from .npy files: {len(data_seeds)}")
    return data_seeds

# Dataset Classes
class DataSeedDataset(Dataset):
    def __init__(self, data_seeds, tokenizer, max_length=512):
        self.data_seeds = data_seeds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_seeds)

    def __getitem__(self, idx):
        data_seed = self.data_seeds[idx]
        encoding = self.tokenizer(
            data_seed,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class RLHFDataset(Dataset):
    def __init__(self, rlhf_data, tokenizer, max_length=512):
        self.data = rlhf_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry['prompt']
        correct_option = entry['correct_option'].strip()
        incorrect_option = entry['incorrect_option'].strip()

        options_text = f"{correct_option}\n{incorrect_option}"
        combined_text = f"{prompt}\nOptions:\n{options_text}"

        encoding = self.tokenizer(
            combined_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        option_label = correct_option.split(')')[0].strip().upper()
        label_map = {chr(65 + i): i for i in range(26)}
        label = label_map.get(option_label, 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'correct_option': correct_option,
            'incorrect_option': incorrect_option
        }

class InstructionDataset(Dataset):
    def __init__(self, instruction_data, tokenizer, max_length=512):
        self.data = instruction_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        instruction = entry['instruction']
        answer = entry['answer']

        encoding = self.tokenizer(
            instruction,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        target_encoding = self.tokenizer(
            answer,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = target_encoding['input_ids'].clone()
        labels[target_encoding['input_ids'] == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels.squeeze(0),
            'instruction': instruction,
            'answer': answer
        }

def load_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None

    checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    latest_checkpoint = os.path.join(output_dir, checkpoint_files[0])
    logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    return torch.load(latest_checkpoint)

# Main Data Generation Function
def generate_data(args):
    logger.info("Starting data generation process...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_path)
    logger.info(f"Model Configuration: {config}")

    model_type = config.model_type
    logger.info(f"Detected model type: {model_type}")

    if model_type in ['gpt2', 'gpt', 'gemma2']:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    elif model_type in ['t5', 'bart', 'mbart']:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
        logger.info(f"Assuming model type '{model_type}' is a causal language model. Using AutoModelForCausalLM.")

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Define steps for data generation
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
        3: {
            'description': "Generate key points for the most important parts of the data about {data_seed}, given outputs of steps (1) and (2).",
            'dependencies': [1, 2],
            'weight': 0.8
        },
        4: {
            'description': "Explain and analyze any relationships in the data about {data_seed}.",
            'dependencies': [1, 2, 3],
            'weight': 0.7
        },
        5: {
            'description': "Generate insights about the data about {data_seed}.",
            'dependencies': [1, 2, 3, 4],
            'weight': 0.6
        },
        6: {
            'description': "Propose actions to be completed for the data about {data_seed}.",
            'dependencies': [1, 2, 3, 4, 5],
            'weight': 0.5
        }
    }

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Extract data seeds from selected sources
    data_seeds = []
    if args.use_pdfs:
        pdf_seeds = get_data_seeds_from_pdfs()
        data_seeds.extend(pdf_seeds)
    if args.use_huggingface:
        hf_seeds = get_data_seeds_from_huggingface_dataset()
        data_seeds.extend(hf_seeds)
    if args.use_npy:
        npy_seeds = get_data_seeds_from_npy_files()
        data_seeds.extend(npy_seeds)

    if not data_seeds:
        logger.error("No data seeds extracted from the selected sources. Exiting.")
        return

    # Remove duplicates and shuffle
    data_seeds = list(set(data_seeds))
    random.shuffle(data_seeds)

    # Compute total number of words and determine number of data seeds
    total_words = sum(len(seed.split()) for seed in data_seeds)
    max_words_per_seed = 256
    num_seeds = total_words // max_words_per_seed
    if total_words % max_words_per_seed != 0:
        num_seeds += 1
    logger.info(f"Total words extracted: {total_words}")
    logger.info(f"Each data seed will have up to {max_words_per_seed} words.")
    logger.info(f"Number of data seeds to generate: {num_seeds}")

    # If the actual number of data_seeds is greater than num_seeds, truncate
    if len(data_seeds) > num_seeds:
        data_seeds = data_seeds[:num_seeds]
        logger.info(f"Truncated data seeds to {num_seeds} for proportionality.")

    logger.info(f"Using {len(data_seeds)} data seeds for data generation.")

    # Prepare filenames
    rlhf_filename = f"{args.dataset_name}_rlhf.jsonl"
    instruction_filename = f"{args.dataset_name}_instruction.jsonl"

    # Check if the files already exist and load existing data
    existing_rlhf_data = []
    existing_instruction_data = []

    if os.path.exists(rlhf_filename):
        logger.info(f"Found existing RLHF data file: {rlhf_filename}. Loading existing data.")
        with jsonlines.open(rlhf_filename, mode='r') as reader:
            existing_rlhf_data = [entry for entry in reader]

    if os.path.exists(instruction_filename):
        logger.info(f"Found existing Instruction data file: {instruction_filename}. Loading existing data.")
        with jsonlines.open(instruction_filename, mode='r') as reader:
            existing_instruction_data = [entry for entry in reader]

    # Keep track of data seeds already processed
    processed_data_seeds = set([entry['data_seed'] for entry in existing_rlhf_data])

    # Open files for appending
    rlhf_file = jsonlines.open(rlhf_filename, mode='a')
    instruction_file = jsonlines.open(instruction_filename, mode='a')

    try:
        # Generate RLHF data for each data seed and each step
        for idx, data_seed in enumerate(data_seeds, start=1):
            if data_seed in processed_data_seeds:
                logger.info(f"Data Seed {idx}/{len(data_seeds)} already processed. Skipping.")
                continue
            logger.info(f"Data Seed {idx}/{len(data_seeds)}: {data_seed}")
            previous_steps_outputs = []
            for step_num in range(1, len(steps) + 1):
                step_info = steps[step_num]
                step_text = step_info['description'].format(data_seed=data_seed)

                attention_weights = compute_attention_weights(step_text, previous_steps_outputs, vectorizer)

                if len(attention_weights) > 0:
                    weighted_outputs = []
                    for weight, output in zip(attention_weights, previous_steps_outputs):
                        weighted_output = f"[Weight: {weight:.2f}] {output}"
                        weighted_outputs.append(weighted_output)
                    prompt = step_text + "\n" + "\n".join(weighted_outputs) + "\nOptions:"
                else:
                    prompt = step_text + "\nOptions:"

                options = generate_options_for_step(prompt, tokenizer, model, device)
                if not options:
                    logger.warning(f"Failed to generate options for Step {step_num} of Data Seed '{data_seed}'. Skipping...")
                    continue

                current_step_desc = step_info['description'].format(data_seed=data_seed)
                rewards = evaluate_options(current_step_desc, options, vectorizer)

                sorted_indices = np.argsort(rewards)[::-1]
                correct_option = options[sorted_indices[0]]
                incorrect_option = options[sorted_indices[-1]]

                rlhf_entry = {
                    'data_seed': data_seed,
                    'step_num': step_num,
                    'step_text': step_text,
                    'prompt': prompt,
                    'correct_option': correct_option,
                    'incorrect_option': incorrect_option,
                    'attention_weights': attention_weights.tolist() if len(attention_weights) > 0 else []
                }
                rlhf_file.write(rlhf_entry)
                logger.info(f"RLHF Entry Written for Step {step_num}.")

                previous_steps_outputs.append(correct_option)

                instruction = generate_instruction(correct_option, tokenizer, model, device)

                if instruction:
                    instruction_entry = {
                        'instruction': instruction,
                        'answer': correct_option
                    }
                    instruction_file.write(instruction_entry)
                    logger.info(f"Instruction Entry Written for Step {step_num}.")
                else:
                    logger.warning(f"Failed to generate instruction for Step {step_num}. Skipping instruction saving.")

                logger.info(f"Step {step_num}:\n{step_text}")
                logger.info(f"Correct Option:\n{correct_option}")
                logger.info(f"Incorrect Option:\n{incorrect_option}")
                logger.info("-" * 80)
            logger.info("=" * 100)
    except KeyboardInterrupt:
        logger.info("Data generation interrupted by user.")
    finally:
        rlhf_file.close()
        instruction_file.close()
        logger.info(f"RLHF dataset saved to {rlhf_filename}")
        logger.info(f"Instruction dataset saved to {instruction_filename}")

# Main Distillation Function
def distill_model(args):
    logger.info("Starting model distillation process...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher model
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
    for param in teacher.parameters():
        param.requires_grad = False

    # Load or initialize student model
    try:
        student = AutoModelForCausalLM.from_pretrained(args.student_model).to(device)
        logger.info(f"Student model '{args.student_model}' loaded successfully.")
    except (OSError, ValueError):
        logger.info(f"Student model '{args.student_model}' not found. Initializing a new student model.")
        student = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
        student.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        logger.info(f"New student model saved to '{args.save_path}'.")

    # Load datasets
    rlhf_data = []
    instruction_data = []
    with jsonlines.open(args.rlhf_data_path, mode='r') as reader:
        rlhf_data = [obj for obj in reader]
    with jsonlines.open(args.instruction_data_path, mode='r') as reader:
        instruction_data = [obj for obj in reader]

    # Extract data seeds from RLHF data
    data_seeds = list(set([entry['data_seed'] for entry in rlhf_data if 'data_seed' in entry]))

    # Create datasets
    data_seed_dataset = DataSeedDataset(data_seeds, tokenizer, max_length=args.max_length)
    rlhf_dataset = RLHFDataset(rlhf_data, tokenizer, max_length=args.max_length)
    instruction_dataset = InstructionDataset(instruction_data, tokenizer, max_length=args.max_length)

    # Split datasets into train and validation
    train_size_form1 = int(0.8 * len(data_seed_dataset))
    val_size_form1 = len(data_seed_dataset) - train_size_form1
    train_data_seed, val_data_seed = random_split(data_seed_dataset, [train_size_form1, val_size_form1])

    train_size_form2 = int(0.8 * len(rlhf_dataset))
    val_size_form2 = len(rlhf_dataset) - train_size_form2
    train_rlhf, val_rlhf = random_split(rlhf_dataset, [train_size_form2, val_size_form2])

    train_size_form3 = int(0.8 * len(instruction_dataset))
    val_size_form3 = len(instruction_dataset) - train_size_form3
    train_instruction, val_instruction = random_split(instruction_dataset, [train_size_form3, val_size_form3])

    # Create DataLoaders
    train_loader_form1 = DataLoader(train_data_seed, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader_form1 = DataLoader(val_data_seed, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loader_form2 = DataLoader(train_rlhf, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader_form2 = DataLoader(val_rlhf, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loader_form3 = DataLoader(train_instruction, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader_form3 = DataLoader(val_instruction, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize distillation components
    d_model = teacher.config.hidden_size
    vocab_dim = tokenizer.vocab_size
    state_dim = 512
    embed_dim = 256
    hidden_dim = 1024
    value_dim = 1

    representation_network = RepresentationNetwork(vocab_dim, d_model, state_dim).to(device)
    action_encoder = ActionEncoder(vocab_dim, embed_dim).to(device)
    dynamics_network = DynamicsNetwork(state_dim, embed_dim, hidden_dim).to(device)
    prediction_network = PredictionNetwork(state_dim, vocab_dim, value_dim).to(device)
    
    # Initialize Transformer
    student_transformer = Transformer(
        input_dim=vocab_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        output_dim=vocab_dim,
        dropout=0.1,
        top_k=args.top_k
    ).to(device)

    # Initialize optimizers
    optimizer_student = torch.optim.AdamW(student_transformer.parameters(), lr=args.learning_rate)
    optimizer_world_model = torch.optim.AdamW(list(representation_network.parameters()) +
                                              list(action_encoder.parameters()) +
                                              list(dynamics_network.parameters()) +
                                              list(prediction_network.parameters()),
                                              lr=args.learning_rate)

    # Initialize PPO agent
    ppo_agent = PPOAgent(
        policy_network=prediction_network,
        optimizer=optimizer_world_model,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef
    )

    # Initialize schedulers
    total_steps = (len(train_loader_form1) + len(train_loader_form2) + len(train_loader_form3)) * args.epochs
    scheduler_student = get_linear_schedule_with_warmup(
        optimizer_student,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scheduler_world_model = get_linear_schedule_with_warmup(
        optimizer_world_model,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Initialize loss functions
    nll_loss = nn.NLLLoss(ignore_index=-100)
    mse_loss = nn.MSELoss()

    # Initialize MCTS
    mcts = MCTS(
        prediction_network=prediction_network,
        dynamics_network=dynamics_network,
        action_encoder=action_encoder,
        action_to_index={chr(i + 65): i for i in range(26)},  # A-Z mapped to 0-25
        num_iterations=10,
        exploration_constant=math.sqrt(2),
        beam_size=5,
        n_tokens_predict=1
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Training
        student_transformer.train()
        representation_network.train()
        action_encoder.train()
        dynamics_network.train()
        prediction_network.train()

        total_loss = 0
        for batch in tqdm(train_loader_form1, desc="Form 1 Training"):
            optimizer_student.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = student_transformer(input_ids, input_ids, src_mask=attention_mask, tgt_mask=attention_mask)
            loss = nll_loss(outputs.view(-1, vocab_dim), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_transformer.parameters(), args.max_grad_norm)
            optimizer_student.step()
            scheduler_student.step()
            
            total_loss += loss.item()

        for batch in tqdm(train_loader_form2, desc="Form 2 Training"):
            optimizer_world_model.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                teacher_hidden_states = teacher_outputs.last_hidden_state
            
            state_representation = representation_network(teacher_hidden_states)
            action_embeddings = action_encoder(labels)
            next_state_prediction = dynamics_network(state_representation, action_embeddings)
            policy_logits, value_estimates = prediction_network(next_state_prediction)
            
            ppo_loss = ppo_agent.compute_loss(state_representation, torch.log_softmax(policy_logits, dim=-1),
                                              labels, value_estimates, torch.zeros_like(value_estimates))
            
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(representation_network.parameters()) +
                                           list(action_encoder.parameters()) +
                                           list(dynamics_network.parameters()) +
                                           list(prediction_network.parameters()),
                                           args.max_grad_norm)
            optimizer_world_model.step()
            scheduler_world_model.step()
            
            total_loss += ppo_loss.item()

        for batch in tqdm(train_loader_form3, desc="Form 3 Training"):
            optimizer_student.zero_grad()
            optimizer_world_model.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            student_outputs = student_transformer(input_ids, input_ids, src_mask=attention_mask, tgt_mask=attention_mask)
            teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
            
            distillation_loss = F.kl_div(
                F.log_softmax(student_outputs / args.temperature, dim=-1),
                F.softmax(teacher_outputs.logits / args.temperature, dim=-1),
                reduction='batchmean'
            ) * (args.temperature ** 2)
            
            nll = nll_loss(student_outputs.view(-1, vocab_dim), labels.view(-1))
            loss = distillation_loss + nll
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(student_transformer.parameters()) +
                                           list(representation_network.parameters()) +
                                           list(action_encoder.parameters()) +
                                           list(dynamics_network.parameters()) +
                                           list(prediction_network.parameters()),
                                           args.max_grad_norm)
            optimizer_student.step()
            optimizer_world_model.step()
            scheduler_student.step()
            scheduler_world_model.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / (len(train_loader_form1) + len(train_loader_form2) + len(train_loader_form3))
        logger.info(f"Average training loss: {avg_loss:.4f}")

        # Validation
        student_transformer.eval()
        representation_network.eval()
        action_encoder.eval()
        dynamics_network.eval()
        prediction_network.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader_form1, desc="Form 1 Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = student_transformer(input_ids, input_ids, src_mask=attention_mask, tgt_mask=attention_mask)
                loss = nll_loss(outputs.view(-1, vocab_dim), labels.view(-1))
                
                total_val_loss += loss.item()

            for batch in tqdm(val_loader_form2, desc="Form 2 Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                teacher_hidden_states = teacher_outputs.last_hidden_state
                
                state_representation = representation_network(teacher_hidden_states)
                action_embeddings = action_encoder(labels)
                next_state_prediction = dynamics_network(state_representation, action_embeddings)
                policy_logits, value_estimates = prediction_network(next_state_prediction)
                
                ppo_loss = ppo_agent.compute_loss(state_representation, torch.log_softmax(policy_logits, dim=-1),
                                                  labels, value_estimates, torch.zeros_like(value_estimates))
                
                total_val_loss += ppo_loss.item()

            for batch in tqdm(val_loader_form3, desc="Form 3 Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                student_outputs = student_transformer(input_ids, input_ids, src_mask=attention_mask, tgt_mask=attention_mask)
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                
                distillation_loss = F.kl_div(
                    F.log_softmax(student_outputs / args.temperature, dim=-1),
                    F.softmax(teacher_outputs.logits / args.temperature, dim=-1),
                    reduction='batchmean'
                ) * (args.temperature ** 2)
                
                nll = nll_loss(student_outputs.view(-1, vocab_dim), labels.view(-1))
                loss = distillation_loss + nll
                
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / (len(val_loader_form1) + len(val_loader_form2) + len(val_loader_form3))
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'student_transformer_state_dict': student_transformer.state_dict(),
                'representation_network_state_dict': representation_network.state_dict(),
                'action_encoder_state_dict': action_encoder.state_dict(),
                'dynamics_network_state_dict': dynamics_network.state_dict(),
                'prediction_network_state_dict': prediction_network.state_dict(),
                'optimizer_student_state_dict': optimizer_student.state_dict(),
                'optimizer_world_model_state_dict': optimizer_world_model.state_dict(),
                'scheduler_student_state_dict': scheduler_student.state_dict(),
                'scheduler_world_model_state_dict': scheduler_world_model.state_dict(),
                'best_val_loss': best_val_loss
            }, f"{args.save_path}/best_model.pt")
            logger.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    logger.info("Training completed.")


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import logging
import argparse
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Integrated LLM Distillation and World Model Training")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=['generate', 'distill', 'integrated'], required=True, help="Mode of operation")
    
    # Model paths
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model")
    parser.add_argument("--student_model", type=str, required=True, help="Path to the student model")
    
    # Data sources
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset")
    parser.add_argument("--use_pdfs", action="store_true", help="Use PDFs as additional data source")
    parser.add_argument("--use_npy", action="store_true", help="Use .npy files as additional data source")
    
    # Output and checkpointing
    parser.add_argument("--save_path", type=str, default="./saved_models", help="Path to save the models")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping")
    
    # Model configuration
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model")
    parser.add_argument("--d_ff", type=int, default=3072, help="Dimension of the feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Distillation parameters
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
    
    # World Model parameters
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts for MoE in World Model")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k experts to use in MoE")
    
    # PPO parameters
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient for PPO")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient for PPO")
    
    # MCTS parameters
    parser.add_argument("--num_mcts_iterations", type=int, default=10, help="Number of MCTS iterations")
    parser.add_argument("--mcts_c_puct", type=float, default=1.0, help="Exploration constant for MCTS")
    
    # Data generation parameters
    parser.add_argument("--num_generate_samples", type=int, default=1000, help="Number of samples to generate for training")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter for generation")
    parser.add_argument("--temperature_generate", type=float, default=0.7, help="Temperature for text generation")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose logging")
    
    # New arguments for memory optimization
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.mode == 'integrated':
        integrated_training(args)
    elif args.mode == 'generate':
        generate_data(args)
    elif args.mode == 'distill':
        distill_model(args)
    else:
        logger.error("Invalid mode. Choose 'generate', 'distill', or 'integrated'.")

def integrated_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    dataset = load_dataset(args.dataset_name)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quantize,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if args.quantize else None
    
    # Initialize models
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=bnb_config,
        device_map="auto" if args.quantize else None,
        attn_implementation='eager',  # Use eager attention implementation
        use_cache=False  # Disable caching for compatibility with gradient checkpointing
    )
    teacher_model.gradient_checkpointing_enable()
    
    student_lm = initialize_student_lm(args, bnb_config)
    world_model = initialize_world_model(args, bnb_config)
    
    # Move models to GPU
    teacher_model = teacher_model.to(device)
    student_lm = student_lm.to(device)
    world_model = world_model.to(device)
    
    # Create data loader with pinned memory
    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    # Initialize optimizers
    optimizer_lm = torch.optim.AdamW(student_lm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_wm = torch.optim.AdamW(world_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize schedulers
    total_steps = len(train_loader) * args.epochs
    scheduler_lm = get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    scheduler_wm = get_linear_schedule_with_warmup(optimizer_wm, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    
    # Training loop
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            # Move batch to GPU
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 1. Language Model Distillation
            with torch.cuda.amp.autocast(enabled=args.fp16):
                lm_loss = distill_language_model_batch(teacher_model, student_lm, batch, tokenizer, args)
            
            optimizer_lm.zero_grad()
            scaler.scale(lm_loss).backward()
            scaler.unscale_(optimizer_lm)
            torch.nn.utils.clip_grad_norm_(student_lm.parameters(), args.max_grad_norm)
            scaler.step(optimizer_lm)
            scheduler_lm.step()
            
            # 2. Data Generation
            with torch.no_grad():
                generated_data = generate_data_batch(student_lm, batch, tokenizer, args)
            
            # 3. World Model Training
            with torch.cuda.amp.autocast(enabled=args.fp16):
                wm_loss = train_world_model_batch(world_model, generated_data, teacher_model, tokenizer, args)
            
            optimizer_wm.zero_grad()
            scaler.scale(wm_loss).backward()
            scaler.unscale_(optimizer_wm)
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer_wm)
            scheduler_wm.step()
            
            scaler.update()
            
            # Log losses
            logger.info(f"LM Loss: {lm_loss.item():.4f}, WM Loss: {wm_loss.item():.4f}")
        
        # Save models after each epoch
        save_checkpoint(student_lm, f"{args.save_path}/student_lm_epoch_{epoch}.pt")
        save_checkpoint(world_model, f"{args.save_path}/world_model_epoch_{epoch}.pt")


def distill_language_model_batch(teacher, student, batch, tokenizer, args):
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(teacher.device)
    with torch.no_grad():
        teacher_outputs = teacher(**inputs)
    student_outputs = student(**inputs)
    
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    loss = loss_fn(
        F.log_softmax(student_outputs.logits / args.temperature, dim=-1),
        F.softmax(teacher_outputs.logits / args.temperature, dim=-1)
    ) * (args.temperature ** 2)
    return loss

def generate_data_batch(model, batch, tokenizer, args):
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(model.device)
    
    generated_outputs = model.generate(
        inputs.input_ids,
        max_length=args.max_length,
        num_return_sequences=1,
        do_sample=True,
        top_p=args.top_p,
        temperature=args.temperature_generate
    )
    
    generated_texts = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
    
    return [{'text': text} for text in generated_texts]

def train_world_model_batch(world_model, generated_data, judge_model, tokenizer, args):
    inputs = tokenizer([item['text'] for item in generated_data], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(world_model.device)
    
    # World model forward pass
    world_model_outputs = world_model(**inputs)
    
    # Judge model evaluation
    with torch.no_grad():
        judge_outputs = judge_model(**inputs)
    
    # Compute loss based on judge's evaluation
    loss = F.mse_loss(world_model_outputs.logits, judge_outputs.logits)
    
    return loss

def initialize_student_lm(args, bnb_config):
    if args.quantize:
        student_lm = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation='eager',  # Use eager attention implementation
            use_cache=False  # Disable caching for compatibility with gradient checkpointing
        )
    else:
        # Try to load a smaller pre-trained model
        student_model_name = args.student_model if args.student_model else 'distilgpt2'
        try:
            student_lm = AutoModelForCausalLM.from_pretrained(student_model_name)
        except Exception as e:
            print(f"Error loading student model {student_model_name}: {e}")
            student_lm = AutoModelForCausalLM.from_pretrained('distilgpt2')
        student_lm.config.use_cache = False  # Disable caching

    student_lm.gradient_checkpointing_enable()

    if args.use_lora:
        student_lm = prepare_model_for_kbit_training(student_lm)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        student_lm = get_peft_model(student_lm, lora_config)

    student_lm = student_lm.to(args.device)
    return student_lm

def initialize_world_model(args, bnb_config):
    if args.quantize:
        world_model = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation='eager',
            use_cache=False
        )
    else:
        # Try to load a smaller pre-trained model
        world_model_name = args.student_model if args.student_model else 'distilgpt2'
        try:
            world_model = AutoModelForCausalLM.from_pretrained(world_model_name)
        except Exception as e:
            print(f"Error loading world model {world_model_name}: {e}")
            world_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        world_model.config.use_cache = False  # Disable caching

    world_model.gradient_checkpointing_enable()

    if args.use_lora:
        world_model = prepare_model_for_kbit_training(world_model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        world_model = get_peft_model(world_model, lora_config)

    world_model = world_model.to(args.device)
    return world_model


def save_checkpoint(model, path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    logger.info(f"Checkpoint saved to {path}")

def generate_data(args):
    logger.info("Starting data generation process...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
    
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    
    generated_data = []
    for _ in tqdm(range(args.num_generate_samples), desc="Generating samples"):
        # Randomly select a prompt from the dataset
        prompt = random.choice(dataset['train'])['text']
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=args.max_length).to(device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=args.max_length,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature_generate
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_data.append({'text': generated_text})
    
    # Save generated data
    output_file = f"{args.save_path}/generated_data.jsonl"
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(generated_data)
    
    logger.info(f"Generated data saved to {output_file}")

def distill_model(args):
    logger.info("Starting model distillation process...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
    teacher_model.eval()
    
    # Initialize student model
    student_model = initialize_student_lm(args, None)  # No quantization for distillation
    
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    
    # Training loop
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            loss = distill_language_model_batch(teacher_model, student_model, batch, tokenizer, args)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(student_model, f"{args.save_path}/distilled_model_epoch_{epoch}.pt")
    
    logger.info("Model distillation completed.")

if __name__ == "__main__":
    main()