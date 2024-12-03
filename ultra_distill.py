# ultra_distill.py

import os
import json
import math
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

import jsonlines
from tqdm import tqdm
import logging

# Import custom classes from lightbulb_custom.py
from lightbulb.lightbulb_custom import (
    Transformer,
    TransformerBlock,
    MultiHeadAttention,
    RotaryPositionalEncoding,
    MoE,
    InfoNCE_Loss,
    CovarianceRegularization,
    DynamicsPerformanceLoss,
    ThoughtConsistencyLoss,
    PolicyValueJointLoss,
    ActionDiversityReward,
    ExpectedThoughtValueLoss,
    ExplorationRegularization,
    KL_DivergenceLoss,
    ActionEncoder,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    ThoughtNode,
    MCTS,
    State,
    PPOAgent
)

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Utility Functions
# -----------------------------

def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load a JSON Lines (.jsonl) file.

    Args:
        filepath (str): Path to the .jsonl file.

    Returns:
        List[Dict]: List of JSON objects.
    """
    data = []
    try:
        with jsonlines.open(filepath, 'r') as reader:
            for obj in reader:
                data.append(obj)
        logger.info(f"Loaded {len(data)} samples from {filepath}.")
    except Exception as e:
        logger.error(f"Error loading JSONL file at {filepath}: {e}")
        raise e
    return data

# -----------------------------
# Dataset Definitions
# -----------------------------

class DataSeedDataset(Dataset):
    """
    Dataset for Form 1: Next Token Prediction using data seeds.
    Each sample contains a chunk of data (data_seed).
    """
    def __init__(self, data_seeds: List[str], tokenizer, max_length: int = 512):
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
        input_ids = encoding['input_ids'].squeeze()  # Shape: (max_length)
        attention_mask = encoding['attention_mask'].squeeze()  # Shape: (max_length)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # Ignoring padding tokens in loss

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class RLHFDataset(Dataset):
    """
    Dataset for Form 2: PPO-based Option Selection using RLHF data.
    Each sample contains a prompt and options.
    """
    def __init__(self, rlhf_data: List[Dict], tokenizer, max_length: int = 512):
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

        # Combine options into a single text
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

        # Label indicating the correct option (e.g., 0 for A), 1 for B), etc.)
        # Assuming options are labeled as A), B), C), etc.
        option_label = correct_option.split(')')[0].strip().upper()
        label_map = {chr(65 + i): i for i in range(26)}  # A:0, B:1, ..., Z:25
        label = label_map.get(option_label, 0)  # Default to 0 if label not found

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'correct_option': correct_option,
            'incorrect_option': incorrect_option
        }

class InstructionDataset(Dataset):
    """
    Dataset for Form 3: Instruction-Output Alignment using instruction data.
    Each sample contains an instruction and its corresponding answer.
    """
    def __init__(self, instruction_data: List[Dict], tokenizer, max_length: int = 512):
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

        input_ids = encoding['input_ids']  # Shape: (1, max_length)
        attention_mask = encoding['attention_mask']  # Shape: (1, max_length)
        labels = target_encoding['input_ids'].clone()  # Clone to avoid modifying original data
        labels[target_encoding['input_ids'] == self.tokenizer.pad_token_id] = -100  # Shape: (1, max_length)

        return {
            'input_ids': input_ids.squeeze(0),         # Shape: (max_length,)
            'attention_mask': attention_mask.squeeze(0),  # Shape: (max_length,)
            'labels': labels.squeeze(0),              # Shape: (max_length,)
            'instruction': instruction,
            'answer': answer
        }

def load_latest_checkpoint(output_dir: str):
    """
    Load the most recent checkpoint from the output directory.

    Args:
        output_dir (str): Path to the directory where checkpoints are saved.

    Returns:
        dict or None: Loaded checkpoint dictionary or None if no checkpoint found.
    """
    if not os.path.exists(output_dir):
        return None

    checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        return None

    # Sort checkpoints by epoch number and select the latest
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    latest_checkpoint = os.path.join(output_dir, checkpoint_files[0])
    logger.info(f"Latest checkpoint found: {latest_checkpoint}")
    return torch.load(latest_checkpoint)

# -----------------------------
# Distillation Pipeline
# -----------------------------

def main():
    # -----------------------------
    # Argument Parsing
    # -----------------------------
    parser = argparse.ArgumentParser(description="Custom Knowledge Distillation Pipeline")
    parser.add_argument('--teacher_model', type=str, required=True, help='Path or name of the teacher model')
    parser.add_argument('--student_model', type=str, required=True, help='Path or name of the student model to be trained')
    parser.add_argument('--rlhf_data_path', type=str, required=True, help='Path to the RLHF JSONL file')
    parser.add_argument('--instruction_data_path', type=str, required=True, help='Path to the Instruction JSONL file')
    # Removed --data_seeds_path argument since it's not generated
    parser.add_argument('--output_dir', type=str, default='./distilled_model', help='Directory to save the distilled model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--clip_epsilon', type=float, default=1.0, help='Gradient clipping epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient for PPO')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient for PPO')
    parser.add_argument('--cosine_loss_weight', type=float, default=1.0, help='Weight for cosine similarity loss')
    parser.add_argument('--ppo_loss_weight', type=float, default=1.0, help='Weight for PPO loss')
    parser.add_argument('--next_token_loss_weight', type=float, default=1.0, help='Weight for next token prediction loss')
    parser.add_argument('--save_steps', type=int, default=1000, help='Number of steps between model saves')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in Transformer')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer layers')
    parser.add_argument('--d_model', type=int, default=512, help='Dimensionality of Transformer model')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimensionality of feedforward network in Transformer')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in MoE')
    parser.add_argument('--top_k', type=int, default=2, help='Top k experts in MoE')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for GAE')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda for GAE')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

    # Add the new checkpoint_path argument
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint file to load')

    args = parser.parse_args()


    # -----------------------------
    # Device Configuration
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # -----------------------------
    # Load Tokenizer and Models
    # -----------------------------
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        start_token_id = (
            tokenizer.cls_token_id
            if tokenizer.cls_token_id is not None
            else tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else 0  # Fallback token ID
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise e

    logger.info("Initializing student Transformer model...")
    vocab_size = tokenizer.vocab_size
    try:
        student_transformer = Transformer(
            input_dim=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            num_experts=args.num_experts,
            output_dim=vocab_size,
            dropout=0.1,
            top_k=args.top_k
        ).to(device)
        logger.info("Student Transformer model initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Transformer: {e}")
        raise e

    # -----------------------------
    # Load Data
    # -----------------------------
    logger.info("Loading RLHF data...")
    rlhf_data = load_jsonl(args.rlhf_data_path)

    logger.info("Loading Instruction data...")
    instruction_data = load_jsonl(args.instruction_data_path)

    # -----------------------------
    # Extract Data Seeds from RLHF Data
    # -----------------------------
    logger.info("Extracting data seeds from RLHF data...")
    try:
        data_seeds = list(set([entry['data_seed'] for entry in rlhf_data if 'data_seed' in entry]))
        logger.info(f"Extracted {len(data_seeds)} unique data seeds from RLHF data.")
    except Exception as e:
        logger.error(f"Error extracting data seeds from RLHF data: {e}")
        raise e

    # -----------------------------
    # Create Datasets and Dataloaders
    # -----------------------------
    logger.info("Creating Datasets and DataLoaders...")
    data_seed_dataset = DataSeedDataset(data_seeds, tokenizer, max_length=args.max_length)
    rlhf_dataset = RLHFDataset(rlhf_data, tokenizer, max_length=args.max_length)
    instruction_dataset = InstructionDataset(instruction_data, tokenizer, max_length=args.max_length)

    # Create validation splits (80-20 split)
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

    logger.info("Datasets and DataLoaders are ready.")

    # -----------------------------
    # Initialize Distillation Components
    # -----------------------------
    logger.info("Initializing distillation components...")

    # -----------------------------
    # Form 2: PPO-based Option Selection
    # -----------------------------
    logger.info("Setting up PPO Agent for Form 2...")

    # Define action_to_index mapping based on your specific actions
    # Ensuring uniqueness and alignment with RLHF data
    action_to_index = {}
    for entry in rlhf_data:
        correct_option = entry.get('correct_option', '').strip()
        incorrect_option = entry.get('incorrect_option', '').strip()

        # Extract option labels (e.g., 'A', 'B', etc.)
        for option in [correct_option, incorrect_option]:
            if not option:
                continue
            option_label = option.split(')')[0].strip().upper()
            if option_label and option_label not in action_to_index:
                action_to_index[option_label] = len(action_to_index)

    # Validate mapping with the provided example
    sample_entry = {
        "correct_option": "C) ",
        "incorrect_option": "A) "
    }
    sample_action_to_index = {}
    for key in [sample_entry['correct_option'].split(')')[0].strip().upper(),
                sample_entry['incorrect_option'].split(')')[0].strip().upper()]:
        if key not in sample_action_to_index:
            sample_action_to_index[key] = len(sample_action_to_index)
    assert action_to_index.get('C') == sample_action_to_index.get('C'), "Action to index mapping for 'C' is incorrect."
    assert action_to_index.get('A') == sample_action_to_index.get('A'), "Action to index mapping for 'A' is incorrect."
    logger.info(f"Action to Index Mapping: {action_to_index}")

    # Initialize index_to_action mapping
    index_to_action = {v: k for k, v in action_to_index.items()}

    # Initialize Action Encoder
    action_vocab_size = len(action_to_index)
    embed_dim = 128  # Define as needed
    action_encoder = ActionEncoder(action_vocab_size=action_vocab_size, embed_dim=embed_dim).to(device)
    logger.info("Action Encoder initialized.")

    # Initialize Representation Network
    vocab_dim = args.d_model  # Assuming vocab_dim equals d_model; adjust if different
    state_dim = args.d_model  # Define state_dim as needed
    representation_network = RepresentationNetwork(
        vocab_dim=vocab_size,        # Correctly set to vocab_size
        d_model=args.d_model,
        state_dim=args.d_model
    ).to(device)    
    logger.info("Representation Network initialized.")

    # Initialize Dynamics Network
    action_dim = embed_dim
    hidden_dim = 2048  # Define as needed
    dynamics_network = DynamicsNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    logger.info("Dynamics Network initialized.")

    # Initialize Prediction Network
    value_dim = 1  # Define as needed
    prediction_network = PredictionNetwork(state_dim=state_dim, action_vocab_size=action_vocab_size, value_dim=value_dim).to(device)
    logger.info("Prediction Network initialized.")

    # Initialize PPO Agent
    ppo_agent = PPOAgent(
        policy_network=prediction_network,
        optimizer=torch.optim.AdamW(prediction_network.parameters(), lr=args.learning_rate),
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef
    )
    logger.info("PPO Agent initialized.")

    # Initialize Scheduler for PPO Agent
    total_steps = (len(train_loader_form1) + len(train_loader_form2) + len(train_loader_form3)) * args.epochs
    scheduler_ppo = get_linear_schedule_with_warmup(
        ppo_agent.optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    ppo_agent.set_scheduler(scheduler_ppo)
    logger.info("Scheduler for PPO Agent initialized.")

    # Form 3: Instruction-Output Alignment - Cosine Similarity Loss
    cosine_similarity_loss_fn = ThoughtConsistencyLoss().to(device)
    logger.info("Cosine Similarity Loss function initialized.")

    # Initialize separate optimizer for student_transformer
    optimizer_student = torch.optim.AdamW(student_transformer.parameters(), lr=args.learning_rate)
    logger.info("Optimizer for Student Transformer initialized.")

    # Initialize Scheduler for student_transformer
    scheduler_student = get_linear_schedule_with_warmup(
        optimizer_student,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    logger.info("Scheduler for Student Transformer initialized.")

    # Initialize MCTS
    logger.info("Initializing MCTS...")
    mcts = MCTS(
        prediction_network=prediction_network,
        dynamics_network=dynamics_network,
        action_encoder=action_encoder,
        action_to_index=action_to_index,
        num_iterations=10,
        exploration_constant=math.sqrt(2),
        beam_size=5,
        n_tokens_predict=1  # Assuming single action selection; adjust if needed
    )
    logger.info("MCTS initialized.")

    # -----------------------------
    # Initialize Objective Functions
    # -----------------------------
    logger.info("Distillation components initialized.")

    # -----------------------------
    # Load Checkpoint if Available
    # -----------------------------
    logger.info("Checking for latest checkpoint...")

    # Modify this section to prioritize loading from --checkpoint_path if provided
    checkpoint = None
    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
        else:
            logger.error(f"Checkpoint path {args.checkpoint_path} does not exist.")
            raise FileNotFoundError(f"Checkpoint path {args.checkpoint_path} does not exist.")
    else:
        checkpoint = load_latest_checkpoint(args.output_dir)

    if checkpoint:
        logger.info(f"Loading checkpoint from {args.output_dir if not args.checkpoint_path else args.checkpoint_path}")
        try:
            student_transformer.load_state_dict(checkpoint['student_transformer_state_dict'])
            prediction_network.load_state_dict(checkpoint['prediction_network_state_dict'])
            action_encoder.load_state_dict(checkpoint['action_encoder_state_dict'])
            representation_network.load_state_dict(checkpoint['representation_network_state_dict'])
            dynamics_network.load_state_dict(checkpoint['dynamics_network_state_dict'])
            ppo_agent.__dict__.update(checkpoint['ppo_agent_state_dict'])

            optimizer_student.load_state_dict(checkpoint['optimizer_student_state_dict'])
            scheduler_student.load_state_dict(checkpoint['scheduler_student_state_dict'])
            scheduler_ppo.load_state_dict(checkpoint['scheduler_ppo_state_dict'])
            logger.info("Checkpoint loaded successfully.")
        except KeyError as ke:
            logger.error(f"Key error while loading checkpoint: {ke}")
            logger.error("Ensure that the checkpoint contains all necessary keys.")
            raise ke
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise e
    else:
        logger.info("No checkpoint found. Initializing models from scratch.")


    # -----------------------------
    # Training Loop
    # -----------------------------
    logger.info("Starting training...")

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # -----------------------------
        # Form 1 Training: Next Token Prediction
        # -----------------------------
        logger.info("\n--- Training Form 1: Next Token Prediction ---")
        student_transformer.train()
        for batch in tqdm(train_loader_form1, desc="Form 1 Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Shift input_ids and labels for next token prediction
            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            src_mask = attention_mask[:, :-1]
            tgt_mask = attention_mask[:, 1:]
            labels = labels[:, 1:]

            # Prepare masks
            src_mask = (src != tokenizer.pad_token_id).long()
            tgt_mask = (tgt != tokenizer.pad_token_id).long()



            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = student_transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                logits = outputs  # Outputs are logits

                # Compute loss
                logits_flat = logits.reshape(-1, vocab_size)
                labels_flat = labels.reshape(-1)
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits_flat, labels_flat) * args.next_token_loss_weight

            optimizer_student.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_student)
                nn.utils.clip_grad_norm_(student_transformer.parameters(), args.clip_epsilon)
                scaler.step(optimizer_student)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(student_transformer.parameters(), args.clip_epsilon)
                optimizer_student.step()
            scheduler_student.step()


        # -----------------------------
        # Form 2 Training: PPO-based Option Selection with MCTS
        # -----------------------------
        logger.info("\n--- Training Form 2: PPO-based Option Selection with MCTS ---")
        prediction_network.train()
        for batch in tqdm(train_loader_form2, desc="Form 2 Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            correct_option = batch['correct_option']  # List of correct options
            incorrect_option = batch['incorrect_option']  # List of incorrect options

            # Prepare lists to collect PPO training data
            states = []
            actions = []
            old_log_probs = []
            advantages = []
            returns = []

            # Collect data for batch processing
            for i in range(len(input_ids)):
                try:
                    # Get the correct label index
                    correct_label = labels[i].item()

                    # Get correct and incorrect option labels
                    correct_label_char = list(action_to_index.keys())[list(action_to_index.values()).index(correct_label)]
                    incorrect_label_char = None
                    for key, value in action_to_index.items():
                        if value != correct_label and key != correct_label_char:
                            incorrect_label_char = key
                            break
                    if incorrect_label_char is None:
                        incorrect_label_char = [k for k in action_to_index.keys() if k != correct_label_char][0]

                    # Create child ThoughtNodes for options
                    child1 = ThoughtNode(name=correct_label_char)
                    child2 = ThoughtNode(name=incorrect_label_char)
                    thought_node = ThoughtNode(name="root")
                    thought_node.add_child(child1)
                    thought_node.add_child(child2)

                    # Initialize State
                    with torch.no_grad():
                        # Ensure student_transformer outputs a tensor aligned with the embedding dimension
                        src = input_ids[i].unsqueeze(0)
                        tgt = attention_mask[i].unsqueeze(0)
                        initial_output = student_transformer(src=src, tgt=tgt)  # Shape should align with embedding_dim
                        initial_representation = representation_network(initial_output)
                        state = State(
                            representation=initial_representation.to(device),
                            dynamics_network=dynamics_network,
                            action_encoder=action_encoder,
                            action_to_index=action_to_index,
                            thought_node=thought_node
                        )

                    # Run MCTS to select the best action
                    best_action_sequence = mcts.search_with_beam(state)
                    if best_action_sequence:
                        selected_action = best_action_sequence[0]
                    else:
                        selected_action = correct_label_char  # Fallback to correct action

                    # Map selected_action to label
                    selected_label = action_to_index.get(selected_action, correct_label)

                    # Compute log_probs for selected action
                    with torch.no_grad():
                        output = prediction_network(state.representation[:, -1, :])
                        policy_logits, _ = output
                        log_probs_all = F.log_softmax(policy_logits, dim=-1)
                        selected_log_prob = log_probs_all.gather(1, torch.tensor([[selected_label]], device=device)).squeeze(1)

                    # Placeholder for advantages and returns
                    advantage = 1.0  # Replace with actual advantage
                    return_ = 1.0     # Replace with actual return

                    # Append to lists
                    states.append(state.representation[:, -1, :]) 
                    actions.append(torch.tensor([selected_label], device=device))
                    old_log_probs.append(selected_log_prob.detach())
                    advantages.append(torch.tensor([advantage], device=device))
                    returns.append(torch.tensor([return_], device=device))

                except Exception as e:
                    logger.error(f"Error processing batch index {i}: {e}")
                    continue

            if not states:
                logger.warning("No valid training samples found in this batch.")
                continue

            # Convert lists to tensors
            states_tensor = torch.cat(states, dim=0)
            actions_tensor = torch.cat(actions, dim=0)
            old_log_probs_tensor = torch.cat(old_log_probs, dim=0)
            advantages_tensor = torch.cat(advantages, dim=0)
            returns_tensor = torch.cat(returns, dim=0)

            # Compute PPO loss
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                ppo_loss = ppo_agent.compute_loss(
                    states=states_tensor,
                    old_log_probs=old_log_probs_tensor,
                    actions=actions_tensor,
                    returns=returns_tensor,
                    advantages=advantages_tensor
                )

            # Backpropagation
            ppo_agent.optimizer.zero_grad()
            if scaler:
                scaler.scale(ppo_loss).backward()
                scaler.unscale_(ppo_agent.optimizer)
                nn.utils.clip_grad_norm_(prediction_network.parameters(), args.clip_epsilon)
                scaler.step(ppo_agent.optimizer)
                scaler.update()
            else:
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(prediction_network.parameters(), args.clip_epsilon)
                ppo_agent.optimizer.step()
            scheduler_ppo.step()

        # -----------------------------
        # Form 3 Training: Instruction-Output Alignment
        # -----------------------------
        logger.info("\n--- Training Form 3: Instruction-Output Alignment ---")
        student_transformer.train()
        for batch in tqdm(train_loader_form3, desc="Form 3 Training"):
            src = batch['input_ids'].to(device)
            src_mask = batch['attention_mask'].to(device)
            tgt = batch['labels'].to(device)
            tgt_mask = (tgt != -100).long()  # Ensure this mask aligns with tgt dimensions

            # Ensure tgt_input is correctly aligned with the pad token and vocab size
            tgt_input = tgt.clone()
            tgt_input[tgt_input == -100] = tokenizer.pad_token_id

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = student_transformer(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                logits = outputs

                # Flatten logits to align with labels and apply cross-entropy loss
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                logits_flat = logits.view(-1, logits.size(-1))  # Flatten the logits correctly
                labels_flat = tgt.view(-1)
                loss = loss_fn(logits_flat, labels_flat) * args.cosine_loss_weight

            optimizer_student.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_student)
                nn.utils.clip_grad_norm_(student_transformer.parameters(), args.clip_epsilon)
                scaler.step(optimizer_student)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(student_transformer.parameters(), args.clip_epsilon)
                optimizer_student.step()
            scheduler_student.step()


        # -----------------------------
        # Validation
        # -----------------------------
        logger.info("\n--- Validation ---")
        student_transformer.eval()
        prediction_network.eval()

        # Form 1 Validation
        logger.info("\n--- Validating Form 1: Next Token Prediction ---")
        form1_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader_form1, desc="Form 1 Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Shift input_ids and labels for next token prediction
                src = input_ids[:, :-1]
                tgt = input_ids[:, 1:]
                src_mask = attention_mask[:, :-1]
                tgt_mask = attention_mask[:, 1:]
                labels = labels[:, 1:]

                # Prepare masks
                src_mask = (src != tokenizer.pad_token_id).long()
                tgt_mask = (tgt != tokenizer.pad_token_id).long()

                # Forward pass
                outputs = student_transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                logits = outputs  # Outputs are logits

                # Compute loss
                logits_flat = logits.reshape(-1, vocab_size)
                labels_flat = labels.reshape(-1)
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits_flat, labels_flat)
                form1_val_loss += loss.item()

        form1_val_loss /= len(val_loader_form1)
        logger.info(f"Form 1 Validation Loss: {form1_val_loss:.4f}")


        # -----------------------------
        # Form 2 Validation: PPO-based Option Selection with MCTS
        # -----------------------------
        logger.info("\n--- Validating Form 2: PPO-based Option Selection with MCTS ---")
        form2_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader_form2, desc="Form 2 Validation")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                correct_option = batch['correct_option']
                incorrect_option = batch['incorrect_option']

                # Prepare lists to collect PPO validation data
                states = []
                actions = []
                old_log_probs = []
                advantages = []
                returns = []

                # Collect data for batch processing
                for i in range(len(input_ids)):
                    try:
                        # Get the correct label index
                        correct_label = labels[i].item()

                        # Get correct and incorrect option labels
                        correct_label_char = list(action_to_index.keys())[list(action_to_index.values()).index(correct_label)]
                        incorrect_label_char = None
                        for key, value in action_to_index.items():
                            if value != correct_label and key != correct_label_char:
                                incorrect_label_char = key
                                break
                        if incorrect_label_char is None:
                            incorrect_label_char = [k for k in action_to_index.keys() if k != correct_label_char][0]

                        # Create child ThoughtNodes for options
                        child1 = ThoughtNode(name=correct_label_char)
                        child2 = ThoughtNode(name=incorrect_label_char)
                        thought_node = ThoughtNode(name="root", children=[child1, child2])

                        # Initialize State
                        with torch.no_grad():
                            # Prepare src and src_mask
                            src = input_ids[i].unsqueeze(0).to(device)  # Shape: (1, seq_len)
                            src_mask = (src != tokenizer.pad_token_id).long()  # Shape: (1, seq_len)

                            # Prepare tgt and tgt_mask
                            seq_len = src.shape[1]
                            tgt = input_ids[i].unsqueeze(0).to(device)  # Shape: (1, seq_len)
                            tgt = torch.cat([torch.tensor([[start_token_id]], device=device), tgt[:, :-1]], dim=1)  # Shifted right
                            tgt_mask = (tgt != tokenizer.pad_token_id).long()  # Shape: (1, seq_len)

                            # Pass through the model
                            initial_output = student_transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                            print(f"[Validation][Batch {batch_idx}][Sample {i}] student_transformer output shape: {initial_output.shape}")

                            # Pass through representation network
                            initial_representation = representation_network(initial_output)
                            print(f"[Validation][Batch {batch_idx}][Sample {i}] representation_network output shape: {initial_representation.shape}")

                            # Do not slice the last time step
                            # Use the full representation as in the training loop
                            state = State(
                                representation=initial_representation.to(device),
                                dynamics_network=dynamics_network,
                                action_encoder=action_encoder,
                                action_to_index=action_to_index,
                                thought_node=thought_node
                            )
                            print(f"[Validation][Batch {batch_idx}][Sample {i}] State Representation Shape: {state.representation.shape}")

                        # Run MCTS to select the best action
                        print(f"[Validation][Batch {batch_idx}][Sample {i}] Running MCTS...")
                        best_action_sequence = mcts.search_with_beam(state)
                        print(f"[Validation][Batch {batch_idx}][Sample {i}] Best Action Sequence: {best_action_sequence}")

                        # Assuming single action selection, take the first action
                        if best_action_sequence:
                            selected_action = best_action_sequence[0]
                        else:
                            selected_action = correct_label_char  # Fallback to correct action

                        # Map selected_action to label
                        selected_label = action_to_index.get(selected_action, correct_label)
                        print(f"[Validation][Batch {batch_idx}][Sample {i}] Selected Action: {selected_action}, Selected Label: {selected_label}")

                        # Compute log_probs for selected action
                        with torch.no_grad():
                            # Use the last time step of the representation
                            output = prediction_network(state.representation[:, -1, :])
                            policy_logits, _ = output
                            print(f"[Validation][Batch {batch_idx}][Sample {i}] policy_logits shape: {policy_logits.shape}")

                            log_probs_all = F.log_softmax(policy_logits, dim=-1)
                            selected_log_prob = log_probs_all.gather(1, torch.tensor([[selected_label]], device=device)).squeeze(1)
                            print(f"[Validation][Batch {batch_idx}][Sample {i}] selected_log_prob shape: {selected_log_prob.shape}")

                        # Placeholder for advantages and returns (To be replaced with actual calculations)
                        advantage = 1.0  # Replace with actual advantage
                        return_ = 1.0     # Replace with actual return

                        # Append to lists
                        states.append(state.representation[:, -1, :])  # Use last time step
                        actions.append(torch.tensor([selected_label], device=device))
                        old_log_probs.append(selected_log_prob.detach())
                        advantages.append(torch.tensor([advantage], device=device))
                        returns.append(torch.tensor([return_], device=device))

                    except Exception as e:
                        logger.error(f"Error processing validation batch index {i}: {e}")
                        continue

                if not states:
                    logger.warning("No valid validation samples found in this batch.")
                    continue

                # Convert lists to tensors
                states_tensor = torch.cat(states, dim=0)  # Shape: (batch_size, state_dim)
                actions_tensor = torch.cat(actions, dim=0)  # Shape: (batch_size,)
                old_log_probs_tensor = torch.cat(old_log_probs, dim=0)  # Shape: (batch_size,)
                advantages_tensor = torch.cat(advantages, dim=0)  # Shape: (batch_size,)
                returns_tensor = torch.cat(returns, dim=0)  # Shape: (batch_size,)

                # Debugging: Verify tensor shapes
                print(f"[Validation][Batch {batch_idx}] States Tensor Shape: {states_tensor.shape}, Dtype: {states_tensor.dtype}")
                print(f"[Validation][Batch {batch_idx}] Actions Tensor Shape: {actions_tensor.shape}, Dtype: {actions_tensor.dtype}")
                print(f"[Validation][Batch {batch_idx}] Old Log Probs Tensor Shape: {old_log_probs_tensor.shape}, Dtype: {old_log_probs_tensor.dtype}")
                print(f"[Validation][Batch {batch_idx}] Advantages Tensor Shape: {advantages_tensor.shape}, Dtype: {advantages_tensor.dtype}")
                print(f"[Validation][Batch {batch_idx}] Returns Tensor Shape: {returns_tensor.shape}, Dtype: {returns_tensor.dtype}")

                # Compute PPO loss
                try:
                    ppo_loss = ppo_agent.compute_loss(
                        states=states_tensor,
                        old_log_probs=old_log_probs_tensor,
                        actions=actions_tensor,
                        returns=returns_tensor,
                        advantages=advantages_tensor
                    )
                    print(f"[Validation][Batch {batch_idx}] Validation PPO Loss: {ppo_loss.item()}")
                    form2_val_loss += ppo_loss.item()
                except Exception as e:
                    logger.error(f"PPO Loss computation failed for batch {batch_idx}: {e}")
                    continue

        form2_val_loss /= len(val_loader_form2)
        logger.info(f"Form 2 Validation PPO Loss: {form2_val_loss:.4f}")



        # Form 3 Validation
        logger.info("\n--- Validating Form 3: Instruction-Output Alignment ---")
        form3_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader_form3, desc="Form 3 Validation"):
                src = batch['input_ids'].to(device)
                src_mask = batch['attention_mask'].to(device)
                tgt = batch['labels'].to(device)
                tgt_mask = (tgt != -100).long()  # Ensure this mask aligns with tgt dimensions

                # Prepare target input
                tgt_input = tgt.clone()
                tgt_input[tgt_input == -100] = tokenizer.pad_token_id

                # Forward pass
                outputs = student_transformer(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                logits = outputs  # Outputs are logits

                # Compute loss
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = tgt.view(-1)
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits_flat, labels_flat) * args.cosine_loss_weight
                form3_val_loss += loss.item()

        form3_val_loss /= len(val_loader_form3)
        logger.info(f"Form 3 Validation Loss: {form3_val_loss:.4f}")

        # -----------------------------
        # Checkpointing
        # -----------------------------
        logger.info("\n--- Checkpointing ---")
        try:
            save_path = os.path.join(args.output_dir, f'epoch_{epoch}.pt')
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'student_transformer_state_dict': student_transformer.state_dict(),
                'prediction_network_state_dict': prediction_network.state_dict(),
                'action_encoder_state_dict': action_encoder.state_dict(),
                'representation_network_state_dict': representation_network.state_dict(),
                'dynamics_network_state_dict': dynamics_network.state_dict(),
                'ppo_agent_state_dict': ppo_agent.__dict__,
                'optimizer_student_state_dict': optimizer_student.state_dict(),
                'scheduler_student_state_dict': scheduler_student.state_dict(),
                'scheduler_ppo_state_dict': scheduler_ppo.state_dict()
            }, save_path)
            logger.info(f"Model checkpoint saved at {save_path}")
        except Exception as e:
            logger.error(f"Error during checkpointing: {e}")

if __name__ == "__main__":
    main()
