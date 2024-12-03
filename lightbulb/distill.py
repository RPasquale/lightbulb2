import argparse
import math
import os
import sys
import json
import jsonlines
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, load_dataset_builder
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# Import Custom Components from lightbulb_custom
# ======================================
from lightbulb_custom import (
    RotaryPositionalEncoding,
    MultiHeadAttention,
    MoE,
    TransformerBlock,
    Transformer,
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
    PPOAgent  # Ensure PPOAgent is correctly defined and imported
)

# ==========================
# Custom Dataset Definition
# ==========================
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'labels': self.labels[idx]}

# ================================
# Utility Functions for Data Loading
# ================================
def load_npy_data(path, role):
    inputs, labels = [], []

    if os.path.isdir(path):
        # If path is a directory, list all .npy files within it
        file_names = [f for f in os.listdir(path) if f.endswith('.npy')]
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            data = np.load(file_path, allow_pickle=True)

            # Debugging output to understand data structure
            print(f"Loading file: {file_path}")
            print(f"Data type: {type(data)}")
            print(f"Data shape: {data.shape}")
            print(f"First element: {data[0]}")

            if role == 'next_token_prediction':
                inputs.extend(data)
                labels.extend(data)
            elif role == 'instruction_output':
                for text in data:
                    try:
                        instruction = text.split('Instruction:')[1].split('Output:')[0].strip()
                        output = text.split('Output:')[1].strip()
                        inputs.append(instruction)
                        labels.append(output)
                    except IndexError:
                        continue  # Skip if format is incorrect
            elif role == 'qa_synthesis':
                for entry in data:
                    question = entry.get('QAList', '')
                    context = entry.get('context', '')
                    answer = context  # Use the context as the answer for synthesis
                    inputs.append(f"Question: {question}")
                    labels.append(answer)
            elif role == 'ppo_training':
                inputs.extend(data['prompt'])
                labels.extend(data['response'])
    elif os.path.isfile(path) and path.endswith('.npy'):
        # If path is a single .npy file
        data = np.load(path, allow_pickle=True)

        # Debugging output to understand data structure
        print(f"Loading file: {path}")
        print(f"Data type: {type(data)}")
        print(f"Data shape: {data.shape}")
        print(f"First element: {data[0]}")

        if role == 'next_token_prediction':
            inputs.extend(data)
            labels.extend(data)
        elif role == 'instruction_output':
            for text in data:
                try:
                    instruction = text.split('Instruction:')[1].split('Output:')[0].strip()
                    output = text.split('Output:')[1].strip()
                    inputs.append(instruction)
                    labels.append(output)
                except IndexError:
                    continue  # Skip if format is incorrect
        elif role == 'qa_synthesis':
            for entry in data:
                question = entry.get('QAList', '')
                context = entry.get('context', '')
                answer = context  # Use the context as the answer for synthesis
                inputs.append(f"Question: {question}")
                labels.append(answer)
        elif role == 'ppo_training':
            inputs.extend(data['prompt'])
            labels.extend(data['response'])
    else:
        print(f"Invalid path provided: {path}. It must be a directory or a .npy file.")
        sys.exit(1)

    return inputs, labels

def prepare_data_from_npy(tokenizer, max_length, batch_size, npy_data_paths, npy_data_roles):
    if len(npy_data_paths) != len(npy_data_roles):
        print("The number of data paths must match the number of roles.")
        sys.exit(1)

    inputs, labels = [], []
    for path, role in zip(npy_data_paths, npy_data_roles):
        data_inputs, data_labels = load_npy_data(path, role)
        inputs.extend(data_inputs)
        labels.extend(data_labels)

    if all(role == 'next_token_prediction' for role in npy_data_roles):
        # Treat inputs and labels as token IDs (integers)
        inputs = np.array(inputs)
        labels = np.array(labels)

        # Define sequence length
        seq_length = max_length

        # Calculate number of sequences
        num_sequences = len(inputs) // (seq_length + 1)

        # Truncate to have full sequences
        inputs = inputs[:num_sequences * (seq_length + 1)]
        labels = labels[:num_sequences * (seq_length + 1)]

        # Reshape into sequences
        inputs = inputs.reshape(num_sequences, seq_length + 1)
        labels = labels.reshape(num_sequences, seq_length + 1)

        # For next_token_prediction, input is first seq_length tokens, label is next tokens
        inputs = inputs[:, :-1]
        labels = labels[:, 1:]  # Corrected: derive labels from the original labels array

        # Convert to tensors
        inputs = torch.tensor(inputs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        # Tokenize inputs and labels (for other roles)
        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokenized_labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        inputs = tokenized_inputs["input_ids"]
        labels = tokenized_labels["input_ids"]

    custom_dataset = CustomDataset(inputs, labels)

    train_size = int(0.9 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

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
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue

        query = item.get('query', '')
        content = item.get('content', '')
        if content == "RAG response generation failed.":
            content = ""

        combined_text = f"Query: {query} Content: {content}"

        # Create a dictionary with processed data
        processed_item = {
            'text': combined_text
        }

        processed_data.append(processed_item)

    return processed_data

def load_custom_data(args, tokenizer, custom_data):
    # Preprocess the custom data
    processed_data = preprocess_custom_data(custom_data)

    # Create a custom dataset
    class CustomDatasetProcessed(torch.utils.data.Dataset):
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
                'attention_mask': encoded['attention_mask'].squeeze()
            }

    # Create dataset and dataloader
    dataset = CustomDatasetProcessed(processed_data, tokenizer, args.max_length)

    # Split the dataset into train and eval
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

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

def prepare_data(tokenizer, dataset, max_length, batch_size):
    # Tokenize the inputs and labels
    tokenized_inputs = tokenizer(dataset["train"]["text"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tokenized_labels = tokenizer(dataset["train"]["text"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Create custom dataset
    custom_dataset = CustomDataset(tokenized_inputs["input_ids"], tokenized_labels["input_ids"])

    # Split into training and validation sets
    train_size = int(0.9 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

# ==========================
# Training and Validation Functions
# ==========================

def save_all_models(transformer_model, representation_network, dynamics_network, prediction_network, action_encoder, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(transformer_model.state_dict(), os.path.join(save_dir, f'transformer_model_epoch_{epoch}.pt'))
    torch.save(representation_network.state_dict(), os.path.join(save_dir, f'representation_network_epoch_{epoch}.pt'))
    torch.save(dynamics_network.state_dict(), os.path.join(save_dir, f'dynamics_network_epoch_{epoch}.pt'))
    torch.save(prediction_network.state_dict(), os.path.join(save_dir, f'prediction_network_epoch_{epoch}.pt'))
    torch.save(action_encoder.state_dict(), os.path.join(save_dir, f'action_encoder_epoch_{epoch}.pt'))

    print(f"All models saved for epoch {epoch}.")

def train_epoch_world_model(
    world_model_components,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    accumulation_steps,
    max_grad_norm,
    model_transformer,
    state_dim,
    embed_dim,
    input_dim
):
    representation_network, dynamics_network, prediction_network, action_encoder, ppo_agent, model_transformer = world_model_components
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

        with torch.cuda.amp.autocast():
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

            info_nce = InfoNCE_Loss()(state_representation.reshape(-1, state_dim),
                                      F.dropout(state_representation.reshape(-1, state_dim), p=0.1, training=True))

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
            loss = loss / accumulation_steps

        print("Backward pass...")
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            print("Gradient clipping...")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [param for group in optimizer.param_groups for param in group['params']],
                max_grad_norm
            )

            print("Optimizer step...")
            scaler.step(optimizer)
            scaler.update()

            print("Zeroing gradients...")
            optimizer.zero_grad()

            print("Updating learning rate...")
            scheduler.step()

        total_loss += loss.item() * accumulation_steps

    avg_loss = total_loss / len(train_loader)
    print(f"World Model training epoch completed. Average loss: {avg_loss:.4f}")
    return avg_loss

def train_step(teacher, student, data_loader, optimizer, criterion, scaler, temperature=2.0):
    teacher.eval()
    student.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            with torch.no_grad():
                teacher_outputs = teacher(inputs).logits
                teacher_logits = teacher_outputs / temperature

            student_outputs = student(inputs).logits
            student_logits = student_outputs / temperature

            # Compute KL Divergence Loss
            loss = criterion(nn.functional.log_softmax(student_logits, dim=-1), nn.functional.softmax(teacher_logits, dim=-1))
            loss = loss * (temperature ** 2)  # Scale loss by temperature squared

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def validate(teacher, student, data_loader, criterion, temperature=2.0):
    teacher.eval()
    student.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            teacher_outputs = teacher(inputs).logits
            teacher_logits = teacher_outputs / temperature

            student_outputs = student(inputs).logits
            student_logits = student_outputs / temperature

            loss = criterion(nn.functional.log_softmax(student_logits, dim=-1), nn.functional.softmax(teacher_logits, dim=-1))
            loss = loss * (temperature ** 2)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def save_checkpoint(state, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# ==========================
# Main Training Function
# ==========================
def distill_model(
    teacher_model_name: str,
    student_model_name: str,
    tokenizer,
    train_loader,
    val_loader,
    distill_full_model: bool = True,
    num_epochs: int = 3,
    batch_size: int = 4,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    temperature: float = 2.0,
    save_path: str = "./distilled_model",
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
    early_stopping_patience: int = 3,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01
):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Load teacher model
    print("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
    print("Teacher model loaded successfully.")

    if distill_full_model:
        # Full World Model Distillation
        print(f"Starting Full World Model Distillation into '{student_model_name}'.")

        # Load or instantiate student model
        print(f"Attempting to load student model '{student_model_name}'...")
        try:
            student = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
            print(f"Student model '{student_model_name}' loaded successfully.")
        except (OSError, ValueError):
            print(f"Student model '{student_model_name}' not found. Instantiating a new student model.")
            try:
                student = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
                student.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"New student model '{student_model_name}' instantiated and saved to '{save_path}'.")
            except Exception as inst_e:
                print(f"Failed to instantiate and save student model: {inst_e}")
                sys.exit(1)

        # Freeze teacher model parameters
        for param in teacher.parameters():
            param.requires_grad = False

        # Extract required dimensions
        d_model = teacher.config.hidden_size  # Verify this value
        print(f"d_model: {d_model}")  # Debugging statement
        vocab_dim = tokenizer.vocab_size
        state_dim = 512      # Adjust as needed
        embed_dim = 256      # Adjust as needed
        hidden_dim = 1024    # Adjust as needed
        value_dim = 1        # Typically 1 for value estimates

        # Define additional hyperparameters for Transformer and MoE
        num_heads = 8                    # Must divide d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        num_layers = 6                   # Adjust as needed
        d_ff = 4 * d_model               # Typically 4 * d_model
        num_experts = 4                  # Adjust as needed
        output_dim = vocab_dim           # Map to vocabulary size

        action_vocab_size = vocab_dim    # Assuming actions correspond to vocabulary tokens
        action_dim = embed_dim           # Dimension of action embeddings

        print(f"Using output_dim: {output_dim}")  # Should be vocab_dim


        print("Initializing custom world model components...")
        # Initialize custom world model components with required arguments
        representation_network = RepresentationNetwork(vocab_dim, d_model, state_dim).to(device)
        action_encoder = ActionEncoder(action_vocab_size, embed_dim).to(device)
        dynamics_network = DynamicsNetwork(state_dim, action_dim, hidden_dim).to(device)
        prediction_network = PredictionNetwork(state_dim, action_vocab_size, value_dim).to(device)
        transformer = Transformer(
            input_dim=vocab_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            num_experts=num_experts,
            output_dim=output_dim,  # Now set to d_model
            dropout=0.1,
            top_k=2
        ).to(device)

        # Define action_to_index mapping (assuming action names are tokens)
        # If actions are different, adjust accordingly
        action_to_index = {token: idx for idx, token in enumerate(tokenizer.get_vocab())}

        # Define optimizer for all components
        optimizer = optim.AdamW(
            list(student.parameters()) +
            list(representation_network.parameters()) +
            list(action_encoder.parameters()) +
            list(dynamics_network.parameters()) +
            list(prediction_network.parameters()) +
            list(transformer.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Initialize PPOAgent without .to(device)
        ppo_agent = PPOAgent(
            policy_network=prediction_network,
            optimizer=optimizer,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5
        )
        # PPOAgent's internal components are already on the device via their initialization

        # Bundle world model components
        world_model_components = (
            representation_network,
            dynamics_network,
            prediction_network,
            action_encoder,
            ppo_agent,
            transformer
        )

        # Define scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Initialize GradScaler for mixed precision
        scaler = GradScaler()

        # Define loss criterion (if needed for other components)
        criterion = nn.KLDivLoss(reduction="batchmean")

        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 20)

            # Training
            train_loss = train_epoch_world_model(
                world_model_components=world_model_components,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                model_transformer=transformer,
                state_dim=state_dim,
                embed_dim=embed_dim,
                input_dim=vocab_dim
            )
            print(f"Training Loss: {train_loss:.4f}")
            writer.add_scalar("Loss/Train", train_loss, epoch)

            # Validation
            val_loss = validate(teacher, student, val_loader, criterion, temperature)
            print(f"Validation Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/Validation", val_loss, epoch)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_dir, epoch)
                student.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Best model saved at epoch {epoch}")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

            # Step the scheduler
            scheduler.step()

        writer.close()
        print("\nFull World Model Distillation completed.")
    else:
        # Standard Language Model Distillation
        print(f"Starting Standard Language Model Distillation into '{student_model_name}'.")

        # Load or instantiate student model
        print(f"Attempting to load student model '{student_model_name}'...")
        try:
            student = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
            print(f"Student model '{student_model_name}' loaded successfully.")
        except (OSError, ValueError):
            print(f"Student model '{student_model_name}' not found. Instantiating a new student model.")
            try:
                student = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
                student.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"New student model '{student_model_name}' instantiated and saved to '{save_path}'.")
            except Exception as inst_e:
                print(f"Failed to instantiate and save student model: {inst_e}")
                sys.exit(1)

        # Freeze teacher model parameters
        for param in teacher.parameters():
            param.requires_grad = False

        # Define optimizer, scheduler, and scaler for mixed precision
        optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler()

        # Define loss criterion
        criterion = nn.KLDivLoss(reduction="batchmean")

        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 20)

            # Training
            train_loss = train_step(teacher, student, train_loader, optimizer, criterion, scaler, temperature)
            print(f"Training Loss: {train_loss:.4f}")
            writer.add_scalar("Loss/Train", train_loss, epoch)

            # Validation
            val_loss = validate(teacher, student, val_loader, criterion, temperature)
            print(f"Validation Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/Validation", val_loss, epoch)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }, checkpoint_dir, epoch)
                student.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Best model saved at epoch {epoch}")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

            # Step the scheduler
            scheduler.step()

        writer.close()
        print("\nStandard Language Model Distillation completed.")

# ==========================
# Argument Parsing
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Distill a large LLM into a smaller one or a full language world model.")

    # Required arguments
    parser.add_argument("--teacher_model_name", type=str, required=True, help="Name of the teacher model")
    parser.add_argument("--student_model_name", type=str, required=True, help="Name of the student model")

    # Data loading arguments
    parser.add_argument("--use_npy_data", action="store_true", default=False, help="Use .npy data files for training")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset (if not using .npy data)")
    parser.add_argument("--config", type=str, default=None, help="Dataset configuration (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--custom_data_paths", type=str, nargs="+", help="Paths to custom data files for standard language model distillation")
    parser.add_argument("--npy_data_paths", type=str, nargs="+", help="Paths to .npy data files")
    parser.add_argument("--npy_data_roles", type=str, nargs="+", help="Roles corresponding to each .npy data path")

    # Mode selection
    parser.add_argument("--distill_full_model", action="store_true", help="Whether to distill into the full language world model")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")

    # Saving and logging
    parser.add_argument("--save_path", type=str, default="./distilled_model", help="Path to save the distilled model")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")

    # Gradient accumulation and optimization
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")

    return parser.parse_args()

# ==========================
# Main Function
# ==========================

def main():
    args = parse_args()
    print("Arguments parsed successfully.")

    # Create save directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Save directory created: {args.save_path}")
    print(f"Log directory created: {args.log_dir}")
    print(f"Checkpoint directory created: {args.checkpoint_dir}")

    # Modify teacher_model_path to include the snapshot
    model_base_path = args.teacher_model_name
    snapshots_dir = os.path.join(model_base_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        print(f"No snapshots found in {model_base_path}.")
        sys.exit(1)
    
    # Assuming you want the latest snapshot
    snapshot_folders = sorted(os.listdir(snapshots_dir), reverse=True)
    if not snapshot_folders:
        print(f"No snapshot folders found in {snapshots_dir}.")
        sys.exit(1)
    
    latest_snapshot = os.path.join(snapshots_dir, snapshot_folders[0])
    print(f"Using latest snapshot: {latest_snapshot}")

    # Load tokenizer from the latest snapshot
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(latest_snapshot)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # Load data
    if args.use_npy_data:
        if not args.npy_data_paths or not args.npy_data_roles:
            print("Data paths and roles are required for npy data files.")
            sys.exit(1)
        print("Loading data from .npy files...")
        train_loader, val_loader = prepare_data_from_npy(
            tokenizer, args.max_length, args.batch_size, args.npy_data_paths, args.npy_data_roles
        )
        print("Data loaded and preprocessed successfully.")
    elif args.dataset_name:
        print(f"Loading dataset '{args.dataset_name}' with config '{args.config if args.config else 'No config'}'...")
        try:
            if args.config:
                dataset = load_dataset(args.dataset_name, args.config)
            else:
                dataset = load_dataset(args.dataset_name)
        except ValueError as ve:
            print(f"Error loading dataset: {ve}")
            sys.exit(1)
        train_loader, val_loader = prepare_data(tokenizer, dataset, args.max_length, args.batch_size)
        print("Data loaded and preprocessed successfully.")
    elif args.custom_data_paths:
        print(f"Loading custom data files: {args.custom_data_paths}")
        custom_data = load_custom_data_from_files(args.custom_data_paths)
        train_loader, val_loader = load_custom_data(
            args=argparse.Namespace(max_length=args.max_length, batch_size=args.batch_size),
            tokenizer=tokenizer,
            custom_data=custom_data
        )
        print("Custom data loaded and preprocessed successfully.")
    else:
        print("No data source specified.")
        sys.exit(1)

    # Call distill_model with latest_snapshot as teacher_model_name
    distill_model(
        teacher_model_name=latest_snapshot,  # Changed from args.teacher_model_name
        student_model_name=args.student_model_name,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        distill_full_model=args.distill_full_model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        save_path=args.save_path,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay
    )

if __name__ == "__main__":
    main()
