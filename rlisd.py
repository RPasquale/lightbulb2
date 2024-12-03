# generate_rlhf.py

import os
import random
import numpy as np
import logging
import yaml
import json
import jsonlines
import csv
from io import StringIO
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from datasets import load_dataset, get_dataset_config_names

# ---------------------- Configuration Management ----------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ----------------------------- Logging Setup -----------------------------

def setup_logging(log_file: str):
    """
    Configure logging for the script.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',  # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# -------------------------- Helper Functions -----------------------------

def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move tensor to the specified device.

    Args:
        tensor (torch.Tensor): The tensor to move.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: Tensor on the target device.
    """
    return tensor.to(device)

def split_into_chunks(text: str, max_words: int = 512) -> List[str]:
    """
    Split text into chunks, each containing up to max_words words.

    Args:
        text (str): The text to split.
        max_words (int, optional): Maximum number of words per chunk. Defaults to 512.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

def compute_attention_weights(current_step_desc: str, previous_outputs: List[str], vectorizer: TfidfVectorizer) -> np.ndarray:
    """
    Compute attention weights based on cosine similarity between the current step description and previous outputs.

    Args:
        current_step_desc (str): Description of the current step.
        previous_outputs (List[str]): Outputs from previous steps.
        vectorizer (TfidfVectorizer): Initialized TF-IDF vectorizer.

    Returns:
        np.ndarray: Attention weights.
    """
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

def generate_options_for_step(prompt: str, tokenizer, model, device: torch.device, config: Dict[str, Any], num_options: int = 3, max_new_tokens: int = 150) -> List[str]:
    """
    Generate multiple-choice options (A, B, C) for a given step.

    Args:
        prompt (str): The prompt to send to the language model.
        tokenizer: Tokenizer associated with the language model.
        model: The language model for text generation.
        device (torch.device): The device to run the model on.
        config (Dict[str, Any]): Configuration parameters for generation.
        num_options (int, optional): Number of options to generate. Defaults to 3.
        max_new_tokens (int, optional): Maximum tokens to generate per option. Defaults to 150.

    Returns:
        List[str]: List of generated options labeled A), B), C), etc.
    """
    options = []
    input_encoding = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = to_device(input_encoding['input_ids'], device)
    attention_mask = to_device(input_encoding['attention_mask'], device)

    for idx in range(num_options):
        try:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=config['model']['generation_settings']['no_repeat_ngram_size'],
                pad_token_id=tokenizer.pad_token_id,
                do_sample=config['model']['generation_settings']['do_sample'],
                temperature=config['model']['generation_settings']['temperature'],
                top_p=config['model']['generation_settings']['top_p']
            )
            option_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the generated option after the prompt
            if prompt.lower() in option_text.lower():
                option = option_text.lower().replace(prompt.lower(), '').strip()
            else:
                # Fallback if prompt is not found in the generated text
                option = option_text.strip()
            # Ensure options are labeled as A), B), C), etc.
            option_label = f"{chr(65 + idx)})"  # 65 is ASCII for 'A'
            # Remove any existing label if present
            if option.startswith(('A)', 'B)', 'C)', 'a)', 'b)', 'c)')):
                option = option.split(')', 1)[1].strip()
            option = f"{option_label} {option}"
            options.append(option)
        except Exception as e:
            logging.error(f"Error generating option {idx+1}: {e}")
            continue
    return options

def evaluate_options(current_step_desc: str, options: List[str], vectorizer: TfidfVectorizer) -> List[float]:
    """
    Evaluate options based on their cosine similarity to the current step description.

    Args:
        current_step_desc (str): Description of the current step.
        options (List[str]): Generated options.
        vectorizer (TfidfVectorizer): Initialized TF-IDF vectorizer.

    Returns:
        List[float]: Similarity scores for each option.
    """
    rewards = []
    for option in options:
        documents = [current_step_desc, option]
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        reward = similarity
        rewards.append(reward)
    return rewards

def generate_instruction(answer_text: str, tokenizer, model, device: torch.device, config: Dict[str, Any]) -> str:
    """
    Generate an instruction that leads to a given answer.

    Args:
        answer_text (str): The answer for which to generate an instruction.
        tokenizer: Tokenizer associated with the language model.
        model: The language model for text generation.
        device (torch.device): The device to run the model on.
        config (Dict[str, Any]): Configuration parameters for generation.

    Returns:
        str: Generated instruction.
    """
    prompt = f"Write an instruction that would lead to the following answer:\n{answer_text}"
    input_encoding = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = to_device(input_encoding['input_ids'], device)
    attention_mask = to_device(input_encoding['attention_mask'], device)

    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config['model']['max_new_tokens_instruction'],
            no_repeat_ngram_size=config['model']['generation_settings']['no_repeat_ngram_size'],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=config['model']['generation_settings']['do_sample'],
            temperature=config['model']['generation_settings']['temperature'],
            top_p=config['model']['generation_settings']['top_p']
        )
        instruction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logging.info(f"Generated Instruction: {instruction}\n")
    except Exception as e:
        logging.error(f"Error generating instruction: {e}")
        instruction = ""
    return instruction

# ---------------------- Data Extraction Functions ------------------------

def get_data_seeds_from_pdfs() -> List[str]:
    """
    Extract data seeds from PDF files provided by the user.

    Returns:
        List[str]: Extracted data seeds.
    """
    logging.info("Selected data source: PDFs")
    pdf_paths_input = input("Enter the paths to your PDF files, separated by commas: ").strip()
    if not pdf_paths_input:
        logging.warning("No PDF paths provided. Skipping PDF data source.")
        return []
    
    # Parse CSV input to handle quotes and commas within paths
    f = StringIO(pdf_paths_input)
    reader = csv.reader(f, skipinitialspace=True)
    pdf_paths = next(reader, [])
    data_seeds = []
    
    for pdf_path in pdf_paths:
        pdf_path = pdf_path.strip('"').strip("'")
        if not os.path.isfile(pdf_path):
            logging.warning(f"File not found: {pdf_path}. Skipping.")
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
                        logging.warning(f"No text found on page {page_num} of {pdf_path}.")
                sentences = text.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        chunks = split_into_chunks(sentence)
                        data_seeds.extend(chunks)
                logging.info(f"Extracted {len(data_seeds)} data seeds from {pdf_path}.")
        except Exception as e:
            logging.error(f"Error reading PDF {pdf_path}: {e}")
            continue
    logging.info(f"Total extracted data seeds from PDFs: {len(data_seeds)}\n")
    return data_seeds

def get_data_seeds_from_huggingface_dataset() -> List[str]:
    """
    Extract data seeds from a Hugging Face dataset provided by the user.

    Returns:
        List[str]: Extracted data seeds.
    """
    logging.info("Selected data source: Hugging Face Dataset")
    dataset_name = input("Enter the name of the Hugging Face dataset (e.g., 'squad'): ").strip()
    if not dataset_name:
        logging.warning("Dataset name cannot be empty. Skipping Hugging Face dataset.")
        return []
    
    try:
        configs = get_dataset_config_names(dataset_name)
        
        if not configs:
            logging.info(f"No configurations found for dataset '{dataset_name}'. Loading without specifying config.")
            dataset = load_dataset(dataset_name)
        elif len(configs) == 1:
            config = configs[0]
            logging.info(f"Only one configuration '{config}' found for dataset '{dataset_name}'. Using it.")
            dataset = load_dataset(dataset_name, config)
        else:
            logging.info(f"Multiple configurations found for dataset '{dataset_name}':")
            for idx, cfg in enumerate(configs, start=1):
                logging.info(f"{idx}. {cfg}")
            selected_idx_input = input(f"Select a configuration [1-{len(configs)}]: ").strip()
            try:
                selected_idx = int(selected_idx_input)
                if 1 <= selected_idx <= len(configs):
                    config = configs[selected_idx - 1]
                    logging.info(f"Loading dataset '{dataset_name}' with configuration '{config}'.")
                    dataset = load_dataset(dataset_name, config)
                else:
                    logging.warning("Invalid selection. Skipping Hugging Face dataset.")
                    return []
            except ValueError:
                logging.warning("Invalid input. Skipping Hugging Face dataset.")
                return []
    except Exception as e:
        error_message = str(e)
        if 'trust_remote_code' in error_message or 'custom code' in error_message:
            logging.error(f"Error loading dataset '{dataset_name}': {e}")
            trust = input("This dataset requires 'trust_remote_code=True'. Do you want to proceed? (y/n): ").strip().lower()
            if trust == 'y':
                try:
                    dataset = load_dataset(dataset_name, trust_remote_code=True)
                except Exception as e2:
                    logging.error(f"Failed to load dataset with 'trust_remote_code=True': {e2}\n")
                    return []
            else:
                logging.info("Skipping Hugging Face dataset.")
                return []
        elif 'Config name is missing' in error_message:
            try:
                configs = get_dataset_config_names(dataset_name)
                if not configs:
                    logging.warning(f"No configurations found for dataset '{dataset_name}'. Cannot load.")
                    return []
                logging.info(f"Dataset '{dataset_name}' requires a configuration. Available configurations:")
                for idx, cfg in enumerate(configs, start=1):
                    logging.info(f"{idx}. {cfg}")
                selected_idx_input = input(f"Select a configuration [1-{len(configs)}]: ").strip()
                try:
                    selected_idx = int(selected_idx_input)
                    if 1 <= selected_idx <= len(configs):
                        config = configs[selected_idx - 1]
                        logging.info(f"Loading dataset '{dataset_name}' with configuration '{config}'.")
                        dataset = load_dataset(dataset_name, config)
                    else:
                        logging.warning("Invalid selection. Skipping Hugging Face dataset.")
                        return []
                except ValueError:
                    logging.warning("Invalid input. Skipping Hugging Face dataset.")
                    return []
            except Exception as e3:
                logging.error(f"Failed to retrieve configurations for dataset '{dataset_name}': {e3}\n")
                return []
        else:
            logging.error(f"Error loading dataset '{dataset_name}': {e}\n")
            return []
    
    # Extract data seeds from the dataset
    try:
        if 'train' in dataset:
            split = 'train'
        else:
            split = list(dataset.keys())[0]
            logging.info(f"No 'train' split found. Using '{split}' split.")
        
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
        
        logging.info(f"Extracted {len(data_seeds)} data seeds from dataset '{dataset_name}'.\n")
        return data_seeds
    except Exception as e:
        logging.error(f"Error processing dataset '{dataset_name}': {e}\n")
        return []

def get_data_seeds_from_npy_files() -> List[str]:
    """
    Extract data seeds from .npy files provided by the user.

    Returns:
        List[str]: Extracted data seeds.
    """
    logging.info("Selected data source: .npy Files")
    npy_paths_input = input("Enter the paths to your .npy files, separated by commas: ").strip()
    if not npy_paths_input:
        logging.warning("No .npy paths provided. Skipping .npy data source.")
        return []
    
    # Parse CSV input to handle quotes and commas within paths
    f = StringIO(npy_paths_input)
    reader = csv.reader(f, skipinitialspace=True)
    npy_paths = next(reader, [])
    data_seeds = []
    
    for npy_path in npy_paths:
        npy_path = npy_path.strip('"').strip("'")
        if not os.path.isfile(npy_path):
            logging.warning(f"File not found: {npy_path}. Skipping.")
            continue
        try:
            data = np.load(npy_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                data = data.flatten()
            for item in data:
                if isinstance(item, (str, np.str_)):
                    chunks = split_into_chunks(item)
                    data_seeds.extend(chunks)
            logging.info(f"Extracted {len(data_seeds)} data seeds from {npy_path}.")
        except Exception as e:
            logging.error(f"Error reading .npy file {npy_path}: {e}")
            continue
    logging.info(f"Total extracted data seeds from .npy files: {len(data_seeds)}\n")
    return data_seeds

# ----------------------------- Main Function ------------------------------

def main():
    """
    Main function to execute the RLHF data generation process.
    """
    # Load configuration
    config_path = "config.yaml"
    if not os.path.isfile(config_path):
        print(f"Configuration file '{config_path}' not found. Exiting.")
        return
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config['output']['log_file'])
    logging.info("Script started.")
    
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load tokenizer and model
    model_path = config['model']['path']
    model_type = config['model']['type']
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Pad token not found. Set pad_token to eos_token.")
        
        config_model = AutoConfig.from_pretrained(model_path)
        logging.info(f"Model Configuration: {config_model}\n")
        
        if model_type in ['gpt2', 'gpt']:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        elif model_type in ['gemma2']:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        elif model_type in ['t5', 'bart', 'mbart']:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            logging.warning(f"Assuming model type '{model_type}' is a causal language model. Using AutoModelForCausalLM.")
        
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
            logging.info("Model pad_token_id not set. Set to tokenizer's pad_token_id.")
        
        logging.info("Model loaded successfully.\n")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    # Define sequential steps
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
    
    # Initialize TF-IDF Vectorizer with configured parameters
    tfidf_params = config['processing']['tfidf_vectorizer_params']
    vectorizer = TfidfVectorizer(**tfidf_params)
    
    # Selection Menu for Data Sources
    logging.info("Select the sources of data seeds (you can select multiple, separated by commas):")
    data_sources = config['data_sources']
    options = []
    if data_sources.get('pdf', {}).get('enabled', False):
        options.append("1. PDFs")
    if data_sources.get('huggingface', {}).get('enabled', False):
        options.append("2. Hugging Face Dataset")
    if data_sources.get('npy', {}).get('enabled', False):
        options.append("3. .npy Files")
    options.append("4. Finish selection")
    
    for option in options:
        logging.info(option)
    
    selections_input = input("Enter the numbers corresponding to your choices (e.g., 1,3): ").strip()
    if not selections_input:
        logging.error("No selections made. Exiting.")
        return
    
    # Parse selections
    f = StringIO(selections_input)
    reader = csv.reader(f, skipinitialspace=True)
    selections = next(reader, [])
    selections = [selection.strip() for selection in selections if selection.strip().isdigit()]
    
    if not selections:
        logging.error("No valid selections made. Exiting.")
        return
    
    # Extract data seeds based on selections
    data_seeds = []
    for selection in selections:
        if selection == '1' and data_sources.get('pdf', {}).get('enabled', False):
            pdf_seeds = get_data_seeds_from_pdfs()
            data_seeds.extend(pdf_seeds)
        elif selection == '2' and data_sources.get('huggingface', {}).get('enabled', False):
            hf_seeds = get_data_seeds_from_huggingface_dataset()
            data_seeds.extend(hf_seeds)
        elif selection == '3' and data_sources.get('npy', {}).get('enabled', False):
            npy_seeds = get_data_seeds_from_npy_files()
            data_seeds.extend(npy_seeds)
        elif selection == '4':
            logging.info("Finished data source selection.")
            break
        else:
            logging.warning(f"Invalid or disabled selection: {selection}. Skipping.")
    
    if not data_seeds:
        logging.error("No data seeds extracted from the selected sources. Exiting.")
        return
    
    # Remove duplicates and shuffle
    data_seeds = list(set(data_seeds))
    random.shuffle(data_seeds)
    
    # Compute total number of words and determine number of data seeds
    total_words = sum(len(seed.split()) for seed in data_seeds)
    max_words_per_seed = config['processing']['max_words_per_seed']
    num_seeds = total_words // max_words_per_seed
    if total_words % max_words_per_seed != 0:
        num_seeds += 1
    logging.info(f"Total words extracted: {total_words}")
    logging.info(f"Each data seed will have up to {max_words_per_seed} words.")
    logging.info(f"Number of data seeds to generate: {num_seeds}\n")
    
    # Truncate data_seeds if necessary
    if len(data_seeds) > num_seeds:
        data_seeds = data_seeds[:num_seeds]
        logging.info(f"Truncated data seeds to {num_seeds} for proportionality.\n")
    
    logging.info(f"Using {len(data_seeds)} data seeds for data generation.\n")
    
        # Prepare filenames for RLHF and instruction datasets
    dataset_name = input("Enter a name for your datasets (without extension): ").strip()
    if not dataset_name:
        logging.error("Dataset name cannot be empty. Exiting.")
        return

    rlhf_filename = f"{dataset_name}{config['output']['rlhf_suffix']}"
    instruction_filename = f"{dataset_name}{config['output']['instruction_suffix']}"
    
    # Load existing data if files already exist
    existing_rlhf_data = []
    existing_instruction_data = []
    
    if os.path.exists(rlhf_filename):
        logging.info(f"Found existing RLHF data file: {rlhf_filename}. Loading existing data.")
        with jsonlines.open(rlhf_filename, mode='r') as reader:
            existing_rlhf_data = [entry for entry in reader]
    
    if os.path.exists(instruction_filename):
        logging.info(f"Found existing Instruction data file: {instruction_filename}. Loading existing data.")
        with jsonlines.open(instruction_filename, mode='r') as reader:
            existing_instruction_data = [entry for entry in reader]

    # Track already processed data seeds
    processed_data_seeds = set([entry['data_seed'] for entry in existing_rlhf_data])
    
    # Open files for appending new entries
    with jsonlines.open(rlhf_filename, mode='a') as rlhf_file, jsonlines.open(instruction_filename, mode='a') as instruction_file:
        try:
            # Generate RLHF data for each data seed and step
            for idx, data_seed in enumerate(data_seeds, start=1):
                if data_seed in processed_data_seeds:
                    logging.info(f"Data Seed {idx}/{len(data_seeds)} already processed. Skipping.")
                    continue
                logging.info(f"Processing Data Seed {idx}/{len(data_seeds)}: {data_seed}")
                
                previous_steps_outputs = []
                for step_num in range(1, len(steps) + 1):
                    step_info = steps[step_num]
                    step_text = step_info['description'].format(data_seed=data_seed)
                    
                    # Compute attention weights based on step description and previous outputs
                    attention_weights = compute_attention_weights(step_text, previous_steps_outputs, vectorizer)
                    
                    # Construct prompt with weighted previous outputs
                    if len(attention_weights) > 0:
                        weighted_outputs = [f"[Weight: {weight:.2f}] {output}" for weight, output in zip(attention_weights, previous_steps_outputs)]
                        prompt = step_text + "\n" + "\n".join(weighted_outputs) + "\nOptions:"
                    else:
                        prompt = step_text + "\nOptions:"
                    
                    # Generate options
                    options = generate_options_for_step(prompt, tokenizer, model, device, config)
                    if not options:
                        logging.warning(f"Failed to generate options for Step {step_num} of Data Seed '{data_seed}'. Skipping...")
                        continue
                    
                    # Evaluate and rank options
                    rewards = evaluate_options(step_text, options, vectorizer)
                    sorted_indices = np.argsort(rewards)[::-1]
                    correct_option = options[sorted_indices[0]]
                    incorrect_option = options[sorted_indices[-1]]
                    
                    # Save RLHF data entry
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
                    logging.info(f"RLHF Entry written for Step {step_num}.")
                    
                    # Store correct option for subsequent step prompts
                    previous_steps_outputs.append(correct_option)
                    
                    # Generate instruction based on the correct option
                    instruction = generate_instruction(correct_option, tokenizer, model, device, config)
                    if instruction:
                        instruction_entry = {
                            'instruction': instruction,
                            'answer': correct_option
                        }
                        instruction_file.write(instruction_entry)
                        logging.info(f"Instruction Entry written for Step {step_num}.")
                    else:
                        logging.warning(f"Failed to generate instruction for Step {step_num}. Skipping instruction saving.")
                    
                    # Display results
                    logging.info(f"Step {step_num}:\n{step_text}\n")
                    logging.info(f"Correct Option:\n{correct_option}\n")
                    logging.info(f"Incorrect Option:\n{incorrect_option}\n")
                    logging.info("-" * 80 + "\n")
                logging.info("=" * 100 + "\n")
        except KeyboardInterrupt:
            logging.warning("Data generation interrupted by user.")
        finally:
            logging.info(f"\nRLHF dataset saved to {rlhf_filename}")
            logging.info(f"Instruction dataset saved to {instruction_filename}\n")

if __name__ == "__main__":
    main()

