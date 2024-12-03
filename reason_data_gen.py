import os
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import torch
import json
import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import csv
from io import StringIO
from datasets import load_dataset, get_dataset_config_names

# Determine the device based on whether CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper function to move tensors to the correct device
def to_device(tensor):
    return tensor.to(device)

# Function to split text into chunks of max_words
def split_into_chunks(text, max_words=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

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

# Function to generate options A), B), C) for each step
def generate_options_for_step(prompt, tokenizer, model, device, num_options=3, max_new_tokens=150):
    options = []
    # Get input IDs and attention mask
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
            # Extract the generated option after the prompt
            option = option_text[len(prompt):].strip()
            # Ensure options are labeled as A), B), C)
            option_label = f"{chr(65 + idx)})"  # Converts 0 to 'A)', 1 to 'B)', etc.
            # If the option already starts with a label, remove it
            if option.startswith(('A)', 'B)', 'C)', 'a)', 'b)', 'c)')):
                # Remove existing label
                option = option.split(')', 1)[1].strip()
            option = f"{option_label} {option}"
            options.append(option)
    except Exception as e:
        print(f"Error during generation: {e}")
        return options
    return options

# Critic function to evaluate options using cosine similarity
def evaluate_options(current_step_desc, options, vectorizer):
    rewards = []
    for option in options:
        # Combine current step description and option for similarity
        documents = [current_step_desc, option]
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Normalize similarity to a range, e.g., 0 to 1
        reward = similarity
        rewards.append(reward)
    return rewards

# Function to generate an instruction from an answer
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
        # Debugging: Print the generated instruction
        print(f"Generated Instruction: {instruction}\n")
    except Exception as e:
        print(f"Error during instruction generation: {e}")
        instruction = ""
    return instruction

# Function to extract data seeds from PDFs using csv module to handle commas within quotes
def get_data_seeds_from_pdfs():
    print("\nYou have selected PDFs.")
    pdf_paths_input = input("Enter the paths to your PDF files, separated by commas: ").strip()
    if not pdf_paths_input:
        print("No PDF paths provided. Skipping PDF data source.")
        return []
    # Use csv module to correctly parse the input, handling quotes and commas within paths
    f = StringIO(pdf_paths_input)
    reader = csv.reader(f, skipinitialspace=True)
    pdf_paths = next(reader, [])
    data_seeds = []
    for pdf_path in pdf_paths:
        # Remove any surrounding quotes
        pdf_path = pdf_path.strip('"').strip("'")
        if not os.path.isfile(pdf_path):
            print(f"File not found: {pdf_path}. Skipping.")
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
                        print(f"Warning: No text found on page {page_num} of {pdf_path}.")
                # Split the text into sentences based on periods
                sentences = text.split('.')
                # Clean, filter, and split sentences into chunks
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        chunks = split_into_chunks(sentence)
                        data_seeds.extend(chunks)
                print(f"Extracted {len(data_seeds)} data seeds from {pdf_path}.")
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            continue
    print(f"Total extracted data seeds from PDFs: {len(data_seeds)}\n")
    return data_seeds

def get_data_seeds_from_huggingface_dataset():
    print("\nYou have selected Hugging Face Dataset.")
    dataset_name = input("Enter the name of the Hugging Face dataset (e.g., 'squad'): ").strip()
    if not dataset_name:
        print("Dataset name cannot be empty. Skipping Hugging Face dataset.")
        return []
    
    try:
        # Retrieve available configurations for the dataset
        configs = get_dataset_config_names(dataset_name)
        
        if not configs:
            # No configurations available; load dataset directly
            print(f"No configurations found for dataset '{dataset_name}'. Loading without specifying config.")
            dataset = load_dataset(dataset_name)
        elif len(configs) == 1:
            # Only one configuration available; use it
            config = configs[0]
            print(f"Only one configuration '{config}' found for dataset '{dataset_name}'. Using it.")
            dataset = load_dataset(dataset_name, config)
        else:
            # Multiple configurations available; prompt user to select one
            print(f"Multiple configurations found for dataset '{dataset_name}':")
            for idx, cfg in enumerate(configs, start=1):
                print(f"{idx}. {cfg}")
            selected_idx = input(f"Select a configuration [1-{len(configs)}]: ").strip()
            try:
                selected_idx = int(selected_idx)
                if 1 <= selected_idx <= len(configs):
                    config = configs[selected_idx - 1]
                    print(f"Loading dataset '{dataset_name}' with configuration '{config}'.")
                    dataset = load_dataset(dataset_name, config)
                else:
                    print("Invalid selection. Skipping Hugging Face dataset.")
                    return []
            except ValueError:
                print("Invalid input. Skipping Hugging Face dataset.")
                return []
    except Exception as e:
        # Handle errors related to trust_remote_code or other issues
        error_message = str(e)
        if 'trust_remote_code' in error_message or 'custom code' in error_message:
            print(f"Error loading dataset '{dataset_name}': {e}")
            trust = input("This dataset requires 'trust_remote_code=True'. Do you want to proceed? (y/n): ").strip().lower()
            if trust == 'y':
                try:
                    # Attempt to load dataset with trust_remote_code=True
                    dataset = load_dataset(dataset_name, trust_remote_code=True)
                except Exception as e2:
                    print(f"Failed to load dataset with 'trust_remote_code=True': {e2}\n")
                    return []
            else:
                print("Skipping Hugging Face dataset.")
                return []
        elif 'Config name is missing' in error_message:
            # Handle missing config names
            try:
                configs = get_dataset_config_names(dataset_name)
                if not configs:
                    print(f"No configurations found for dataset '{dataset_name}'. Cannot load.")
                    return []
                print(f"Dataset '{dataset_name}' requires a configuration. Available configurations:")
                for idx, cfg in enumerate(configs, start=1):
                    print(f"{idx}. {cfg}")
                selected_idx = input(f"Select a configuration [1-{len(configs)}]: ").strip()
                try:
                    selected_idx = int(selected_idx)
                    if 1 <= selected_idx <= len(configs):
                        config = configs[selected_idx - 1]
                        print(f"Loading dataset '{dataset_name}' with configuration '{config}'.")
                        dataset = load_dataset(dataset_name, config)
                    else:
                        print("Invalid selection. Skipping Hugging Face dataset.")
                        return []
                except ValueError:
                    print("Invalid input. Skipping Hugging Face dataset.")
                    return []
            except Exception as e3:
                print(f"Failed to retrieve configurations for dataset '{dataset_name}': {e3}\n")
                return []
        else:
            print(f"Error loading dataset '{dataset_name}': {e}\n")
            return []
    
    # Proceed to extract data seeds
    try:
        # Use the 'train' split if available, else the first available split
        if 'train' in dataset:
            split = 'train'
        else:
            split = list(dataset.keys())[0]
            print(f"No 'train' split found. Using '{split}' split.")
        
        data_seeds = []
        for example in dataset[split]:
            # Attempt to extract 'text', 'content', 'question', or all string fields
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
                # Extract and split all string fields
                for value in example.values():
                    if isinstance(value, str):
                        chunks = split_into_chunks(value)
                        data_seeds.extend(chunks)
        
        print(f"Extracted {len(data_seeds)} data seeds from dataset '{dataset_name}'.\n")
        return data_seeds
    except Exception as e:
        print(f"Error processing dataset '{dataset_name}': {e}\n")
        return []

# Function to extract data seeds from .npy files using csv module to handle commas within quotes
def get_data_seeds_from_npy_files():
    print("\nYou have selected .npy Files.")
    npy_paths_input = input("Enter the paths to your .npy files, separated by commas: ").strip()
    if not npy_paths_input:
        print("No .npy paths provided. Skipping .npy data source.")
        return []
    # Use csv module to correctly parse the input, handling quotes and commas within paths
    f = StringIO(npy_paths_input)
    reader = csv.reader(f, skipinitialspace=True)
    npy_paths = next(reader, [])
    data_seeds = []
    for npy_path in npy_paths:
        # Remove any surrounding quotes
        npy_path = npy_path.strip('"').strip("'")
        if not os.path.isfile(npy_path):
            print(f"File not found: {npy_path}. Skipping.")
            continue
        try:
            data = np.load(npy_path, allow_pickle=True)
            # Flatten the array in case it's multi-dimensional
            if isinstance(data, np.ndarray):
                data = data.flatten()
            # Convert all elements to strings and split into chunks
            for item in data:
                if isinstance(item, (str, np.str_)):
                    chunks = split_into_chunks(item)
                    data_seeds.extend(chunks)
            print(f"Extracted {len(data_seeds)} data seeds from {npy_path}.")
        except Exception as e:
            print(f"Error reading .npy file {npy_path}: {e}")
            continue
    print(f"Total extracted data seeds from .npy files: {len(data_seeds)}\n")
    return data_seeds

# Main execution starts here
def main():
    # Prompt the user for a dataset name
    dataset_name = input("Enter a name for your datasets (without extension): ").strip()
    if not dataset_name:
        raise ValueError("Dataset name cannot be empty.")
    
    # Specify the local path for the cached Gemma model
    model_path = r"C:\Users\Admin\.cache\huggingface\hub\models--google--gemma-2-2b-it\snapshots\299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    
    # Initialize the tokenizer and model from the local path, and move the model to the correct device
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure that the pad_token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load configuration to inspect model type
        config = AutoConfig.from_pretrained(model_path)
        print(f"\nModel Configuration: {config}\n")
        
        # Identify model type
        model_type = config.model_type
        print(f"Detected model type: {model_type}\n")
        
        # Based on model_type, decide which AutoModel class to use
        if model_type in ['gpt2', 'gpt']:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        elif model_type in ['gemma2']:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        elif model_type in ['t5', 'bart', 'mbart']:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        else:
            # Default to AutoModelForCausalLM if model type is unrecognized but assumed to be causal LM
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            print(f"Assuming model type '{model_type}' is a causal language model. Using AutoModelForCausalLM.")
        
        # Ensure that the pad_token_id is set in the model config
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        print(f"Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Define the sequential steps with dependencies and weightings
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
    
    # Selection Menu
    print("Select the sources of data seeds (you can select multiple, separated by commas):")
    print("1. PDFs")
    print("2. Hugging Face Dataset")
    print("3. .npy Files")
    print("4. Finish selection")
    
    selections_input = input("Enter the numbers corresponding to your choices (e.g., 1,3): ").strip()
    if not selections_input:
        print("No selections made. Exiting.")
        exit(1)
    
    # Parse selections
    f = StringIO(selections_input)
    reader = csv.reader(f, skipinitialspace=True)
    selections = next(reader, [])
    selections = [selection.strip() for selection in selections if selection.strip().isdigit()]
    
    if not selections:
        print("No valid selections made. Exiting.")
        exit(1)
    
    # Extract data seeds based on selections
    data_seeds = []
    for selection in selections:
        if selection == '1':
            # PDFs
            pdf_seeds = get_data_seeds_from_pdfs()
            data_seeds.extend(pdf_seeds)
        elif selection == '2':
            # Hugging Face Dataset
            hf_seeds = get_data_seeds_from_huggingface_dataset()
            data_seeds.extend(hf_seeds)
        elif selection == '3':
            # .npy Files
            npy_seeds = get_data_seeds_from_npy_files()
            data_seeds.extend(npy_seeds)
        elif selection == '4':
            # Finish selection
            break
        else:
            print(f"Invalid selection: {selection}. Skipping.")
    
    if not data_seeds:
        print("No data seeds extracted from the selected sources. Exiting.")
        exit(1)
    
    # Remove duplicates and shuffle
    data_seeds = list(set(data_seeds))
    random.shuffle(data_seeds)
    
    # Compute total number of words and determine number of data seeds
    total_words = sum(len(seed.split()) for seed in data_seeds)
    max_words_per_seed = 256
    num_seeds = total_words // max_words_per_seed
    if total_words % max_words_per_seed != 0:
        num_seeds += 1
    print(f"Total words extracted: {total_words}")
    print(f"Each data seed will have up to {max_words_per_seed} words.")
    print(f"Number of data seeds to generate: {num_seeds}\n")
    
    # If the actual number of data_seeds is greater than num_seeds, truncate
    if len(data_seeds) > num_seeds:
        data_seeds = data_seeds[:num_seeds]
        print(f"Truncated data seeds to {num_seeds} for proportionality.\n")
    
    print(f"Using {len(data_seeds)} data seeds for data generation.\n")
    
    # Prepare filenames
    rlhf_filename = f"{dataset_name}_rlhf.jsonl"
    instruction_filename = f"{dataset_name}_instruction.jsonl"
    
    # Check if the files already exist and load existing data
    existing_rlhf_data = []
    existing_instruction_data = []
    
    if os.path.exists(rlhf_filename):
        print(f"Found existing RLHF data file: {rlhf_filename}. Loading existing data.")
        with jsonlines.open(rlhf_filename, mode='r') as reader:
            existing_rlhf_data = [entry for entry in reader]
    
    if os.path.exists(instruction_filename):
        print(f"Found existing Instruction data file: {instruction_filename}. Loading existing data.")
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
                print(f"Data Seed {idx}/{len(data_seeds)} already processed. Skipping.")
                continue
            print(f"Data Seed {idx}/{len(data_seeds)}: {data_seed}\n")
            previous_steps_outputs = []
            for step_num in range(1, len(steps) + 1):
                step_info = steps[step_num]
                # Format the step with data_seed
                step_text = step_info['description'].format(data_seed=data_seed)
                
                # Compute attention weights based on step description and previous outputs
                attention_weights = compute_attention_weights(step_text, previous_steps_outputs, vectorizer)
                
                # Include weighted previous step outputs based on attention
                if len(attention_weights) > 0:
                    weighted_outputs = []
                    for weight, output in zip(attention_weights, previous_steps_outputs):
                        weighted_output = f"[Weight: {weight:.2f}] {output}"
                        weighted_outputs.append(weighted_output)
                    prompt = step_text + "\n" + "\n".join(weighted_outputs) + "\nOptions:"
                else:
                    prompt = step_text + "\nOptions:"
                
                # Generate options
                options = generate_options_for_step(prompt, tokenizer, model, device)
                if not options:
                    print(f"Failed to generate options for Step {step_num} of Data Seed '{data_seed}'. Skipping...\n")
                    continue
                
                # Evaluate options
                current_step_desc = step_info['description'].format(data_seed=data_seed)
                rewards = evaluate_options(current_step_desc, options, vectorizer)
                
                # Get indices sorted by reward (descending)
                sorted_indices = np.argsort(rewards)[::-1]
                
                # Select the best option as the correct option
                correct_option = options[sorted_indices[0]]
                
                # Select the worst option as the incorrect option
                incorrect_option = options[sorted_indices[-1]]
                
                # Store the RLHF data
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
                print(f"RLHF Entry Written for Step {step_num}.\n")
                
                # Store the correct option as output of this step
                previous_steps_outputs.append(correct_option)
                
                # Generate and save instruction immediately after the step
                instruction = generate_instruction(correct_option, tokenizer, model, device)
                
                if instruction:
                    instruction_entry = {
                        'instruction': instruction,
                        'answer': correct_option
                    }
                    instruction_file.write(instruction_entry)
                    print(f"Instruction Entry Written for Step {step_num}.\n")
                else:
                    print(f"Failed to generate instruction for Step {step_num}. Skipping instruction saving.\n")
                
                # Print the results
                print(f"Step {step_num}:\n{step_text}\n")
                print(f"Correct Option:\n{correct_option}\n")
                print(f"Incorrect Option:\n{incorrect_option}\n")
                print("-"*80 + "\n")
            print("="*100 + "\n")
    except KeyboardInterrupt:
        print("\nData generation interrupted by user.")
    finally:
        # Close the files
        rlhf_file.close()
        instruction_file.close()
        print(f"\nRLHF dataset saved to {rlhf_filename}")
        print(f"Instruction dataset saved to {instruction_filename}\n")

if __name__ == "__main__":
    main()
