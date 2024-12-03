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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import csv
from io import StringIO

# Determine the device based on whether CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper function to move tensors to the correct device
def to_device(tensor):
    return tensor.to(device)

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
    print(f"Model Configuration: {config}\n")
    
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

# Function to compute attention weights based on similarity
def compute_attention_weights(current_step_desc, previous_outputs):
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
def generate_options_for_step(prompt, num_options=3, max_new_tokens=150):
    options = []
    # Get input IDs and attention mask
    input_encoding = tokenizer(prompt, return_tensors='pt')
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
def evaluate_options(current_step_desc, options):
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
def generate_instruction(answer_text):
    prompt = "Write an instruction that would lead to the following answer:\n" + answer_text
    input_encoding = tokenizer(prompt, return_tensors='pt')
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
    except Exception as e:
        print(f"Error during instruction generation: {e}")
        instruction = ""
    return instruction

# Function to extract data seeds from PDFs using csv module to handle commas within quotes
def get_data_seeds_from_pdfs():
    print("You have selected PDFs.")
    pdf_paths_input = input("Enter the paths to your PDF files, separated by commas: ").strip()
    if not pdf_paths_input:
        print("No PDF paths provided. Exiting.")
        exit(1)
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
                # Clean and filter sentences
                cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
                data_seeds.extend(cleaned_sentences)
                print(f"Extracted {len(cleaned_sentences)} sentences from {pdf_path}.")
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            continue
    print(f"Total extracted data seeds from PDFs: {len(data_seeds)}")
    return data_seeds

# Main execution starts here
if __name__ == "__main__":
    # Extract data seeds from PDFs
    data_seeds = get_data_seeds_from_pdfs()
    
    # Limit the number of data seeds for practical purposes
    data_seeds = data_seeds[:10]
    print(f"\nUsing {len(data_seeds)} data seeds for data generation.\n")
    
    # Generate RLHF data for each data seed and each step
    rlhf_dataset = []
    
    for data_seed in data_seeds:
        print(f"Data Seed: {data_seed}\n")
        previous_steps_outputs = []
        for step_num in range(1, len(steps) + 1):
            step_info = steps[step_num]
            # Format the step with data_seed
            step_text = step_info['description'].format(data_seed=data_seed)
            
            # Compute attention weights based on step description and previous outputs
            attention_weights = compute_attention_weights(step_text, previous_steps_outputs)
            
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
            options = generate_options_for_step(prompt)
            if not options:
                print(f"Failed to generate options for Step {step_num} of Data Seed '{data_seed}'. Skipping...\n")
                continue
            
            # Evaluate options
            current_step_desc = step_info['description'].format(data_seed=data_seed)
            rewards = evaluate_options(current_step_desc, options)
            
            # Get indices sorted by reward (descending)
            sorted_indices = np.argsort(rewards)[::-1]
            
            # Select the best option as the correct option
            correct_option = options[sorted_indices[0]]
            
            # Select the worst option as the incorrect option
            incorrect_option = options[sorted_indices[-1]]
            
            # Store the RLHF data
            rlhf_dataset.append({
                'data_seed': data_seed,
                'step_num': step_num,
                'step_text': step_text,
                'prompt': prompt,
                'correct_option': correct_option,
                'incorrect_option': incorrect_option,
                'attention_weights': attention_weights.tolist() if len(attention_weights) > 0 else []
            })
            
            # Store the correct option as output of this step
            previous_steps_outputs.append(correct_option)
            
            # Print the results
            print(f"Step {step_num}:\n{step_text}\n")
            print(f"Correct Option:\n{correct_option}\n")
            print(f"Incorrect Option:\n{incorrect_option}\n")
            print("-"*80 + "\n")
        print("="*100 + "\n")
    
    # Save RLHF dataset to JSON
    rlhf_filename = f"{dataset_name}_rlhf.json"
    try:
        with open(rlhf_filename, 'w', encoding='utf-8') as f:
            json.dump(rlhf_dataset, f, ensure_ascii=False, indent=4)
        print(f"RLHF dataset saved to {rlhf_filename}\n")
    except Exception as e:
        print(f"Error saving RLHF dataset: {e}")
    
    # Generate instruction data
    instruction_data = []
    
    for item in rlhf_dataset:
        answer_text = item['correct_option']
        instruction = generate_instruction(answer_text)
        
        instruction_data.append({
            'instruction': instruction,
            'answer': answer_text
        })
        print(f"Instruction for Data Seed '{item['data_seed']}', Step {item['step_num']}:\n{instruction}\n")
        print("-"*80 + "\n")
    
    # Save Instruction data to JSON
    instruction_filename = f"{dataset_name}_instruction.json"
    try:
        with open(instruction_filename, 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, ensure_ascii=False, indent=4)
        print(f"Instruction dataset saved to {instruction_filename}\n")
    except Exception as e:
        print(f"Error saving Instruction dataset: {e}")
    
    # Optionally, you can further process or analyze the datasets as needed
