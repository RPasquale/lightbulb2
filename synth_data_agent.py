import random
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
from concurrent.futures import ThreadPoolExecutor

# Helper function to move tensors to the GPU if available
def to_device(tensor):
    return tensor.to('cuda') if torch.cuda.is_available() else tensor

# MCTS and Node Class Implementation
class Node:
    def __init__(self, instruction, parent=None):
        self.instruction = instruction
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0

    def is_terminal(self):
        return self.visits > 5

    def expand(self):
        new_instructions = [generate_variation(self.instruction) for _ in range(2)]
        self.children = [Node(instr, parent=self) for instr in new_instructions]
        return self.children

    def evaluate(self):
        return critic_response(self.instruction)

class MCTS:
    def __init__(self):
        self.exploration_weight = 1.0

    def select(self, node):
        while not node.is_terminal():
            if not node.children:
                return node.expand()[0]
            else:
                node = self.best_child(node)
        return node

    def best_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            exploit = child.value / (child.visits + 1)
            explore = np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
            score = exploit + self.exploration_weight * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def run(self, root, iterations=100):
        for _ in range(iterations):
            leaf = self.select(root)
            reward = leaf.evaluate()
            self.backpropagate(leaf, reward)
        return self.best_child(root)

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
critic_model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')

# Sample raw data seeds
raw_data_seeds = [
    "The mitochondrion is known as the powerhouse of the cell.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    "Einstein's theory of relativity revolutionized the understanding of space, time, and gravity."
]

# Content Transformation Flow
def content_transformation(seed_text):
    transformations = [
        lambda x: x + " Explain this in detail.",
        lambda x: "Discuss the significance of: " + x,
        lambda x: x + " How does this relate to real-world scenarios?",
        lambda x: "What are the implications of the following statement: " + x
    ]
    transformation = random.choice(transformations)
    return transformation(seed_text)

transformed_data = [content_transformation(seed) for seed in raw_data_seeds]
print("Transformed Data:", transformed_data)  # Debug print

# Seed Instruction Generation Flow
def generate_instruction(transformed_text, max_new_tokens=150):
    prompts = [
        lambda x: "Please provide a detailed explanation on: " + x,
        lambda x: "Can you elaborate on the following topic? " + x,
        lambda x: "Write an essay about: " + x,
        lambda x: "Create a comprehensive guide on: " + x
    ]
    prompt = random.choice(prompts)
    input_text = prompt(transformed_text)
    input_ids = to_device(tokenizer.encode(input_text, return_tensors='pt'))
    attention_mask = to_device(tokenizer(input_text, return_tensors='pt').attention_mask)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
    instruction = tokenizer.decode(output[0], skip_special_tokens=True)
    return instruction

generated_instructions = [generate_instruction(text) for text in transformed_data]
print("Generated Instructions:", generated_instructions)  # Debug print

# Instruction Refinement Flow
def refine_instruction(instruction):
    refinements = [
        lambda x: x + " Now provide an example to illustrate this.",
        lambda x: x + " Can you break this down into simpler terms for a beginner?",
        lambda x: x + " What are the key takeaways from this explanation?",
        lambda x: x + " How would you explain this to someone new to the topic?"
    ]
    refinement = random.choice(refinements)
    return refinement(instruction)

def iterative_refinement(instruction, iterations=3):
    refined_instruction = instruction
    for _ in range(iterations):
        refined_instruction = refine_instruction(refined_instruction)
    return refined_instruction

refined_instructions = [iterative_refinement(instr) for instr in generated_instructions]
print("Refined Instructions:", refined_instructions)  # Debug print

# Cleaning the refined instructions
def clean_text(text):
    text = re.sub(r'\.{2,}', '.', text)  # Replace multiple dots with a single dot
    text = re.sub(r',\s*:\.\,', '', text)  # Remove patterns like ", :.,"
    text = re.sub(r'\(and\)\s*\(and\)', '', text)  # Remove repeated "(and) (and)"
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'-\s+', '', text)  # Remove dash followed by space
    text = re.sub(r'([.?!])\s*(?=[a-zA-Z])', r'\1 ', text)  # Ensure proper sentence spacing
    text = re.sub(r'\s*([.,?!])\s*', r'\1 ', text)  # Ensure proper punctuation spacing
    text = re.sub(r'(\s*[.,?!]){2,}', r'\1', text)  # Remove repeated punctuation
    return text.strip()

cleaned_instructions = [clean_text(instr) for instr in refined_instructions]
print("Cleaned Instructions:", cleaned_instructions)  # Debug print

# Remove Repeated Phrases
def remove_repeated_phrases(text):
    sentences = text.split('. ')
    seen = set()
    result = []
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            result.append(sentence)
    return '. '.join(result)

cleaned_instructions = [remove_repeated_phrases(instr) for instr in cleaned_instructions]
print("Instructions after Removing Repeated Phrases:", cleaned_instructions)  # Debug print

# Generate new variations of the instruction
def generate_variation(instruction):
    input_text = f"Generate a variation of: {instruction}"
    input_ids = to_device(tokenizer.encode(input_text, return_tensors='pt'))
    attention_mask = to_device(tokenizer(input_text, return_tensors='pt').attention_mask)
    output = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
    variation = tokenizer.decode(output[0], skip_special_tokens=True)
    return variation

# Monte Carlo Tree Search Integration for Few-Shot Learning and RLHF
def critic_response(instruction):
    critic_prompts = [
        lambda x: f"Is the following response legible and meaningful? {x}",
        lambda x: f"Does the following explanation make sense and is it coherent? {x}",
        lambda x: f"Is the following description clear and easy to understand? {x}"
    ]
    critic_prompt_func = random.choice(critic_prompts)
    critic_prompt = critic_prompt_func(instruction)
    input_ids = to_device(tokenizer.encode(critic_prompt, return_tensors='pt'))
    attention_mask = to_device(tokenizer(critic_prompt, return_tensors='pt').attention_mask)
    output = critic_model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
    critique = tokenizer.decode(output[0], skip_special_tokens=True)
    return "yes" in critique.lower() or "good" in critique.lower()

def run_mcts(root, iterations=100):
    mcts = MCTS()
    for _ in range(iterations):
        node = mcts.select(root)
        if node.is_terminal():
            value = node.evaluate()
        else:
            children = node.expand()
            value = node.evaluate()
        mcts.backpropagate(node, value)
    best_node = mcts.best_child(root)
    return best_node.instruction

root_nodes = [Node(instr) for instr in cleaned_instructions]
with ThreadPoolExecutor() as executor:
    final_instructions = list(executor.map(run_mcts, root_nodes))

print("Final Instructions after MCTS:", final_instructions)  # Debug print

# Final formatting with minimal structure
def minimally_structure_instruction(instr):
    sections = [
        "Introduction",
        "Detailed Explanation",
        "Real-world Applications",
        "Examples",
        "Key Takeaways",
        "Simplified Explanation for Beginners",
        "Conclusion"
    ]
    return "\n\n".join([f"{section}:\n{content.strip()}" for section, content in zip(sections, instr.split('. ')) if content])

structured_instructions = [minimally_structure_instruction(instr) for instr in final_instructions]

# Print the refined instructions in the desired structured format
def print_instructions(instructions):
    for instr in instructions:
        print(instr)
        print("\n" + "-"*80 + "\n")

print_instructions(structured_instructions)
