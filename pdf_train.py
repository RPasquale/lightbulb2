import os
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import networkx as nx
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, BertModel, BertTokenizer
import pandas as pd
import json
import math
import random
import torch
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import wikipediaapi
import torch
from gensim.models import Word2Vec
import numpy as np
from datetime import datetime, timedelta
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SpaCy model for NER and dependency parsing
nlp = spacy.load("en_core_web_sm")

# Load BERT model for contextual relationship detection
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load Model and Tokenizer
model_name = "gpt2-medium"  # Upgraded to a more capable model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Pretrained Keyphrase Extraction model setup
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
# NLTK data setup
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def preprocess_text(raw_text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', raw_text).strip()
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        # Remove leading/trailing whitespace
        cleaned = sentence.strip()
        
        # Remove very short sentences (likely noise)
        if len(cleaned) > 10:
            # Ensure the sentence ends with proper punctuation
            if not cleaned[-1] in '.!?':
                cleaned += '.'
            cleaned_sentences.append(cleaned)
    
    return cleaned_sentences

# Advanced Concept Extraction
def advanced_concept_extraction(text):
    doc = nlp(text)
    concepts = []
    
    # Named Entity Recognition (NER)
    for ent in doc.ents:
        concepts.append((ent.text, ent.label_))
    
    # Keyphrase Extraction
    try:
        # Limit the input text length to avoid potential issues
        max_input_length = 1024
        truncated_text = text[:max_input_length]
        
        keyphrases = summarization_pipeline(
            truncated_text, 
            max_length=10, 
            min_length=5, 
            do_sample=False,
            num_return_sequences=3  # Reduce the number of returned sequences
        )
        for phrase in keyphrases:
            concepts.append((phrase['summary_text'], "KEYPHRASE"))
    except Exception as e:
        print(f"Error in keyphrase extraction: {str(e)}")
        # Fallback to a simple noun chunk extraction if the pipeline fails
        for chunk in doc.noun_chunks:
            concepts.append((chunk.text, "NOUN_CHUNK"))
    
    # Remove duplicates and return only the text part of the concepts
    unique_concepts = list(set(concept[0] for concept in concepts))
    return unique_concepts

# Enhanced Relationship Building
def build_advanced_knowledge_graph(concepts, sentences):
    graph = nx.Graph()
    
    # Add concepts as nodes
    for concept in concepts:
        graph.add_node(concept)
    
    # Add edges based on dependency parsing and co-occurrence
    for sentence in sentences:
        doc = nlp(sentence)
        entities = [ent.text for ent in doc.ents]
        
        # Co-occurrence Analysis
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                graph.add_edge(entities[i], entities[j], weight=1)
        
        # Dependency Parsing
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj'):
                graph.add_edge(token.head.text, token.text, weight=2)
    
    return graph

def enrich_graph_with_external_data(graph):
    # Define a proper user agent
    user_agent = "YourAppName/1.0 (https://yourappwebsite.com/; contact@yourappwebsite.com)"
    
    # Create a Wikipedia API object with a user agent
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent
    )

    for node in graph.nodes:
        if "WIKIPEDIA" not in graph.nodes[node]:
            try:
                # Get the Wikipedia page for the node
                page = wiki_wiki.page(node)
                if page.exists():
                    # Extract the summary of the page
                    summary = page.summary[:500]  # Limit the summary to the first 500 characters
                    graph.nodes[node]["WIKIPEDIA"] = summary
            except Exception as e:
                print(f"Failed to get Wikipedia summary for {node}: {e}")
                continue
    return graph


# Graph Embeddings with GNNs
def learn_graph_embeddings(graph):
    # Create sentences (random walks) from the graph
    def graph_to_sentences(G, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            for node in G.nodes():
                walk = [node]
                for _ in range(walk_length - 1):
                    neighbors = list(G.neighbors(walk[-1]))
                    if not neighbors:
                        break
                    walk.append(random.choice(neighbors))
                walks.append([str(node) for node in walk])
        return walks

    # Generate random walks
    walks = graph_to_sentences(graph, num_walks=10, walk_length=5)

    # Train Word2Vec model
    model = Word2Vec(sentences=walks, vector_size=64, window=5, min_count=0, sg=1, workers=4)

    # Add embeddings to the graph
    for node in graph.nodes():
        graph.nodes[node]['embedding'] = model.wv[str(node)]

    return graph


# Semantic Similarity for Nodes
def calculate_semantic_similarity(graph):
    node_embeddings = {}
    
    for node in graph.nodes:
        tokens = bert_tokenizer(node, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**tokens)
        node_embeddings[node] = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            sim_score = cosine_similarity(node_embeddings[nodes[i]], node_embeddings[nodes[j]])[0][0]
            if sim_score > 0.7:  # Similarity threshold
                graph.add_edge(nodes[i], nodes[j], weight=sim_score)
    
    return graph

# Temporal Information (Placeholder for incorporating time aspect)
def incorporate_temporal_information(graph, timestamp):
    # Convert timestamp to datetime object if it's not already
    if not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(timestamp)

    # Add timestamp to graph metadata
    graph.graph['timestamp'] = timestamp

    # Add temporal information to each node
    for node in graph.nodes():
        # Simulate some temporal data (e.g., last update time)
        last_update = timestamp - timedelta(days=random.randint(0, 30))
        graph.nodes[node]['last_update'] = last_update

        # Add a temporal relevance score (higher for more recent updates)
        time_diff = (timestamp - last_update).total_seconds()
        relevance_score = 1 / (1 + time_diff / (24 * 3600))  # Normalize to [0, 1]
        graph.nodes[node]['temporal_relevance'] = relevance_score

    # Add temporal edges or modify existing edge weights
    for u, v, data in graph.edges(data=True):
        u_time = graph.nodes[u]['last_update']
        v_time = graph.nodes[v]['last_update']
        time_diff = abs((u_time - v_time).total_seconds())
        temporal_weight = 1 / (1 + time_diff / (24 * 3600))  # Normalize to [0, 1]
        
        # Modify edge weight based on temporal information
        current_weight = data.get('weight', 1.0)
        new_weight = (current_weight + temporal_weight) / 2
        graph[u][v]['weight'] = new_weight

    return graph

# MCTS Implementation (no changes needed)
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Represents the current context or text segment
        self.parent = parent
        self.action = action
        self.children = []
        self._number_of_visits = 0
        self._total_reward = 0

    def expand(self):
        children = []
        for variation in generate_variations(self.state):
            child_node = Node(state=variation, parent=self)
            children.append(child_node)
        return children

    def is_terminal(self):
        return len(self.state.split()) > 50 or len(generate_variations(self.state)) == 0

    def simulate(self):
        return evaluate_context(self.state)

    def backpropagate(self, reward):
        self._number_of_visits += 1
        self._total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def best_child(self, c_param=1.4):
        choices_weights = []
        for child in self.children:
            if child._number_of_visits == 0:
                choices_weights.append(float('inf'))  # Assign a high value to unexplored nodes
            else:
                weight = (child._total_reward / child._number_of_visits) + c_param * math.sqrt(
                    (2 * math.log(self._number_of_visits)) / child._number_of_visits
                )
                choices_weights.append(weight)
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children, key=lambda c: c._number_of_visits)

class MCTS:
    def __init__(self, root, model, tokenizer, device='cpu'):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run(self, iterations=1000):
        for _ in range(iterations):
            node = self.select(self.root)
            reward = node.simulate()
            node.backpropagate(reward)
        return self.root.most_visited_child()

    def select(self, node):
        while not node.is_terminal():
            if not node.children:
                return self.expand(node)
            elif any(child._number_of_visits == 0 for child in node.children):
                return random.choice([child for child in node.children if child._number_of_visits == 0])
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        children = node.expand()
        node.children.extend(children)
        return random.choice(children)

class SentenceCompletionCriteria(StoppingCriteria):
    def __init__(self, tokens_per_sentence=20):
        self.tokens_per_sentence = tokens_per_sentence
        self.sentence_end_tokens = {'.', '!', '?', '\n'}

    def __call__(self, input_ids, scores, **kwargs):
        last_token = tokenizer.decode([input_ids[0][-1]])
        if last_token in self.sentence_end_tokens and len(input_ids[0]) >= self.tokens_per_sentence:
            return True
        return False

def generate_text(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    stopping_criteria = StoppingCriteriaList([SentenceCompletionCriteria()])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            min_length=50  # Encourage longer generations
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_process_text(generated_text)  # Apply post-processing directly

def generate_variations(text_segment):
    variations = [
        text_segment + " Provide more context.",
        text_segment + " Explain this in simpler terms.",
        text_segment + " What are the implications?",
        text_segment + " Compare and contrast with a similar situation."
    ]
    return variations

def evaluate_context(text_segment):
    inputs = tokenizer(text_segment, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reward = len(generated_text) + len(set(generated_text.split()))  # Reward for length and diversity
    return reward

def extract_concepts(sentences):
    words = [word_tokenize(sentence) for sentence in sentences]
    flat_words = [item for sublist in words for item in sublist]
    word_counts = Counter(flat_words)
    concepts = [word for word, count in word_counts.items() if count > 2]
    return concepts

def post_process_text(text):
    # Split the text into sentences
    sentences = text.split('.')
    
    unique_sentences = []
    for sentence in sentences:
        # Remove leading/trailing whitespace
        cleaned_sentence = sentence.strip()
        
        # Skip empty sentences
        if not cleaned_sentence:
            continue
        
        # Ensure the sentence starts with a capital letter
        cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
        
        # Add the sentence if it's not a duplicate and ends with proper punctuation
        if cleaned_sentence not in unique_sentences:
            if not cleaned_sentence[-1] in '.!?':
                cleaned_sentence += '.'
            unique_sentences.append(cleaned_sentence)
    
    # Join the sentences
    clean_text = ' '.join(unique_sentences)
    
    # Ensure the text ends with proper punctuation
    if not clean_text[-1] in '.!?':
        clean_text += '.'
    
    return clean_text

def determine_root_context(sentences, concepts):
    for sentence in sentences:
        if any(concept in sentence for concept in concepts):
            return sentence
    return sentences[0]

def create_instruction_template(task_type, content):
    templates = {
        "summarization": f"Summarize the following text:\n{content}",
        "qa": f"Answer the following question based on the given context:\nContext: {content}\nQuestion: [QUESTION]",
        "sentiment": f"Analyze the sentiment of the following text:\n{content}",
        "paraphrase": f"Paraphrase the following text:\n{content}",
        "key_points": f"Extract the key points from the following text:\n{content}",
    }
    return templates.get(task_type, f"Perform the following task:\n{content}")

def generate_few_shot_prompts(sentences):
    few_shot_prompts = []
    task_types = ["summarization", "qa", "sentiment", "paraphrase", "key_points"]
    for sentence in sentences:
        task_type = random.choice(task_types)
        instruction = create_instruction_template(task_type, sentence)
        
        input_text = f"{instruction}\n{sentence}"
        generated_output = generate_text(input_text)
        clean_output = post_process_text(generated_output)
        
        entry = {
            "instruction": instruction,
            "input": sentence,
            "output": clean_output
        }
        if quality_check(entry):
            few_shot_prompts.append(entry)
    return few_shot_prompts

def generate_graph_augmented_prompt(graph, text, max_input_length=400, max_new_tokens=150):
    related_concepts = []
    for node in graph.nodes:
        if isinstance(node, str) and node in text:
            neighbors = list(graph.neighbors(node))
            related_concepts.extend(neighbors[:2])  # Limit to 2 related concepts per node
    
    augmented_text = text + " Related concepts: " + ", ".join(str(c) for c in related_concepts)
    inputs = tokenizer(augmented_text, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
    
    # Update to use max_new_tokens and set do_sample=True
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, 
                             no_repeat_ngram_size=3, temperature=0.8, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_dataset_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def post_process_text(text):
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        if sentence.strip() and sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())
    clean_text = '. '.join(unique_sentences) + '.'
    
    if not clean_text.endswith(('.', '!', '?')):
        clean_text = clean_text.rsplit('.', 1)[0] + '.'
    
    return clean_text

def quality_check(entry):
    min_length = 20
    max_length = 500
    if len(entry['output']) < min_length or len(entry['output']) > max_length:
        return False
    if entry['output'].count('.') < 2:
        return False
    if len(set(entry['output'].split('.'))) < 2:
        return False
    return True

class ContextNode(Node):
    def __init__(self, text_segment, parent=None, action=None):
        super().__init__(text_segment, parent, action)
        self.text_segment = text_segment
    
    def expand(self):
        children = []
        for i, variation in enumerate(generate_variations(self.text_segment)):
            child_node = ContextNode(variation, parent=self, action=i)
            children.append(child_node)
        return children

    def is_terminal(self):
        return len(self.text_segment) > 500 or len(generate_variations(self.text_segment)) == 0

    def simulate(self):
        return evaluate_context(self.text_segment)

# Process PDFs in a folder and generate data
def process_pdfs_in_folder(folder_path):
    all_prompts = []
    
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith(".pdf"):
            try:
                pdf_path = os.path.join(folder_path, pdf_file)
                print(f"Processing file: {pdf_file}")
                raw_text = extract_text_from_pdf(pdf_path)
                print(f"Extracted text length: {len(raw_text)}")
                sentences = preprocess_text(raw_text)
                print(f"Number of sentences: {len(sentences)}")
                concepts = advanced_concept_extraction(raw_text)
                print(f"Number of concepts: {len(concepts)}")
                knowledge_graph = build_advanced_knowledge_graph(concepts, sentences)
                print(f"Graph built with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
                
                
                root_text = determine_root_context(sentences, concepts)
                
                # Initialize the root node with the dynamic root context
                root_node = ContextNode(root_text)
                
                # Run MCTS to explore possible contexts
                mcts = MCTS(root_node, model, tokenizer, device)
                best_node = mcts.run(iterations=100)
                best_context = best_node.text_segment
                print(f"Best Context for {pdf_file}: {best_context}")
                
                # Generate synthetic prompts using the best context from MCTS
                synthetic_prompts = []
                for sentence in sentences:
                    context_with_mcts = generate_graph_augmented_prompt(knowledge_graph, best_context + " " + sentence)
                    task_types = ["summarization", "qa", "sentiment", "paraphrase", "key_points"]
                    task_type = random.choice(task_types)
                    instruction = create_instruction_template(task_type, context_with_mcts)
                    
                    input_text = f"{instruction}\n{context_with_mcts}"
                    generated_output = generate_text(input_text)
                    clean_output = post_process_text(generated_output)
                    
                    entry = {
                        "instruction": instruction,
                        "input": context_with_mcts,
                        "output": clean_output
                    }
                    if quality_check(entry):
                        synthetic_prompts.append(entry)

                # Combine few-shot and synthetic prompts
                few_shot_prompts = generate_few_shot_prompts(sentences[:50])  # Generate 50 few-shot examples
                all_prompts.extend(few_shot_prompts + synthetic_prompts)
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                print(f"Error details: {type(e).__name__}, {str(e)}")
                import traceback
                traceback.print_exc()

    # Save the combined dataset as JSONL
    save_dataset_to_jsonl(all_prompts, 'combined_instruction_few_shot_and_synthetic_data.jsonl')

    print(f"Generated {len(all_prompts)} instruction-input-output triples.")
# Folder path containing PDFs
#folder_path = "C:/Users/Admin/Desktop/ai-ml-web-app/backend/pdfs"
#process_pdfs_in_folder(folder_path)
