import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from sklearn.model_selection import train_test_split
from modules.tabular_ml import TabularMLProcessor
from modules.pdf_train_enhance import extract_text_from_pdf, preprocess_text, advanced_concept_extraction, build_advanced_knowledge_graph
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

class EnhancedCombinedLoRATrainer:
    def __init__(self, csv_folder, pdf_folder, output_dir, base_model_name="microsoft/phi-3.5-mini-instruct"):
        self.csv_folder = csv_folder
        self.pdf_folder = pdf_folder
        self.output_dir = output_dir
        self.base_model_name = base_model_name
        self.tabular_processor = TabularMLProcessor()
        self.combined_graph = nx.Graph()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)

    def process_csvs(self, csv_target_columns):
        for csv_file in os.listdir(self.csv_folder):
            if csv_file.endswith(".csv"):
                file_path = os.path.join(self.csv_folder, csv_file)
                df = pd.read_csv(file_path)
                
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()
                
                if csv_file not in csv_target_columns:
                    raise ValueError(f"Target column not specified for {csv_file}")
                
                target_column = csv_target_columns[csv_file].strip()  # Strip whitespace from target column name
                
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in {csv_file}. Available columns: {', '.join(df.columns)}")
                
                graph_data = self.tabular_processor.get_knowledge_graph_data(file_path, target_column)
                if graph_data:
                    csv_graph = nx.node_link_graph(graph_data)
                    self.combined_graph = nx.compose(self.combined_graph, csv_graph)


    def process_pdfs(self):
        for pdf_file in os.listdir(self.pdf_folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, pdf_file)
                raw_text = extract_text_from_pdf(pdf_path)
                sentences = preprocess_text(raw_text)
                concepts = advanced_concept_extraction(raw_text)
                pdf_graph = build_advanced_knowledge_graph(concepts, sentences)
                self.combined_graph = nx.compose(self.combined_graph, pdf_graph)

    def generate_contextual_embeddings(self):
        embeddings = {}
        for node in self.combined_graph.nodes():
            # Get the node and its neighbors
            neighbors = list(self.combined_graph.neighbors(node))
            context = f"{node} - Related: {', '.join(map(str, neighbors[:5]))}"
            
            # Generate embedding using SentenceTransformer
            embedding = self.sentence_transformer.encode(context, convert_to_tensor=True)
            embeddings[node] = embedding.cpu().numpy()
        
        # Normalize embeddings
        scaler = MinMaxScaler()
        normalized_embeddings = scaler.fit_transform(list(embeddings.values()))
        
        return {node: emb for node, emb in zip(embeddings.keys(), normalized_embeddings)}

    def create_enhanced_training_data(self):
        embeddings = self.generate_contextual_embeddings()
        data = []
        for node, embedding in embeddings.items():
            neighbors = list(self.combined_graph.neighbors(node))
            if neighbors:
                # Select multiple neighbors
                selected_neighbors = random.sample(neighbors, min(3, len(neighbors)))
                for neighbor in selected_neighbors:
                    data.append({
                        "input_embedding": embedding.tolist(),
                        "target_embedding": embeddings[neighbor].tolist(),
                        "is_neighbor": 1  # Explicitly set is_neighbor for positive samples
                    })
                
                # Add negative samples
                non_neighbors = list(set(self.combined_graph.nodes()) - set(neighbors) - {node})
                negative_samples = random.sample(non_neighbors, min(3, len(non_neighbors)))
                for neg_sample in negative_samples:
                    data.append({
                        "input_embedding": embedding.tolist(),
                        "target_embedding": embeddings[neg_sample].tolist(),
                        "is_neighbor": 0
                    })
        return data

    def train_enhanced_lora(self):
        training_data = self.create_enhanced_training_data()
        
        # Convert embeddings to text representations
        def embedding_to_text(embedding):
            return " ".join([f"{i}:{v:.4f}" for i, v in enumerate(embedding)])

        texts = [
            f"Input: {embedding_to_text(d['input_embedding'])} "
            f"Target: {embedding_to_text(d['target_embedding'])} "
            f"Is Neighbor: {d['is_neighbor']}"
            for d in training_data
        ]

        # Tokenize the texts
        tokenized_data = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Create a new dataset with the tokenized data
        dataset = Dataset.from_dict(tokenized_data)

        # Perform train-test split
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=['qkv_proj', 'o_proj']  # Updated target modules for Phi model
        )

        peft_model = get_peft_model(self.model, peft_config)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            learning_rate=2e-5,
            fp16=True,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            load_best_model_at_end=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": (predictions == labels).astype(np.float32).mean().item()
            }

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        
        # Save the full model (base + LoRA)
        self.save_model(f"{self.output_dir}/final_model_full")
        
        # Save only the LoRA weights
        peft_model.save_pretrained(f"{self.output_dir}/final_model_lora")

        # Return some metrics for progress tracking
        return trainer.state.log_history

    def save_model(self, path):
        # Save the full model (base + LoRA)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        # Load the full model (base + LoRA)
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def run_with_progress(self, csv_target_columns):
        try:
            yield {"status": "Processing CSVs", "progress": 0}
            self.process_csvs(csv_target_columns)
            yield {"status": "Processing PDFs", "progress": 25}
            self.process_pdfs()
            yield {"status": "Training LoRA model", "progress": 50}
            log_history = self.train_enhanced_lora()
            total_steps = len(log_history)
            for i, log in enumerate(log_history):
                progress = 50 + (i / total_steps) * 50
                yield {"status": "Training", "progress": progress, "metrics": log}
            yield {"status": "Complete", "progress": 100}
        except Exception as e:
            yield {"status": "Error", "message": str(e)}

class PhiWithEnhancedLoRA:
    def __init__(self, base_model_name, lora_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self.load_lora(lora_weights_path)
        
    def load_lora(self, lora_weights_path):
        peft_config = PeftConfig.from_pretrained(lora_weights_path)
        self.model = PeftModel.from_pretrained(self.base_model, lora_weights_path)
        self.model.to(self.device)

    def save_full_model(self, path):
        # Save the full model (base + LoRA)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_full_model(self, path):
        # Load the full model (base + LoRA)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def generate_response(self, prompt, max_length=150):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

