import argparse
import sys
import os
from lightbulb.train_agent import train_agent
from lightbulb.test_agent import TestAgent, run_test_session
from twisted.internet import reactor, task
from lightbulb_custom import main as lightbulb_custom_main
from lightbulb.distill import distill_model  # Import the distillation function
from transformers import AutoTokenizer, logging

# Suppress transformers warnings for cleaner output
logging.set_verbosity_error()

def parse_main_args():
    parser = argparse.ArgumentParser(description="Main Menu for Selecting Tasks")
    
    # Task selection
    parser.add_argument('--task', type=str, choices=[
                        'train_llm_world', 
                        'train_agent', 
                        'test_agent', 
                        'inference_llm', 
                        'inference_world_model', 
                        'advanced_inference',
                        'distill_full_model',       # New option for full model distillation
                        'distill_domain_specific'   # New option for selective distillation
                    ], 
                        required=True, 
                        help='Choose task to execute: train_llm_world, train_agent, test_agent, inference_llm, inference_world_model, advanced_inference, distill_full_model, distill_domain_specific')
    
    # Common arguments
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pretrained model name for LLM')
    parser.add_argument('--student_model_name', type=str, default='distilgpt2', help='Name of the student model for distillation')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name for training')
    parser.add_argument('--dataset_config', type=str, default=None, help='Dataset configuration name')
    parser.add_argument('--tokenizer_name', type=str, help='Tokenizer name for data processing')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for training')
    parser.add_argument('--temperature', type=float, default=2.0, help='Distillation temperature')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    
    # Distillation-specific arguments
    parser.add_argument('--save_path', type=str, default="./distilled_model", help="Path to save the distilled model")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for TensorBoard logs")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--early_stopping_patience', type=int, default=3, help="Early stopping patience")
    
    # Inference-specific arguments
    parser.add_argument('--query', type=str, default='', help='Query for the test_agent or inference tasks')
    parser.add_argument('--inference_mode', type=str, choices=['without_world_model', 'world_model', 'world_model_tree_of_thought'], help='Inference mode')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search during inference')
    parser.add_argument('--n_tokens_predict', type=int, default=3, help='Number of tokens to predict at each step during inference')
    parser.add_argument('--mcts_iterations', type=int, default=10, help='Number of MCTS iterations during inference')
    parser.add_argument('--mcts_exploration_constant', type=float, default=1.414, help='Exploration constant for MCTS during inference')
    
    # Distillation-specific arguments
    parser.add_argument('--distill_full_model', action="store_true", help="Whether to distill the full model or not")
    parser.add_argument('--query_terms', type=str, nargs="+", help="Query terms for domain-specific distillation")
    
    # Load model for inference
    parser.add_argument('--load_model', type=str, help='Path to load the distilled model for inference')
    
    return parser.parse_args()

def main():
    # Parse arguments for the main function
    args = parse_main_args()
    
    # Set up the tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name  # Use model_name if tokenizer_name is not provided
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Check if vocabulary size is divisible by 2
    vocab_size = tokenizer.vocab_size
    if vocab_size % 2 != 0:
        raise ValueError(f"Tokenizer vocabulary size {vocab_size} is not divisible by 2. Please use a compatible tokenizer.")
    
    # Execute tasks based on user input
    if args.task == 'train_llm_world':
        print("Starting LLM and World Model Training...")
        # Directly call the world model main function with appropriate arguments
        sys.argv = [
            'lightbulb_custom.py', 
            '--mode', 'train', 
            '--model_name', args.model_name, 
            '--dataset_name', args.dataset_name, 
            '--dataset_config', args.dataset_config,
            '--batch_size', str(args.batch_size), 
            '--num_epochs', str(args.num_epochs),
            '--max_length', str(args.max_length)
        ]
        lightbulb_custom_main()
    
    elif args.task == 'train_agent':
        print("Starting Agent Training...")
        # Call the train_agent function from train_agent.py using Twisted reactor
        d = task.deferLater(reactor, 0, train_agent)
        d.addErrback(lambda failure: print(f"An error occurred: {failure}", exc_info=True))
        d.addBoth(lambda _: reactor.stop())
        reactor.run()
    
    elif args.task == 'test_agent':
        print("Starting Test Agent...")
        test_agent = TestAgent()
        if args.query:
            # Directly process a single query
            result = test_agent.process_query(args.query)
            print("\nAgent's response:")
            print(result)
        else:
            # Run the interactive session
            reactor.callWhenRunning(run_test_session)
            reactor.run()
    
    elif args.task in ['inference_llm', 'inference_world_model', 'advanced_inference']:
        print("Starting Inference Task...")
        inference_mode_map = {
            'inference_llm': 'without_world_model',
            'inference_world_model': 'world_model',
            'advanced_inference': 'world_model_tree_of_thought'
        }
        selected_inference_mode = inference_mode_map.get(args.task, 'world_model_tree_of_thought')
        lightbulb_inf_args = [
            'lightbulb_custom.py',
            '--mode', 'inference',
            '--model_name', args.model_name,
            '--query', args.query,
            '--max_length', str(args.max_length),
            '--inference_mode', selected_inference_mode,
            '--beam_size', str(args.beam_size),
            '--n_tokens_predict', str(args.n_tokens_predict),
            '--mcts_iterations', str(args.mcts_iterations),
            '--mcts_exploration_constant', str(args.mcts_exploration_constant)
        ]
        if args.load_model:
            lightbulb_inf_args += ['--load_model', args.load_model]
        sys.argv = lightbulb_inf_args
        lightbulb_custom_main()
    
    elif args.task == 'distill_full_model':
        print("Starting Full Model Distillation...")
        distill_model(
            teacher_model_name=args.model_name,
            student_model_name=args.student_model_name,
            dataset_name=args.dataset_name,
            config=args.dataset_config,
            distill_full_model=True,
            query_terms=None,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            save_path=args.save_path,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            early_stopping_patience=args.early_stopping_patience,
            accumulation_steps=1,  # Default or specify if needed
            max_grad_norm=1.0,      # Default or specify if needed
            weight_decay=0.01       # Default or specify if needed
        )
    
    elif args.task == 'distill_domain_specific':
        print("Starting Domain-Specific Distillation...")
        if not args.query_terms:
            print("Error: --query_terms must be provided for domain-specific distillation.")
            sys.exit(1)
        distill_model(
            teacher_model_name=args.model_name,
            student_model_name=args.student_model_name,
            dataset_name=args.dataset_name,
            config=args.dataset_config,
            distill_full_model=False,
            query_terms=args.query_terms,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            save_path=args.save_path,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            early_stopping_patience=args.early_stopping_patience,
            accumulation_steps=1,  # Default or specify if needed
            max_grad_norm=1.0,      # Default or specify if needed
            weight_decay=0.01       # Default or specify if needed
        )
    
    else:
        print(f"Unknown task: {args.task}")
        sys.exit(1)

if __name__ == "__main__":
    main()
