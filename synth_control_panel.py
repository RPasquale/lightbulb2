import argparse
from dataset_generation import main as generate_datasets
from ultra_distill import main as distill_model
'''
Instructions:

To generate datasets:
python main.py generate_datasets --dataset_name my_dataset

To distill a model:
python main.py distill_model --teacher_model teacher --student_model student --rlhf_d

'''
def main():
    parser = argparse.ArgumentParser(description="Control Dataset Generation and Model Distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Dataset Generation Subparser
    dataset_parser = subparsers.add_parser("generate_datasets", help="Generate datasets for model distillation")
    dataset_parser.add_argument("--dataset_name", type=str, required=True, help="Name for the generated datasets (without extension)")

    # Model Distillation Subparser
    distill_parser = subparsers.add_parser("distill_model", help="Distill a model using the generated datasets")
    distill_parser.add_argument("--teacher_model", type=str, required=True, help="Path or name of the teacher model")
    distill_parser.add_argument("--student_model", type=str, required=True, help="Path or name of the student model to be trained")
    distill_parser.add_argument("--rlhf_data_path", type=str, required=True, help="Path to the RLHF JSONL file")
    distill_parser.add_argument("--instruction_data_path", type=str, required=True, help="Path to the Instruction JSONL file")
    distill_parser.add_argument("--output_dir", type=str, default="./distilled_model", help="Directory to save the distilled model")
    distill_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    distill_parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    distill_parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    distill_parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    distill_parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to load")

    args = parser.parse_args()

    if args.command == "generate_datasets":
        generate_datasets(args)
    elif args.command == "distill_model":
        distill_model(args)

if __name__ == "__main__":
    main()