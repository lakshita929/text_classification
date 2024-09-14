from datasets import load_dataset
from tokenizer import get_tokenizer, tokenize_dataset
from model import load_model
from train import train_model
from evaluate import evaluate_model
from save_model import save_model

# Step 1: Load the dataset
dataset = load_dataset('imdb')

# Step 2: Get the tokenizer and tokenize the dataset
tokenizer = get_tokenizer()
tokenized_dataset = tokenize_dataset(dataset, tokenizer)

# Step 3: Load the pre-trained model
model = load_model()

# Step 4: Train the model
trainer = train_model(model, tokenized_dataset)

# Step 5: Evaluate the model
eval_results = evaluate_model(trainer)
print("Evaluation results:", eval_results)

# Step 6: Save the model and tokenizer
save_model(model, tokenizer)
