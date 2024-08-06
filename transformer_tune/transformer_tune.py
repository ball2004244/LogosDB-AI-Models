'''
Finetune Huggingface transformer models on multidata summarization task
'''
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Sample inputs and outputs
data = {
    "inputs": [
        "Quantum entanglement is a phenomenon in quantum physics where two or more particles become interconnected, regardless of the distance between them. When particles are entangled, the quantum state of each particle cannot be described independently, even when separated by large distances. This 'spooky action at a distance,' as Einstein called it, has fascinating implications for quantum computing and communication. Scientists are exploring ways to harness entanglement for ultra-secure encryption and faster information processing. However, maintaining entanglement over long distances remains a significant challenge due to environmental interference.",
        "Tardigrades, also known as water bears, are microscopic animals renowned for their extraordinary resilience. These eight-legged creatures can survive extreme conditions that would be fatal to most life forms. They can withstand temperatures from near absolute zero to over 150°C, pressures six times greater than those in the deepest ocean trenches, and radiation levels far beyond what humans can tolerate. Tardigrades achieve this remarkable feat through cryptobiosis, a state of extreme metabolic depression. In this state, they can survive without water for decades, only to reanimate when conditions improve.",
        "Dark energy is a mysterious force that appears to be causing the expansion of the universe to accelerate. Discovered in the late 1990s through observations of distant supernovae, dark energy contradicts the long-held belief that the universe's expansion should be slowing due to gravity. This invisible energy is estimated to make up about 68% of the universe, yet its nature remains one of the biggest puzzles in modern cosmology. Theories range from a cosmological constant inherent to space itself to dynamic fields that change over time. Understanding dark energy could revolutionize our comprehension of the universe's past, present, and future.",
        "CRISPR-Cas9 is a revolutionary gene-editing tool that has transformed molecular biology and genetics research. Derived from a bacterial defense mechanism, CRISPR allows scientists to make precise changes to DNA sequences in living cells. This technology holds immense potential for treating genetic disorders, developing more resilient crops, and even tackling complex environmental issues. However, it also raises ethical concerns, particularly regarding its potential use in human embryos. As CRISPR techniques continue to advance, scientists and policymakers grapple with balancing its tremendous potential benefits against possible risks and ethical implications.",
        "Neuroplasticity refers to the brain's ability to reorganize itself by forming new neural connections throughout life. This remarkable feature allows the brain to adapt to new experiences, learn, recover from brain injuries, and compensate for lost functions. Contrary to early beliefs that the adult brain was fixed, we now know that it remains plastic well into adulthood. Neuroplasticity underlies many cognitive functions, including memory formation and skill acquisition. Understanding and harnessing neuroplasticity could lead to more effective treatments for neurological disorders and improved strategies for lifelong learning and cognitive enhancement."
    ],
    "outputs": [
        "Quantum entanglement is a phenomenon where particles become interconnected regardless of distance, with potential applications in quantum computing and secure communication. This 'spooky action at a distance' challenges our understanding of physics and poses difficulties in maintaining over long distances.",
        "Tardigrades, or water bears, are microscopic animals known for their incredible ability to survive extreme conditions through a state called cryptobiosis. These resilient creatures can withstand extreme temperatures, pressures, and radiation, remaining dormant for decades before reanimating when conditions improve.",
        "Dark energy is a mysterious force causing the accelerated expansion of the universe, contradicting the expected slowing effect of gravity. Comprising about 68% of the universe, its nature remains one of the biggest puzzles in modern cosmology, with various theories attempting to explain it.",
        "CRISPR-Cas9 is a groundbreaking gene-editing tool that allows scientists to make precise changes to DNA sequences in living cells. While it holds immense potential for treating genetic disorders and developing resilient crops, it also raises ethical concerns, particularly regarding its use in human embryos.",
        "Neuroplasticity refers to the brain's ability to reorganize itself by forming new neural connections throughout life, enabling adaptation, learning, and recovery from injuries. This phenomenon, which occurs well into adulthood, underlies many cognitive functions and could lead to improved treatments for neurological disorders and strategies for lifelong learning."
    ]
}

# Step 1: Prepare the dataset
dataset = Dataset.from_dict(data)

# Step 2: Choose a pre-trained model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize the inputs and outputs
def preprocess_function(examples):
    inputs = [f"summarize: {doc}" for doc in examples["inputs"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["outputs"], max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 3: Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

# Step 4: Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Step 5: Save the model
save_path = "./finetuned_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
