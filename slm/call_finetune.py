from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the saved model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_qwen2_0.5b_4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Verify model and tokenizer are loaded correctly
assert model is not None, "Model failed to load"
assert tokenizer is not None, "Tokenizer failed to load"

# Define the input text
input_text = '''
Biology is the scientific study of life and living organisms, encompassing a vast array of topics from the microscopic world of cells to the complexities of entire ecosystems. At its core, biology explores the fundamental processes that govern all living things, including growth, reproduction, metabolism, and adaptation. The field is divided into numerous subdisciplines, each focusing on specific aspects of life. Molecular biology delves into the intricate workings of DNA, RNA, and proteins, while cellular biology examines the structure and function of cells, the basic units of life. Genetics investigates the inheritance and variation of traits across generations, while evolutionary biology studies the changes in species over time. Ecology explores the interactions between organisms and their environment, while physiology examines how living systems function. Advances in technology have revolutionized biological research, enabling scientists to manipulate genes, visualize cellular processes in real-time, and analyze vast amounts of genomic data. These breakthroughs have led to significant progress in fields such as medicine, agriculture, and conservation. As our understanding of biology continues to grow, it offers solutions to global challenges like disease prevention, food security, and environmental protection, while also raising important ethical questions about the boundaries of scientific intervention in living systems.
'''

# Debugging: Print the shape of the inputs
print(f"Inputs: {input_text}")

def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
print("Summary:")
generate_text(input_text)


