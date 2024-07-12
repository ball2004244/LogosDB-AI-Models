from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    # Load model and tokenizer
    model_name = "google/t5-efficient-tiny"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    print("Model and tokenizer loaded successfully")

    # Prepare input text for summarization
    input_text = "summarize: Wormholes, hypothetical topological features of spacetime, have long captivated the imagination of scientists and science fiction enthusiasts alike. These theoretical tunnels through space and time offer the tantalizing possibility of shortcuts across the universe, potentially allowing for faster-than-light travel. First conceptualized in 1935 by Einstein and Rosen, wormholes emerge from solutions to the equations of general relativity. While mathematically possible, the existence of traversable wormholes faces significant challenges. They would require exotic matter with negative energy density to remain open and stable, a concept that pushes the boundaries of known physics. If they exist, wormholes could connect distant regions of space-time, even linking different universes or timelines. This property has led to speculation about their potential for time travel, though the paradoxes this might create remain unresolved. Despite their theoretical intrigue, no observational evidence for wormholes has been found. Current research focuses on refining mathematical models and exploring potential detection methods. As our understanding of quantum gravity and the nature of spacetime evolves, wormholes continue to serve as a fascinating intersection of theoretical physics, cosmology, and our quest to unravel the universe's deepest mysteries."

    input_tokens = tokenizer.encode(input_text, return_tensors="pt")

    print("Input tokenized, generating output...")

    # Generate output
    output_tokens = model.generate(input_tokens, max_new_tokens=100, num_beams=4, length_penalty=2.0, early_stopping=True)

    # Decode and print the output
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("Generated output (raw):")
    print(output_tokens)
    print("Generated output (decoded):")
    print(output_text)
    print("Output length:", len(output_text))
except Exception as e:
    print(f"An error occurred: {e}")