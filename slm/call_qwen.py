from typing import List
from rich import print
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

'''
This function takes a list of bullet points and generates a summary for each point using the Qwen2 model.
'''
def call_qwen(data: List[str], model="qwen2:0.5b-instruct-q8_0") -> List[str]:
    start = time.perf_counter()

    prompt = '''Summarize each of the following input into strictly 1 sentence each. Answer nothing but summarized answers.

    
        You output format:
        ```
        - <Summarized answer 1>
        - <Summarized answer 2>
        ```
        
        Inputs for summarize:
        '''

    # Concatenate all data into one for summarization, then convert output back to list
    for datum in data:
        prompt += f"\n- {datum}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    res = response.json()['response']
    time_taken = time.perf_counter() - start

    
    # convert response back to list, each item start with '-'
    print('Time taken:', time_taken)
    print('Prompt:', prompt)
    print('Response:', res)
    return []


# Example usage
data = [
    r"A black hole is an extreme cosmic phenomenon where gravity's pull is so intense that nothing, not even light, can escape. Born from the collapse of massive stars or the merging of galaxies, these enigmatic objects warp the very fabric of spacetime. At their heart lies the singularity, a point of infinite density shrouded by the event horizon—the boundary beyond which all paths lead inexorably inward. Black holes come in various sizes, from stellar-mass to supermassive behemoths lurking at galactic centers. They shape the universe around them, devouring nearby matter and emitting powerful jets of energy. Despite their mysterious nature, black holes continue to captivate scientists and fuel groundbreaking research in astrophysics and cosmology.",
    r"Income inequality refers to the disparate distribution of wealth and income within a society. This growing chasm between the rich and poor has become a pressing global issue, exacerbating social tensions and hindering economic mobility. Factors contributing to this divide include technological advancements, globalization, and policy decisions favoring the wealthy. The consequences are far-reaching, affecting access to education, healthcare, and opportunities for advancement. Critics argue that extreme inequality undermines social cohesion, democratic processes, and overall economic stability. Proposed solutions range from progressive taxation and social welfare programs to education reforms and labor market interventions. As the debate continues, addressing income inequality remains a complex challenge for policymakers and societies worldwide.",
    r"The Renaissance, meaning 'rebirth,' was a transformative period in European history spanning the 14th to 17th centuries. Originating in Italy, this cultural movement marked the transition from the Middle Ages to modernity. It was characterized by a revival of classical learning, innovative artistic expression, and scientific inquiry. Figures like Leonardo da Vinci, Michelangelo, and Raphael revolutionized art, while thinkers such as Erasmus and Machiavelli challenged traditional philosophies. The invention of the printing press by Johannes Gutenberg accelerated the spread of knowledge. The Renaissance fostered humanism, emphasizing individual potential and critical thinking. This era's impact on art, literature, science, and politics laid the foundation for the modern Western world.",
    r"Electoral systems are the mechanisms by which voters choose political representatives and governments. These systems vary widely across democracies, each with its own strengths and weaknesses. The most common types include first-past-the-post, proportional representation, and mixed systems. First-past-the-post, used in countries like the UK and US, typically results in two-party dominance but can lead to disproportionate representation. Proportional representation, common in European countries, allows for more diverse political voices but may result in coalition governments and political instability. Mixed systems attempt to balance these approaches. The choice of electoral system significantly influences political landscapes, affecting party structures, policy-making, and voter engagement. Debates about electoral reform often center on balancing fair representation with governmental stability.",
    r"Photosynthesis is a fundamental biological process that sustains life on Earth. It's the mechanism by which plants, algae, and some bacteria convert light energy into chemical energy, producing oxygen as a byproduct. The process occurs in chloroplasts, specialized cell organelles containing chlorophyll. Photosynthesis consists of two main stages: light-dependent reactions and light-independent reactions (Calvin cycle). In the first stage, light energy is captured and converted into ATP and NADPH. The Calvin cycle then uses these products to convert carbon dioxide into glucose. This process not only provides energy for plants but also forms the base of most food chains. Additionally, photosynthesis plays a crucial role in the global carbon cycle and oxygen production, making it essential for life as we know it.",
    r"Climate change refers to long-term shifts in global or regional climate patterns, primarily due to human activities. The burning of fossil fuels, deforestation, and industrial processes release greenhouse gases, leading to the warming of the Earth's atmosphere. This phenomenon results in rising sea levels, extreme weather events, and disruptions to ecosystems. The impacts of climate change are far-reaching, affecting agriculture, water resources, and human health. Mit"
]

if __name__ == "__main__":
    response = call_qwen(data)
    print(f'Num items: {len(response)}')
    print(response)
