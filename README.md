# LogosDB-AI-Models
This repository contains the code for the AI models used in the LogosDB project. 

1. Extractive Summary: extract portions of sentences that have the highest importance in the text using Tf-idf and TextRank algorithms.

2. Abstractive Summary: finetune Google T5 model on Reddit TIFU & Wikipedia Summary Dataset.

## Installation
1. Clone the repository
2. Install the required packages (Different methods have different requirements, see below)
2.1 Keyword Summary:
```bash
pip install keybert
```

2.2 Extractive Summary:
For extractive summary requirements, install the following:
```bash
pip install cython nltk networkx numpy scikit-learn
```

Compile the cython code:
```bash
cd extractive_sum/cython
python3 setup.py build_ext --inplace
```

To use compiled Cython code, import the compiled module to your C++ code:
```cpp
#include "extractive_sum/cython/summarizer.h"
```

