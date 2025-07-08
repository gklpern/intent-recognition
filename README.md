# intent-recognition
The objective of this project is to classify user intents into seven predefined categories: creative, music, playlist, restaurant, screening, rate_book, and weather.



## Intent Recognition with Fine-tuned Gemma Model

### Project Overview

The objective of this project is to classify user intents into seven predefined categories: creative, music, playlist, restaurant, screening, rate_book, and weather. We carried out the following steps in this notebook:

- Performed basic exploratory data analysis, examining class distributions across different intent categories using ratio calculations to understand dataset balance and representation.

![image](https://github.com/user-attachments/assets/eeb0e0d6-3052-4cee-80c9-f034214f594f)


- Analyzed the distribution of intents in the training dataset, calculating proportional ratios for each category (creative, music, playlist, restaurant, screening, rate_book, weather) relative to the total dataset size.
- 
![image](https://github.com/user-attachments/assets/c4e02f36-1103-487e-8815-0292efe8cf5d)


- Employed text preprocessing and normalization techniques to prepare the intent classification dataset for model training.

- Utilized the PEFT (Parameter-Efficient Fine-Tuning) framework to fine-tune the pre-trained google/gemma-1.1-2b-it model, applying the LoRA method with specific configuration parameters for efficient adaptation.

- Configured LoRA with optimized hyperparameters: rank (r=16), alpha scaling (lora_alpha=32), dropout rate (lora_dropout=0.2), and bias settings (bias="none") for sequence classification tasks.

- Conducted training on the intent recognition dataset, evaluated model performance using validation accuracy metrics, and optimized the fine-tuning process for maximum effectiveness.

- Achieved exceptional validation accuracy of 0.981 using the google/gemma-1.1-2b-it model fine-tuned with LoRA configuration, demonstrating highly effective intent classification performance.

- Demonstrated that the combination of advanced language models with parameter-efficient fine-tuning techniques can achieve superior results in intent recognition tasks with significantly reduced computational overhead.

### Key Results

- **Model**: google/gemma-1.1-2b-it
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Final Validation Accuracy**: 0.981
- **Intent Categories**: 7 classes (creative, music, playlist, restaurant, screening, rate_book, weather)

### Technical Configuration

- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.2
- **Bias**: None
- **Task Type**: Sequence Classification
