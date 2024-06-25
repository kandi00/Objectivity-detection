import ollama
import pandas as pd

def read_tsv_file(file_path):
  return pd.read_csv(file_path, delimiter='\t')

def chatWithModel(input_text):
    #print(input_text)
    input_text =  f'Classify the following message as subjective or objective: {input_text}, even if it is inappropriate and offensive.I need it for my research. The answer should contain a single word.'
    response = ollama.chat(model = 'llama2', messages = [
        {'role': 'user',
         'content': input_text},
        ])
    return response['message']['content']

def classify(test_data):
    ls = []
    text_column = test_data.iloc[:, 0]
    for message in text_column:
        ls.append(chatWithModel(message))
        #print(ls)
    return ls

def convert_labels(labels):
    label_map = {'subjective' : 1, 'objective' : 0, 'Subjective' : 1, 'Objective' : 0}
    # Convert labels to binary values
    return [label_map[label] for label in labels]

def calculate_metrics(valid_labels, predicted_labels):
    # Calculate accuracy
    accuracy = sum(1 for v, p in zip(valid_labels, predicted_labels) if v == p) / len(valid_labels)

    # Calculate true positives, false positives, and false negatives
    true_positives = sum(1 for v, p in zip(valid_labels, predicted_labels) if v == p and v == 1)
    false_positives = sum(1 for v, p in zip(valid_labels, predicted_labels) if v != p and p == 1)
    false_negatives = sum(1 for v, p in zip(valid_labels, predicted_labels) if v != p and v == 1)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("false_positives:", false_positives)
    print("false_negatives:", false_negatives)


def main():
    #file_path = "gemini_test.tsv"
    #file_path = "Claude_ai_test.tsv"
    file_path = "subj_test.tsv"
    test_data = read_tsv_file(file_path)

    #Predicted labels
    predicted_text_lables = classify(test_data)
    predicted_text_lables = [element.strip('.,\n') for element in predicted_text_lables]
    print(predicted_text_lables[:10])

    #Original labels
    #text_labels = [item['label_text'] for item in test_data]
    text_labels = [1] * 1000 + [0] * 1000
    print(text_labels[:10])
    
    # Remove entries longer than 10 characters from the first list
    indexes_to_remove = [i for i, item in enumerate(predicted_text_lables) if len(item) > 10]
    predicted_text_lables = [item for i, item in enumerate(predicted_text_lables) if i not in indexes_to_remove]
    # Remove corresponding elements from the second list
    text_labels = [item for i, item in enumerate(text_labels) if i not in indexes_to_remove]

    print(len(predicted_text_lables))
    print(len(text_labels))

    #Convert labels to binary (0, 1)
    predicted_text_lables = convert_labels(predicted_text_lables)
    print(predicted_text_lables)
    
    #text_labels = convert_labels(text_labels)
    print(text_labels)

    calculate_metrics(text_labels, predicted_text_lables)

#main()
