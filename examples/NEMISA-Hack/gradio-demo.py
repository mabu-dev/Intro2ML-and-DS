import gradio as gr
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering, TapasConfig

model_name = 'google/tapas-base-finetuned-wtq'
model = TapasForQuestionAnswering.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)

df_table = pd.read_csv("df_table.csv")
df_table = {c: [str(x) for x in df_table[c].tolist()]
            for c in df_table.columns}
df_table = pd.DataFrame.from_dict(df_table)


def predict(table, queries):
    inputs = tokenizer(table=table, queries=queries,
                       padding='max_length', return_tensors="pt")
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        outputs.logits_aggregation.detach())

    # let's print out the results:
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    aggregation_predictions_string = [id2aggregation[x]
                                      for x in predicted_aggregation_indices]
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(", ".join(cell_values))

    return {"queries": queries, "asnwers": answers, "aggregation": aggregation_predictions_string}


def answer_question(paragraph, question):
    response = predict(table=df_table, queries=question)
    return response["asnwers"]


gr.Interface(fn=answer_question, inputs=[
             "textbox", "text"], outputs="text").launch()
