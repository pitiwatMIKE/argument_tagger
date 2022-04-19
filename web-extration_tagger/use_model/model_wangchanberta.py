from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from pythainlp.tokenize import word_tokenize

model_path = 'pitiwat/argument_wangchanberta2'
tokenizer = AutoTokenizer.from_pretrained(model_path,  model_max_length=512)
model = AutoModelForTokenClassification.from_pretrained(model_path)

pipe = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

def predict(text):
    # word tokenize
    text_token = word_tokenize(text)
    text_token = ' '.join(text_token)
    prediction = pipe(text_token, grouped_entities=True, ignore_labels=[])
    # covert predict to html tag
    text_pred = ""
    for dict_pred in prediction:
        open_tag = f"<{dict_pred['entity_group'].lower()}>"
        close_tag = f"</{dict_pred['entity_group'].lower()}>"
        if open_tag == "<o>":
            continue
        group_word = dict_pred['word']

        if group_word.strip() == "":
            text_pred += group_word
        else:
            group_word = ''.join(group_word.split(" "))
            text_pred += open_tag + group_word + close_tag
    return text_pred

def call_model_wanchanberta(text):
    text_predict = predict(text)
    return text_predict
