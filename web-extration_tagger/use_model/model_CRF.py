from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
import sklearn_crfsuite


from utils.use import tag_html_format

path_model = "./trained_model/CRF/"

def doc2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    
    #test add features
    key_claim = ["ดังนั้น","เพราะฉะนั้น","แสดงว่า"]
    key_premise = ["เพราะ", "เพราะว่า", "เนื่องจาก","เพื่อ","เช่น","เหตุผล","คือ"]
    word_claim = word in key_claim
    word_premise = word in key_premise
    # Features from current word
    features={
        'word.word': word,
        'word.isspace':word.isspace(),
        'postag':postag,
        'word.isdigit()': word.isdigit(),
        'woed.claim':word_claim,
        'word.premise':word_premise
    }
    if i > 0:
        prevword = doc[i-1][0]
        postag1 = doc[i-1][1]
        features['word.prevword'] = prevword
        features['word.previsspace']=prevword.isspace()
        features['word.prepostag'] = postag1
        features['word.prevwordisdigit'] = prevword.isdigit()
    else:
        features['BOS'] = True # Special "Beginning of Sequence" tag
    # Features from next word
    if i < len(doc)-1:
        nextword = doc[i+1][0]
        postag1 = doc[i+1][1]
        features['word.nextword'] = nextword
        features['word.nextisspace']=nextword.isspace()
        features['word.nextpostag'] = postag1
        features['word.nextwordisdigit'] = nextword.isdigit()
    else:
        features['EOS'] = True # Special "End of Sequence" tag
    return features

def extract_features(doc):
    return [doc2features(doc, i) for i in range(len(doc))]

def get_ner(text):
    word_cut=word_tokenize(text,engine="newmm")
    list_word=pos_tag(word_cut,engine='perceptron')
    X_test = extract_features([(data,list_word[i][1]) for i,data in enumerate(word_cut)])
    y_=crf.predict_single(X_test)
    return [(word_cut[i],list_word[i][1],data) for i,data in enumerate(y_)]

def predict_argument(text):
    text_preporcess = text.replace("\n"," ")
    text_preporcess = text_preporcess[:-1] if text_preporcess[-1] == " " else text_preporcess
    w_ner = get_ner(text_preporcess)
    return w_ner

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=400,
    all_possible_transitions=True,
    model_filename=path_model+"model_CRF.model0" # ตั้งชื่อโมเดล
)

def call_model_CRF(text):
    return tag_html_format(predict_argument(text), pos=True)