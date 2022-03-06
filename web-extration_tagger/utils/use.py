from pythainlp.tokenize import word_tokenize

def tag_html_format(predict_list, pos=False): # get ist of tuple 1 sentent [(word, pos, tag), .....]
    text_result = ""
    label_start = ""
    start_tag = False
    tag_label = ""
        
    for token in predict_list: # list of tuple
        if pos == True:
            word = token[0]
            tag = token[2]
        else:
            word = token[0]
            tag = token[1]
        
        if tag == "O":
            if start_tag == True :
                label_end = "</claim>" if label_start == "<claim>" else "</premise>"
                text_result += label_end
                text_result += word
                start_tag = False
            else:
                text_result += word
        else:
            if start_tag == False:
                tag_label = tag.split("-")[1]  #I-c  = c 
                label_start = "<claim>" if tag_label == "c" else "<premise>"
                text_result += label_start
                text_result += word
                start_tag = True
            else:
                if tag_label != tag.split("-")[1]: #กรณีที่tag ต่างกันอยู่ติดกัน
                    label_end = "</claim>" if label_start == "<claim>" else "</premise>"
                    text_result += label_end
                    tag_label = tag.split("-")[1]  #I-c  = c 
                    label_start = "<claim>" if tag_label == "c" else "<premise>"
                    text_result += label_start
                    text_result += word
                    start_tag = True
                else:
                    text_result += word
     
    if start_tag == True:
        label_end = "</claim>" if label_start == "<claim>" else "</premise>"
        text_result += label_end
                
    return text_result

def prepocess_text(text, token=True):
    text = text.replace("\n", "")
    if token == True:
        text = word_tokenize(text)
    return text