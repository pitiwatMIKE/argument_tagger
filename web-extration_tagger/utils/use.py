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

def tag_html_format2(pred_list):
    LIST_TAGS = ['claim', 'premise', 'o']
    REPRESEN_TAGS = ['c', 'p', 'O']
    text_convert = ''
    prev_tag = ''
    trigger_tag = False 


    for word, label in pred_list:
        tags = label.split('-')
        next_tag = tags[0] if len(tags) == 1 else tags[1]
        
        if prev_tag != next_tag:
            if prev_tag:
                text_convert += '</' + html_tag + '>'

            html_tag = LIST_TAGS[REPRESEN_TAGS.index(next_tag)]
            prev_tag = next_tag
            trigger_tag = not(trigger_tag)

            if trigger_tag:
                text_convert += '<' + html_tag + '>'
            else:
                text_convert += '<' + html_tag + '>'

        text_convert += word
    text_convert += '</'+ LIST_TAGS[REPRESEN_TAGS.index(prev_tag)]+'>'
    text_convert = text_convert.replace('<o>', '').replace('</o>', '')
            
    return text_convert

def prepocess_text(text, token=True):
    text = text.replace("\n", "")
    if token == True:
        text = word_tokenize(text)
    return text
