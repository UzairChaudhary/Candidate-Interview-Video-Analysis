def extractTextFromPdf(doc, nlp):
    text = " "
    for page in doc:
        text = text + str(page.get_text())

    label_list=[]
    text_list = []
    dic = {}

    doc = nlp(text)
    for ent in doc.ents:
        label_list.append(ent.label_)
        text_list.append(ent.text)

    
    for i in range(len(label_list)):
        if label_list[i] in dic:
            # if the key already exists, append the new value to the list of values
            dic[label_list[i]].append(text_list[i])
        else:
            # if the key does not exist, create a new key-value pair
            dic[label_list[i]] = [text_list[i]]
    
    #print(dic)
    return dic