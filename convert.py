import os 
import re
import json
import markdown


jsonFol ="jsons"
os.makedirs(jsonFol, exist_ok=True)

jsonLinks = []


with open('Readme.md') as f:
    text = f.read()


qBlock = text.split('<!-- LoQ -->')[1]
qList = re.findall(r'\[([^\]]+)\]\(cards/([^)]+)\.md\)', qBlock)


for n,(ques, file) in enumerate(qList):
    
    jsonLinks.append({
        "question":ques,
        "file": file
    })

    with open(f"cards/{file}.md", 'r', encoding='utf-8') as f:
        question,answer = f.read().split('---')
    question = question.replace('###','').strip()

    with open(f"{jsonFol}/{file}.json",'w', encoding="utf-8") as f:
        json.dump({
            "question":markdown.markdown(question),
            "answer": markdown.markdown(answer),
            "prev":qList[n-1 if n>0 else 0][1] ,
            "next":qList[n+1 if n<len(qList)-1 else n][1] 
        }, f, ensure_ascii=False, indent=4)


with open(f'{jsonFol}/full.json','w', encoding="utf-8") as f:
    json.dump(jsonLinks, f, ensure_ascii=False, indent=4)


