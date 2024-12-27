import os 
import re
import json
import markdown


jsonFol ="jsons"
os.makedirs(jsonFol, exist_ok=True)

jsonLinks = []


with open('Readme.md', encoding="utf8") as f:
    text = f.read()


qBlock = text.split('<!-- LoQ -->')[1]
qList = re.findall(r'\[([^\]]+)\]\(cards/([^)]+)\.md\)', qBlock)


for n,(ques, file) in enumerate(qList):
    
    jsonLinks.append({
        "question":ques,
        "file": file
    })

    with open(f"cards/{file}.md", 'r', encoding='utf-8') as f:
        question,answer = f.read().replace('../assets','./assets').split('---')
        #^ For html `assets` path must be on top level
    question = question.replace('###','').strip()

    with open(f"{jsonFol}/{file}.json",'w', encoding="utf-8") as f:
        json.dump({
            "question":markdown.markdown(question),
            "answer": markdown.markdown(answer),
            "prev":qList[n-1][1] if n>0 else '',
            "next":qList[n+1][1] if n<len(qList)-1 else '' 
        }, f, ensure_ascii=False, indent=4)


with open(f'{jsonFol}/full.json','w', encoding="utf-8") as f:
    json.dump(jsonLinks, f, ensure_ascii=False, indent=4)


