import os 
import re
import json
import markdown2


# 1. Read the readme.md between the tage <!-- LoQ --> to read the list of cards.
# 2. Create individual jsons for each of the cards
# 3. Create a full.json with all the topics and json path

jsonFol ="jsons"
os.makedirs(jsonFol, exist_ok=True)

jsonLinks = []

converter = markdown2.Markdown(extras=["tables","code-friendly"])

with open('Readme.md', encoding="utf8") as f:
    text = f.read()


qBlock = text.split('<!-- LoQ -->')[1]
qList = re.findall(r'\[([^\]]+)\]\(cards/([^)]+)\.md\)', qBlock)


#--- print the newly added ones
def newAdded():
    from glob import glob
    cards = glob('cards/*.md')
    
    for file in cards:
        # print(file)
        with open(file, 'r', encoding='utf-8') as f:
            question,_ = f.read().split('---',1)
        question = question.replace('###','').strip()
        tf = re.findall(r'cards/([^)]+)\.md',file)[0]
        if (question, tf) not in qList:
            print(f"- [{question}]({file})")

newAdded()
#---------------


for n,(ques, file) in enumerate(qList):
    
    jsonLinks.append({
        "question":ques,
        "file": file
    })

    with open(f"cards/{file}.md", 'r', encoding='utf-8') as f:
        question,answer = f.read().replace('../assets','./assets').split('---',1)
        #^ For html `assets` path must be on top level
    question = question.replace('###','').strip()

    with open(f"{jsonFol}/{file}.json",'w', encoding="utf-8") as f:
        json.dump({
            "question":converter.convert(question),
            "answer": converter.convert(answer),
            "prev":qList[n-1][1] if n>0 else '',
            "next":qList[n+1][1] if n<len(qList)-1 else '' 
        }, f, ensure_ascii=False, indent=4)


with open(f'{jsonFol}/full.json','w', encoding="utf-8") as f:
    json.dump(jsonLinks, f, ensure_ascii=False, indent=4)


