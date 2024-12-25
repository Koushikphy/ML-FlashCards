from glob import glob
from pathlib import Path
import json

# 1. read the index md
# 2. Stip off the questions and links in an orderly fashion
# 3. Convert the md links to json links
# 4. Create global json
# 5. create local json, each with question, answer, prev json and next json



files = glob('cards/*.md')
output_json_folder = Path("jsons")
mdLinks = ''

jsonLinks = []

# create html files 
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        question,answer = f.read().split('---')
    question = question.replace('###','').strip()
    answer = answer.strip()

    mdLinks += f"- [{question}]({file})\n"

    # create single html file
    
    file = Path(file)
    filePath = output_json_folder/file.with_suffix('.json').name
    jsonLinks.append({
        "question":question,
        "file": filePath.name
    })

    with open(filePath,'w', encoding="utf-8") as f:
        json.dump({
            "question":question,
            "answer": answer
        }, f, ensure_ascii=False, indent=4)

# global markdown
with open('Readme.md','w') as f:
    f.write(f"""
## ML Cards

### List of Questions
{mdLinks}
""")

#global html
with open(f'{output_json_folder}/full.json','w', encoding="utf-8") as f:
    json.dump(jsonLinks, f, ensure_ascii=False, indent=4)




