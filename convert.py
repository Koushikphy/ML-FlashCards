from glob import glob
from pathlib import Path


files = glob('cards/*.md')
output_html_folder = Path("html_output")
mdLinks, htmlLinks = '', ''


# create html files 
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        question,answer = f.read().split('---')
    question = question.replace('###','').strip()
    answer = answer.strip()

    mdLinks += f"- [{question}]({file})"
    # create single html file
    
    file = Path(file)
    filePath = output_html_folder/file.with_suffix('.html').name
    htmlLinks +=f'<li><a href="{filePath}">{question}</a></li>\n'
    with open(filePath,'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
<title>{file.stem.replace('_',' ').title()}</title>
<script type="text/javascript">window.MathJax = {{tex: {{inlineMath: [['$', '$']],displayMath: [['$$', '$$']]}}}};</script>
<script type="text/javascript" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
</head>
<body>
<h3>{question}</h3>
<p>{answer}</p>
</body>
</html>
""")

# global markdown
with open('Readme.md','w') as f:
    f.write(f"""
## ML Cards

### List of Questions
{mdLinks}
""")


#global html
with open('index.html','w') as f:
    f.write(f"""
<h2> ML Cards</h2>

<h3> List of Questions</h3>
<ul>
{htmlLinks}
</ul>
""")



