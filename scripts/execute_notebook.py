from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'Laboratorio_7_NLP_Embeddings.ipynb'
nb=nbformat.read(path,as_version=4)
client=NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
client.execute()
nbformat.write(nb,path)
errors=[(i,o.get('ename'),o.get('evalue')) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.get('output_type')=='error']
assert not errors,errors
print('Notebook ejecutado:',sum(c.cell_type=='code' for c in nb.cells),'celdas de código, sin errores.',flush=True)
