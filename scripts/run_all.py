"""Reproducción con caché. Ejecutar desde la raíz; no reevalúa test sellado."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
def run(script):
    print('\nEjecutando',script,flush=True)
    subprocess.run([sys.executable,script],cwd=ROOT,env=env,check=True)
run('scripts/download_data.py')
if not (ROOT/'results/preprocessing.json').exists():
    subprocess.run([sys.executable,'-m','src.preprocessing'],cwd=ROOT,env=env,check=True)
for script in ['scripts/verify_pipeline.py','scripts/train_embeddings.py','scripts/reference_and_final.py','scripts/train_classifiers.py','scripts/make_figures.py','scripts/write_analysis.py','scripts/make_notebook.py','scripts/make_report.py']:
    run(script)
print('Pipeline completo. El notebook puede ejecutarse sin repetir entrenamientos.',flush=True)
