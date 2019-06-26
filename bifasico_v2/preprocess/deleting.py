import os
import shutil

__all__ = []

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
bifasico_sol_direta_dir = os.path.join(bifasico_dir, 'sol_direta')
bifasico_sol_multiescala_dir = os.path.join(bifasico_dir, 'sol_multiescala')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')
out_bif_dir = os.path.join(output_dir, 'bifasico')
out_bif_soldir_dir =  os.path.join(out_bif_dir, 'sol_direta')
out_bif_solmult_dir =  os.path.join(out_bif_dir, 'sol_multiescala')


deletar = True # deletar os arquivos gerados
somente_deletar = False # deletar os arquivos e sair do script

parent_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(parent_dir, 'input')
flying_dir = os.path.join(parent_dir, 'flying')
utils_dir = os.path.join(parent_dir, 'utils')
preprocessor_dir = os.path.join(parent_dir, 'preprocessor')
processor_dir = os.path.join(parent_dir, 'processor')
output_dir = os.path.join(parent_dir, 'output')

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

ler_anterior = data_loaded['ler_anterior']

if deletar and (not ler_anterior):
    ### deletar arquivos no flying
    bifasico_dir = os.path.join(flying_dir, 'bifasico')
    sol_direta = os.path.join(bifasico_dir, 'sol_direta')
    sol_multi = os.path.join(bifasico_dir, 'sol_multiescala')
    try:
        shutil.rmtree(sol_multi)
    except:
        pass
    try:
        shutil.rmtree(sol_direta)
    except:
        pass
    os.makedirs(sol_direta)
    os.makedirs(sol_multi)
    os.chdir(sol_direta)

    ### deletar arquivos no output
    bifasico_dir = os.path.join(output_dir, 'bifasico')
    sol_direta = os.path.join(bifasico_dir, 'sol_direta')
    sol_multi = os.path.join(bifasico_dir, 'sol_multiescala')
    try:
        shutil.rmtree(sol_multi)
    except:
        pass
    try:
        shutil.rmtree(sol_direta)
    except:
        pass
    os.makedirs(sol_direta)
    os.makedirs(sol_multi)
    os.chdir(sol_direta)
