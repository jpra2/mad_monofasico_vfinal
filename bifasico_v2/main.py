import os
import yaml
import shutil
import sys
import pdb
import numpy as np

__all__= []

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
verif = True
while verif:
    if ler_anterior == False:
        print('voce tem certeza que deseja perder todos os dados de simulacao?')
        z = input('s/n\n')
        if z == 's':
            print('tem certeza absoluta?')
            y = input('s/n\n')
            if y == 's':
                verif = False
            else:
                print('Reinicie a simulacao')
                sys.exit(0)
        else:
            print('Reinicie a simulacao')
            sys.exit(0)

np.save('ler_anterior', np.array([ler_anterior]))

# if ler_anterior == False:
#     resp = input('tem certeza que deseja perder todos os dados: s/n \n')
#     if resp == 's':
#         pass
#     else:
#         print('reinicie a simulacao')
#         sys.exit(0)



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
    # os.chdir(sol_direta)
    # with open('__init__.py', 'w') as f:
    #     pass
    # os.chdir(sol_multi)
    # with open('__init__.py', 'w') as f:
    #     pass

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
    # os.chdir(sol_direta)
    # with open('__init__.py', 'w') as f:
    #     pass
    # os.chdir(sol_multi)
    # with open('__init__.py', 'w') as f:
    #     pass

    if somente_deletar:
        sys.exit(0)

os.chdir(parent_dir)

n = int(data_loaded['loops_antes_de_reiniciar'])
verif = True
cont = 0

cont += 1
os.system('python rodar_bif.py')

if ler_anterior == False:
    ler_anterior = True
    os.chdir(input_dir)
    np.save('ler_anterior', np.array([ler_anterior]))
    os.chdir(parent_dir)

if cont >= n:
    cont = 0
    pdb.set_trace()

while verif:
    cont += 1
    os.system('python rodar_bif.py')

    if cont >= n:
        cont = 0
        pdb.set_trace()


# import processor.run_bifasico

# from processor import run_bifasico



# os.system('python rodar_bif.py')
