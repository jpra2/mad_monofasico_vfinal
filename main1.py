import os
import shutil
import yaml

with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

rodar_monofasico = data_loaded['rodar_monofasico']
if rodar_monofasico:

    os.system('python rodar_mono.py')
    input_name = data_loaded['input_name']
    ext_h5m_out_mono = input_name + '_malha_adm.h5m'

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dest = 'bifasico_v2/flying'
    destino = os.path.join(parent_dir, dest)
    fonte = [ext_h5m_out_mono, 'faces_adjs_by_dual.npy', 'intern_adjs_by_dual.npy']

    for name in fonte:
        shutil.copy(name, destino)

rodar_bifasico = data_loaded['rodar_bifasico']

if rodar_bifasico:
    os.system('python rodar_bifasico.py')
