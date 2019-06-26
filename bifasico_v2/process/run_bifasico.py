from process.loading import mesh, bif_elems
import os
import time
import sys
import numpy as np

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
bifasico_sol_direta_dir = os.path.join(bifasico_dir, 'sol_direta')
bifasico_sol_multiescala_dir = os.path.join(bifasico_dir, 'sol_multiescala')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

verif = True
loop = mesh.ultimo_loop
n_impressoes = int(mesh.data_loaded['numero_de_impressoes'])
interv_loops = int(mesh.data_loaded['loops_de_cada_rodada'])
t = mesh.t
bif_elems.vpi = mesh.vpi
contador = 0
cont_imp = 0

ids0 = bif_elems.ids0
ids1 = bif_elems.ids1

loops2 = 0
t2 = 0.0

if mesh.ADM:
    os.chdir(bifasico_sol_multiescala_dir)
    from solucao.sol_adm import SolAdm

    sol = SolAdm(mesh.mb, mesh.wirebasket_elems, mesh.wirebasket_numbers, mesh.tags, mesh.all_volumes, mesh.faces_adjs_by_dual, mesh.intern_adjs_by_dual)
    import pdb; pdb.set_trace()
    pass

elif not mesh.ADM:
    os.chdir(bifasico_sol_direta_dir)
    from solucao.sol_direta import SolDireta

    sol = SolDireta()

    while verif:
        contador += 1

        t0 = time.time()
        bif_elems.get_Tf_and_b()
        bif_elems.Pf = sol.solucao_pressao(bif_elems.Tf2, bif_elems.b2)
        mobi_in_faces = bif_elems.all_mobi_in_faces
        s_grav_f = bif_elems.all_s_gravs
        Pf = bif_elems.Pf
        fw_in_faces = bif_elems.all_fw_in_face
        volumes = mesh.all_volumes
        gravity = mesh.gravity
        bif_elems.fluxos, bif_elems.fluxos_w, bif_elems.flux_in_faces, bif_elems.flux_w_in_faces, bif_elems.fluxo_grav_volumes = sol.calculate_total_flux(ids0, ids1, mobi_in_faces, s_grav_f, Pf, fw_in_faces, volumes, gravity)
        t1 = time.time()
        dt = t1-t0
        bif_elems.set_solutions1()
        bif_elems.calc_cfl()
        bif_elems.verificar_cfl(loop)
        bif_elems.get_hist(t, dt, loop)

        loop += 1
        t += bif_elems.delta_t

        ext_h5m = mesh.input_file + 'sol_direta_' + str(loop) + '.h5m'
        ext_vtk = mesh.input_file + 'sol_direta_' + str(loop) + '.vtk'

        print(f'loop: {loop}')
        print(f'vpi: {bif_elems.vpi}')
        print(f'delta_t: {bif_elems.delta_t}')
        print(f'wor: {bif_elems.hist[4]} \n')

        if verif:
            # bifasico.calculate_sat(all_volumes, loop)
            # bifasico.verificar_cfl(all_volumes, loop)
            bif_elems.set_lamb()
            bif_elems.set_mobi_faces()

        if contador % interv_loops == 0:
            cont_imp += 1
            bif_elems.print_hist(loop)
            mesh.mb.write_file(ext_h5m)
            mesh.mb.write_file(ext_vtk, [mesh.vv])
            with open('tempos_simulacao_direta.txt', 'a+') as fil:
                fil.write(str(dt)+'\n')
            os.chdir(input_dir)
            ultimo_loop = np.array([loop])
            np.save('ultimo_loop', ultimo_loop)
            os.chdir(bifasico_sol_direta_dir)
            if cont_imp >= n_impressoes:
                sys.exit(0)

        import pdb; pdb.set_trace()

import pdb; pdb.set_trace()



import pdb; pdb.set_trace()
print('saiu run bifasico \n')
