import os
import numpy as np
import yaml
from pymoab import core, types, rng, topo_util
from preprocess import set_informations as setinfo

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

class Mesh:

    def __init__(self):
        global input_dir
        global flying_dir
        global bifasico_sol_direta_dir
        global bifasico_sol_multiescala_dir
        os.chdir(input_dir)
        with open("inputs.yaml", 'r') as stream:
            data_loaded = yaml.load(stream)
            # data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
            # data_loaded = yaml.full_load(stream)

        input_file = data_loaded['input_file']
        self.input_file = input_file
        ext_h5m_adm = input_file + '_malha_adm.h5m'

        self.mb = core.Core()
        self.mtu = topo_util.MeshTopoUtil(self.mb)
        self.data_loaded = data_loaded
        converter_unidades = data_loaded['converter_unidades']
        ADM = data_loaded['ADM']
        ler_anterior = np.load('ler_anterior.npy')[0]
        self.ler_anterior = ler_anterior
        if ler_anterior == False:
            os.chdir(flying_dir)
            self.mb.load_file(ext_h5m_adm)
            self.ultimo_loop = 0
            self.vpi = 0.0
            self.t = 0.0

        else:
            os.chdir(input_dir)
            ultimo_loop = np.load('ultimo_loop.npy')[0]
            self.ultimo_loop = ultimo_loop
            if ADM == False:
                ext_h5m = input_file + 'sol_direta_' + str(ultimo_loop) + '.h5m'
                os.chdir(bifasico_sol_direta_dir)
                self.mb.load_file(ext_h5m)
            else:
                ext_h5m = input_file + 'sol_direta_' + str(ultimo_loop) + '.h5m'
                os.chdir(bifasico_sol_direta_dir)
                self.mb.load_file(ext_h5m)
            hist = np.load('historico_' + str(ultimo_loop) + '.npy')
            self.vpi = hist[0]
            self.t = hist[1]

        self.all_volumes = self.mb.get_entities_by_dimension(0, 3)
        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)
        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(0, 2)
        self.all_edges = self.mb.get_entities_by_dimension(0, 1)
        self.tags = LoadADMMesh.load_tags(self.mb)
        self.tags.update(LoadADMMesh.create_tags_bifasico(self.mb, ADM, self.all_nodes))

        self.volumes_d = self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([self.tags['P']]), np.array([None]))
        self.volumes_n = self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([self.tags['Q']]), np.array([None]))

        self.mi_w = float(data_loaded['dados_bifasico']['mi_w'])
        self.mi_o = float(data_loaded['dados_bifasico']['mi_o'])
        self.gama_w = float(data_loaded['dados_bifasico']['gama_w'])
        self.gama_o = float(data_loaded['dados_bifasico']['gama_o'])
        self.Sor = float(data_loaded['dados_bifasico']['Sor'])
        self.Swc = float(data_loaded['dados_bifasico']['Swc'])
        self.nw = float(data_loaded['dados_bifasico']['nwater'])
        self.no = float(data_loaded['dados_bifasico']['noil'])
        self.loops = int(data_loaded['dados_bifasico']['loops'])
        self.total_time = float(data_loaded['dados_bifasico']['total_time'])
        self.gravity = data_loaded['gravity']

        if not ler_anterior and converter_unidades:
            setinfo.convert_to_SI(self.mb, self.tags, self.all_volumes, self.all_faces,
                                  self.all_nodes, self.volumes_d, self.volumes_n)

        if ADM:
            self.internos, self.faces, self.arestas, self.vertices, \
            self.wirebasket_elems, self.mv, self.bound_faces_nv, \
            self.wirebasket_numbers, self.meshsets_levels, self.finos0, self.intermediarios, \
            self.L2_meshset, self.faces_adjs_by_dual, self.intern_adjs_by_dual = LoadADMMesh.load_elems_adm(self.mb, self.tags)

        self.wirebasket_elems_nv1 = LoadADMMesh.load_wirebasket_elems_nv1(self.mb, self.tags)

        # if len(self.volumes_n) == 0:
        #     self.wells_injector, self.wells_producer = setinfo.injector_producer_press(self.mb,
        #     self.gama_w, self.gama_o, self.gravity, self.all_nodes, self.volumes_d,
        #     self.tags)
        #
        # else:
        #     self.wells_injector, self.wells_producer = setinfo.injector_producer(self.mb, self.gama_w, self.gama_o, self.gravity, self.all_nodes, self.volumes_d, self.volumes_n, self.tags)

        self.wells_injector, self.wells_producer = setinfo.load_injectors_producers(self.mb,
        self.gama_w, self.gama_o, self.gravity, self.all_nodes, self.volumes_d,
        self.tags)

        self.all_boundary_faces = self.mb.get_entities_by_handle(self.mb.tag_get_data(self.tags['FACES_BOUNDARY'], 0, flat=True)[0])
        self.all_faces_in = rng.subtract(self.all_faces, self.all_boundary_faces)
        self.Adjs = np.array([np.array(self.mb.get_adjacencies(f, 3)) for f in self.all_faces_in])
        self.all_centroids = self.mb.tag_get_data(self.tags['CENT'], self.all_volumes)
        self.all_kharm = self.mb.tag_get_data(self.tags['KHARM'], self.all_faces_in, flat=True)
        self.ADM = ADM
        self.vv = self.mb.create_meshset()
        self.mb.add_entities(self.vv, self.all_volumes)

        if not ler_anterior:
            LoadADMMesh.set_sat_in(self.mb, self.wells_injector, self.all_volumes, self.tags, self.all_centroids)


class LoadADMMesh:

    @staticmethod
    def load_tags(mb):
        list_names_tags = ['d1', 'd2', 'l1_ID', 'l2_ID', 'l3_ID', 'P', 'Q', 'FACES_BOUNDARY',
                           'FACES_BOUNDARY_MESHSETS_LEVEL_2', 'FACES_BOUNDARY_MESHSETS_LEVEL_3',
                           'FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC', 'PRIMAL_ID_1',
                           'PRIMAL_ID_2', 'L2_MESHSET', 'ID_reord_tag', 'CENT', 'AREA',
                           'K_EQ', 'KHARM', 'finos0', 'intermediarios', 'PHI']

        tags = {}
        for name in list_names_tags:
            try:
                tag = mb.tag_get_handle(str(name))
            except:

                print(name, 'Nao existe no arquivo')
                continue
                # sys.exit(0)
                # import pdb; pdb.set_trace()

            tags[name] = tag

        return tags

    @staticmethod
    def load_elems_adm(mb, tags):
        internos = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([0]))
        faces = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([1]))
        arestas = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([2]))
        vertices = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([3]))
        wirebasket_elems_nv1 = np.array(list(internos) + list(faces) + list(arestas) + list(vertices))
        wirebasket_elems = []
        wirebasket_numbers = []
        wirebasket_elems.append([np.array(internos), np.array(faces), np.array(arestas), np.array(vertices)])
        mv = mb.create_meshset()
        mb.add_entities(mv, vertices)

        ni=len(internos)
        nf=len(faces)
        na=len(arestas)
        nv=len(vertices)
        wirebasket_numbers.append([ni, nf, na, nv])

        inte = mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([tags['d2']]), np.array([0]))
        fac = mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([tags['d2']]), np.array([1]))
        are = mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([tags['d2']]), np.array([2]))
        ver = mb.get_entities_by_type_and_tag(mv, types.MBHEX, np.array([tags['d2']]), np.array([3]))
        wirebasket_elems.append([np.array(inte), np.array(fac), np.array(are), np.array(ver)])
        wirebasket_numbers.append([len(inte), len(fac), len(are), len(ver)])

        name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
        boundary_faces_nv2 = mb.get_entities_by_handle(mb.tag_get_data(mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(2)), 0, flat=True)[0])
        boundary_faces_nv3 = mb.get_entities_by_handle(mb.tag_get_data(mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(3)), 0, flat=True)[0])
        bound_faces_nv = [boundary_faces_nv2, boundary_faces_nv3]

        meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags['PRIMAL_ID_1']]), np.array([None]))
        meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags['PRIMAL_ID_2']]), np.array([None]))
        meshsets_levels = [meshsets_nv1, meshsets_nv2]

        finos0 = mb.tag_get_data(tags['finos0'], 0, flat=True)[0]
        finos0 = mb.get_entities_by_handle(finos0)

        intermediarios = mb.get_entities_by_handle(mb.tag_get_data(tags['intermediarios'], 0, flat=True)[0])

        L2_meshset = mb.tag_get_data(mb.tag_get_handle('L2_MESHSET'), 0, flat=True)[0]

        global flying_dir
        os.chdir(flying_dir)
        faces_adjs_by_dual = np.load('faces_adjs_by_dual.npy')
        intern_adjs_by_dual = np.load('intern_adjs_by_dual.npy')
        return internos, faces, arestas, vertices, wirebasket_elems, mv, bound_faces_nv, wirebasket_numbers, meshsets_levels, finos0, intermediarios, L2_meshset, faces_adjs_by_dual, intern_adjs_by_dual

    @staticmethod
    def create_tags_bifasico(mb, ADM, all_nodes):
        tags = {}
        n1 = ['SAT_LAST', 'VOLUME', 'SAT', 'FW', 'LAMB_W', 'LAMB_O', 'LBT', 'MOBI_IN_FACES',
              'FW_IN_FACES', 'TOTAL_FLUX', 'FLUX_W', 'FLUX_IN_FACES', 'S_GRAV',
              'S_GRAV_VOLUME', 'DFDS', 'GAMAV', 'GAMAF', 'VOLUME', 'PF']

        for name in n1:
            tags[name] = mb.tag_get_handle(name, 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        n2 = ['PMS2', 'PF', 'PCORR2']

        if ADM:
            for name in n2:
                tags[name] = mb.tag_get_handle(name, 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        n3 = ['NODES']

        for name in n3:
            tags[name] = mb.tag_get_handle(name, 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        coords = mb.get_coords(all_nodes)
        # coords = coords.reshape([len(all_nodes), 3])
        mb.tag_set_data(tags['NODES'], all_nodes, coords)


        return tags

    @staticmethod
    def set_sat_in(mb, injectors, all_volumes, tags, all_centroids=None):
        mb.tag_set_data(tags['SAT'], all_volumes, np.repeat(0.2, len(all_volumes)))
        mb.tag_set_data(tags['SAT'], injectors, np.repeat(1.0, len(injectors)))
        # lx = 20
        # ly = 10
        # lz = 2
        #
        # nx = 30
        # ny = 30
        # nz = 45
        #
        # lx1 = nx*lx
        # ly1 = ny*ly
        # lz1 = nz*lz
        #
        # b1 = np.array([np.array([0.0, 0.0, 0.0]), np.array([lx1, ly1, lz1/2])])
        # b2 = np.array([np.array([0.0, 0.0, lz1/2]), np.array([lx1, ly1, lz1])])
        #
        # inds0 = np.where(all_centroids[:,0] > b1[0,0])[0]
        # inds1 = np.where(all_centroids[:,1] > b1[0,1])[0]
        # inds2 = np.where(all_centroids[:,2] > b1[0,2])[0]
        # c1 = set(inds0) & set(inds1) & set(inds2)
        # inds0 = np.where(all_centroids[:,0] < b1[1,0])[0]
        # inds1 = np.where(all_centroids[:,1] < b1[1,1])[0]
        # inds2 = np.where(all_centroids[:,2] < b1[1,2])[0]
        # c2 = set(inds0) & set(inds1) & set(inds2)
        # inds_vols1 = np.array(list(c1 & c2))
        #
        # volsb1 = rng.Range(np.array(all_volumes)[inds_vols1])
        #
        # mb.tag_set_data(tags['SAT'], all_volumes, np.repeat(0.8, len(all_volumes)))
        # mb.tag_set_data(tags['SAT'], volsb1, np.repeat(0.2, len(volsb1)))

    @staticmethod
    def load_wirebasket_elems_nv1(mb, tags):
        internos = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([0]))
        faces = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([1]))
        arestas = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([2]))
        vertices = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags['d1']]), np.array([3]))
        wirebasket_elems_nv1 = np.array(list(internos) + list(faces) + list(arestas) + list(vertices))

        return wirebasket_elems_nv1
