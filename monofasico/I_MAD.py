import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
from matplotlib import pyplot as plt
import sympy
import cython
import yaml
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, vstack, hstack, linalg, identity, find

# __all__ = ['M1']
__all__ = []
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
bifasico_dir = os.path.join(parent_parent_dir, 'bifasico_v2')
flying_dir = os.path.join(bifasico_dir, 'flying')

class MeshManager:
    def __init__(self,mesh_file, data_loaded, dim=3):
        self.dimension = dim
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

        self.mb.load_file(mesh_file)
        set_homog = data_loaded['set_homog']

        self.physical_tag = self.mb.tag_get_handle("MATERIAL_SET")
        self.physical_sets = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, np.array(
            (self.physical_tag,)), np.array((None,)))

        self.dirichlet_tag = self.mb.tag_get_handle(
            "Dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.neumann_tag = self.mb.tag_get_handle(
            "Neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        #self.perm_tag = self.mb.tag_get_handle(
        #    "Permeability", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.source_tag = self.mb.tag_get_handle(
            "Source term", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)


        self.all_volumes = self.mb.get_entities_by_dimension(0, self.dimension)

        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)

        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(0, self.dimension-1) #ADJs=np.array([M1.mb.get_adjacencies(face, 3) for face in M1.all_faces])

        self.dirichlet_faces = set()
        self.neumann_faces = set()

        '''self.GLOBAL_ID_tag = self.mb.tag_get_handle(
            "Global_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)'''

        self.create_tags()
        if set_homog:
            self.set_k_homog()
            os.chdir(flying_dir)
            np.save('hs', np.array([1.0, 1.0, 1.0]))
        else:
            self.set_k_and_phi_structured_spe10()
            hs = np.array([6.0959998, 3.0479999, 0.60959998])
            dir_atual = os.getcwd()
            os.chdir(flying_dir)
            np.save('hs', hs)
            os.chdir(dir_atual)
        # self.set_k()
        # self.set_k_homog()
        #self.set_information("PERM", self.all_volumes, 3)
        self.get_boundary_faces()
        self.gravity = False
        self.gama = 10
        self.mi = 1
        # t0=time.time()
        # print('set área')
        # self.get_kequiv_by_face_quad(self.all_faces)
        # '''
        # print('set área')
        # for f in self.all_faces:
        #     self.set_area(f)'''
        # print("took",time.time()-t0)
        self.get_faces_boundary


    def create_tags(self):
        self.perm_tag = self.mb.tag_get_handle("PERM", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.finos_tag = self.mb.tag_get_handle("finos", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_dirichlet_tag = self.mb.tag_get_handle("WELLS_D", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_neumann_tag = self.mb.tag_get_handle("WELLS_N", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.press_value_tag = self.mb.tag_get_handle("P", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.vazao_value_tag = self.mb.tag_get_handle("Q", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.all_faces_boundary_tag = self.mb.tag_get_handle("FACES_BOUNDARY", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.area_tag = self.mb.tag_get_handle("AREA", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.GLOBAL_ID_tag = self.mb.tag_get_handle("G_ID_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.ID_reordenado_tag = self.mb.tag_get_handle("ID_reord_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.phi_tag = self.mb.tag_get_handle("PHI", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.k_eq_tag = self.mb.tag_get_handle("K_EQ", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.kharm_tag = self.mb.tag_get_handle("KHARM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    def create_vertices(self, coords):
        new_vertices = self.mb.create_vertices(coords)
        self.all_nodes.append(new_vertices)
        return new_vertices

    def create_element(self, poly_type, vertices):
        new_volume = self.mb.create_element(poly_type, vertices)
        self.all_volumes.append(new_volume)
        return new_volume

    def set_information(self, information_name, physicals_values,
                        dim_target, set_connect=False):
        information_tag = self.mb.tag_get_handle(information_name)
        for physical_value in physicals_values:
            for a_set in self.physical_sets:
                physical_group = self.mb.tag_get_data(self.physical_tag,
                                                      a_set, flat=True)

                if physical_group == physical:
                    group_elements = self.mb.get_entities_by_dimension(a_set, dim_target)

                    if information_name == 'Dirichlet':
                        # print('DIR GROUP', len(group_elements), group_elements)
                        self.dirichlet_faces = self.dirichlet_faces | set(
                                                    group_elements)

                    if information_name == 'Neumann':
                        # print('NEU GROUP', len(group_elements), group_elements)
                        self.neumann_faces = self.neumann_faces | set(
                                                  group_elements)

                    for element in group_elements:
                        self.mb.tag_set_data(information_tag, element, value)

                        if set_connect:
                            connectivities = self.mtu.get_bridge_adjacencies(
                                                                element, 0, 0)
                            self.mb.tag_set_data(
                                information_tag, connectivities,
                                np.repeat(value, len(connectivities)))

    def set_k(self):
        k = 1.0
        perm_tensor = [k, 0, 0,
                       0, k, 0,
                       0, 0, k]

        k01 = 1.0
        perm01 = [k01, 0, 0,
                  0, k01, 0,
                  0, 0, k01]

        k02 = 0.002
        perm02 = [k02, 0, 0,
                  0, k02, 0,
                  0, 0, k02]

        all_centroids = np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])

        box1 = np.array([np.array([180.0, 0.0, 0.0]), np.array([220.0, 200.0, 90.0])])
        box2 = np.array([np.array([380.0, 100.0, 0.0]), np.array([420.0, 300.0, 90.0])])

        inds0 = np.where(all_centroids[:,0] > box1[0,0])[0]
        inds1 = np.where(all_centroids[:,1] > box1[0,1])[0]
        inds2 = np.where(all_centroids[:,2] > box1[0,2])[0]
        # c1 = set(inds0) & set(inds1) & set(inds2)
        c1 = np.intersect1d(inds0, np.intersect1d(inds1, inds2))
        inds0 = np.where(all_centroids[:,0] < box1[1,0])[0]
        inds1 = np.where(all_centroids[:,1] < box1[1,1])[0]
        inds2 = np.where(all_centroids[:,2] < box1[1,2])[0]
        # c2 = set(inds0) & set(inds1) & set(inds2)
        c2 = np.intersect1d(inds0, np.intersect1d(inds1, inds2))
        # inds_vols = list(c1 & c2)
        inds_vols1 = np.intersect1d(c1, c2)

        inds0 = np.where(all_centroids[:,0] > box2[0,0])[0]
        inds1 = np.where(all_centroids[:,1] > box2[0,1])[0]
        inds2 = np.where(all_centroids[:,2] > box2[0,2])[0]
        # c1 = set(inds0) & set(inds1) & set(inds2)
        c1 = np.intersect1d(inds0, np.intersect1d(inds1, inds2))
        inds0 = np.where(all_centroids[:,0] < box2[1,0])[0]
        inds1 = np.where(all_centroids[:,1] < box2[1,1])[0]
        inds2 = np.where(all_centroids[:,2] < box2[1,2])[0]
        # c2 = set(inds0) & set(inds1) & set(inds2)
        c2 = np.intersect1d(inds0, np.intersect1d(inds1, inds2))
        inds_vols2 = np.intersect1d(c1, c2)

        inds_vols = np.union1d(inds_vols1, inds_vols2)

        vols2 = rng.Range(np.array(self.all_volumes)[inds_vols])
        vols1 = rng.subtract(self.all_volumes, vols2)

        for v in vols1:
            self.mb.tag_set_data(self.perm_tag, v, perm01)
            #v_tags=self.mb.tag_get_tags_on_entity(v)
            #print(self.mb.tag_get_data(v_tags[1],v,flat=True))

        for v in vols2:
            self.mb.tag_set_data(self.perm_tag, v, perm02)

        verif_perm_tag = self.mb.tag_get_handle('verif_perm', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.mb.tag_set_data(verif_perm_tag, vols1, np.repeat(k01, len(vols1)))
        self.mb.tag_set_data(verif_perm_tag, vols2, np.repeat(k02, len(vols2)))
        phi = 0.3
        self.mb.tag_set_data(self.phi_tag, self.all_volumes, np.repeat(phi, len(self.all_volumes)))

        vv = self.mb.create_meshset()
        self.mb.add_entities(vv, self.all_volumes)
        self.mb.write_file('testtt.vtk', [vv])

    def set_k_homog(self):
        k = 1.0
        perm_tensor = [k, 0, 0,
                       0, k*1000, 0,
                       0, 0, k]

        for v in self.all_volumes:
            self.mb.tag_set_data(self.perm_tag, v, perm_tensor)

        self.mb.tag_set_data(self.phi_tag, self.all_volumes, np.repeat(0.3, len(self.all_volumes)))

    def set_area(self, face):
        points = self.mtu.get_bridge_adjacencies(face, 2, 0)
        points = [self.mb.get_coords([vert]) for vert in points]
        if len(points) == 3:
            n1 = np.array(points[0] - points[1])
            n2 = np.array(points[0] - points[2])
            area = np.cross(n1, n2)/2.0

        #calculo da area para quadrilatero regular
        elif len(points) == 4:
            n = np.array([np.array(points[0] - points[1]), np.array(points[0] - points[2]), np.array(points[0] - points[3])])
            norms = np.array(list(map(np.linalg.norm, n)))
            ind_norm_max = np.where(norms == max(norms))[0]
            n = np.delete(n, ind_norm_max, axis = 0)
            area = np.cross(n[0], n[1])
        self.mb.tag_set_data(self.area_tag, face, area)

    def calc_area(self, face):
        points = self.mtu.get_bridge_adjacencies(face, 2, 0)
        points = [self.mb.get_coords([vert]) for vert in points]
        if len(points) == 3:
            n1 = np.array(points[0] - points[1])
            n2 = np.array(points[0] - points[2])
            area = np.cross(n1, n2)/2.0

        #calculo da area para quadrilatero regular
        elif len(points) == 4:
            n = np.array([np.array(points[0] - points[1]), np.array(points[0] - points[2]), np.array(points[0] - points[3])])
            norms = np.array(list(map(np.linalg.norm, n)))
            ind_norm_max = np.where(norms == max(norms))[0]
            n = np.delete(n, ind_norm_max, axis = 0)
            area = np.cross(n[0], n[1])
        return(area)

    def get_kequiv_by_face_quad(self, conj_faces):
        """
        retorna os valores de k equivalente para colocar na matriz
        a partir da face
        input:
            face: face do elemento
        output:
            kequiv: k equivalente
            elems: elementos vizinhos pela face
            s: termo fonte da gravidade
        """
        ADJs=np.array([self.mb.get_adjacencies(face, 3) for face in self.all_faces])

        centroids=np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])

        ADJsv=np.array([self.mb.get_adjacencies(face, 0) for face in self.all_faces])

        ks=self.mb.tag_get_data(self.perm_tag, self.all_volumes)
        #vol_to_pos=dict(zip(M1.all_volumes,range(len(M1.all_volumes))))
        vol_to_pos=dict(zip(self.all_volumes,range(len(self.all_volumes))))
        cont=0
        K_eq=[]
        kharm = []
        for f in self.all_faces:
            adjs=ADJs[cont]
            adjsv=ADJsv[cont]
            cont+=1
            if len(adjs)==2:
                v1=adjs[0]
                v2=adjs[1]
                k1 = ks[vol_to_pos[v1]].reshape([3, 3])
                k2 = ks[vol_to_pos[v2]].reshape([3, 3])
                centroid1 = centroids[vol_to_pos[v1]]
                centroid2 = centroids[vol_to_pos[v2]]
                direction = centroid2 - centroid1
                norm=np.linalg.norm(direction)
                uni = np.absolute(direction/norm)
                k1 = np.dot(np.dot(k1,uni), uni)
                k2 = np.dot(np.dot(k2,uni), uni)
                k_harm = (2*k1*k2)/(k1+k2)

                vertex_cent=np.array([self.mb.get_coords([np.uint64(a)]) for a in adjsv])
                dx=max(vertex_cent[:,0])-min(vertex_cent[:,0])
                dy=max(vertex_cent[:,1])-min(vertex_cent[:,1])
                dz=max(vertex_cent[:,2])-min(vertex_cent[:,2])
                if dx<0.001:
                    dx=1
                if dy<0.001:
                    dy=1
                if dz<0.001:
                    dz=1
                area=dx*dy*dz
                #area = self.mb.tag_get_data(self.area_tag, face, flat=True)[0]
                #s_gr = self.gama*keq*(centroid2[2]-centroid1[2])
                keq = k_harm*area/(self.mi*norm)

                kharm.append(k_harm*area/norm)
                K_eq.append(keq)
            else:
                K_eq.append(0.0)
                kharm.append(0.0)
        self.mb.tag_set_data(self.k_eq_tag, self.all_faces, K_eq)
        self.mb.tag_set_data(self.kharm_tag, self.all_faces, kharm)


    def set_k_and_phi_structured_spe10(self):
        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']

        nx = 60
        ny = 220
        nz = 85
        perms = []
        phis = []

        k = 1.0  #para converter a unidade de permeabilidade
        centroids=np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])
        cont=0
        for v in self.all_volumes:
            centroid = centroids[cont]
            cont+=1
            ijk = np.array([centroid[0]//20.0, centroid[1]//10.0, centroid[2]//2.0])
            e = int(ijk[0] + ijk[1]*nx + ijk[2]*nx*ny)
            # perm = ks[e]*k
            # fi = phi[e]
            perms.append(ks[e]*k)
            phis.append(phi[e])

        self.mb.tag_set_data(self.perm_tag, self.all_volumes, perms)
        self.mb.tag_set_data(self.phi_tag, self.all_volumes, phis)

    def get_boundary_nodes(self):
        all_faces = self.dirichlet_faces | self.neumann_faces
        boundary_nodes = set()
        for face in all_faces:
            nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
            boundary_nodes.update(nodes)
        return boundary_nodes

    def get_faces_boundary(self):
        """
        cria os meshsets
        all_faces_set: todas as faces do dominio
        all_faces_boundary_set: todas as faces no contorno
        """
        all_faces_boundary_set = self.mb.create_meshset()

        for face in self.all_faces:
            size = len(self.mb.get_adjacencies(face, 3))
            self.set_area(face)
            if size < 2:
                self.mb.add_entities(all_faces_boundary_set, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_faces_boundary_set)
        self.all_faces_boundary=self.mb.get_entities_by_handle(all_faces_boundary_set)
    def get_non_boundary_volumes(self, dirichlet_nodes, neumann_nodes):
        volumes = self.all_volumes
        non_boundary_volumes = []
        for volume in volumes:
            volume_nodes = set(self.mtu.get_bridge_adjacencies(volume, 0, 0))
            if (volume_nodes.intersection(dirichlet_nodes | neumann_nodes)) == set():
                non_boundary_volumes.append(volume)
        return non_boundary_volumes

    def set_media_property(self, property_name, physicals_values,
                           dim_target=3, set_nodes=False):

        self.set_information(property_name, physicals_values,
                             dim_target, set_connect=set_nodes)

    def set_boundary_condition(self, boundary_condition, physicals_values,
                               dim_target=3, set_nodes=False):

        self.set_information(boundary_condition, physicals_values,
                             dim_target, set_connect=set_nodes)

    def get_tetra_volume(self, tet_nodes):
        vect_1 = tet_nodes[1] - tet_nodes[0]
        vect_2 = tet_nodes[2] - tet_nodes[0]
        vect_3 = tet_nodes[3] - tet_nodes[0]
        vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3))/6.0
        return vol_eval

    def get_boundary_faces(self):
        all_boundary_faces = self.mb.create_meshset()
        for face in self.all_faces:
            elems = self.mtu.get_bridge_adjacencies(face, 2, 3)
            if len(elems) < 2:
                self.mb.add_entities(all_boundary_faces, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_boundary_faces)
        self.all_boundary_faces=self.mb.get_entities_by_handle(all_boundary_faces)



    @staticmethod
    def imprima(self, text = None):
        m1 = self.mb.create_meshset()
        self.mb.add_entities(m1, self.all_nodes)
        m2 = self.mb.create_meshset()
        self.mb.add_entities(m2, self.all_faces)
        m3 = self.mb.create_meshset()
        self.mb.add_entities(m3, self.all_volumes)
        if text == None:
            text = "output"
        extension = ".vtk"
        text1 = text + "-nodes" + extension
        text2 = text + "-face" + extension
        text3 = text + "-volume" + extension
        self.mb.write_file(text1,[m1])
        self.mb.write_file(text2,[m2])
        self.mb.write_file(text3,[m3])
        print(text, "Arquivos gerados")

def get_box(conjunto, all_centroids, limites, return_inds):
    # conjunto-> lista
    # all_centroids->coordenadas dos centroides do conjunto
    # limites-> diagonal que define os volumes objetivo (numpy array com duas coordenadas)
    # Retorna os volumes pertencentes a conjunto cujo centroide está dentro de limites
    inds0 = np.where(all_centroids[:,0] > limites[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > limites[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > limites[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < limites[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < limites[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < limites[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols = list(c1 & c2)
    if return_inds:
        return (rng.Range(np.array(conjunto)[inds_vols]),inds_vols)
    else:
        return rng.Range(np.array(conjunto)[inds_vols])

#--------------Início dos parâmetros de entrada-------------------
# M1= MeshManager('27x27x27.msh')          # Objeto que armazenará as informações da malha

with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

input_name = data_loaded['input_name']
ext_msh = input_name + '.msh'
M1 = MeshManager(ext_msh, data_loaded)     # Objeto que armazenará as informações da malha
all_volumes=M1.all_volumes
press = 4000.0
vazao = 10000.0
calc_TPFA=True
# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)

M1.all_centroids=np.array([M1.mtu.get_average_position([v]) for v in all_volumes])
cent_tag = M1.mb.tag_get_handle("CENT", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(cent_tag, M1.all_volumes, M1.all_centroids)

all_centroids = M1.all_centroids

cr1 = data_loaded['Crs']['Cr1']
cr2 = data_loaded['Crs']['Cr2']

nx=data_loaded['nx']
ny=data_loaded['ny']
nz=data_loaded['nz']

lx=data_loaded['lx']
ly=data_loaded['ly']
lz=data_loaded['lz']

l1=[cr1[0]*lx,cr1[1]*ly,cr1[2]*lz]
l2=[cr2[0]*lx,cr2[1]*ly,cr2[2]*lz]

x1=nx*lx
y1=ny*ly
z1=nz*lz
# Distância, em relação ao poço, até onde se usa malha fina
r0 = 1
# Distância, em relação ao poço, até onde se usa malha intermediária
r1 = 1
'''
bvd = np.array([np.array([x1-lx, 0.0, 0.0]), np.array([x1, y1, lz])])
bvn = np.array([np.array([0.0, 0.0, z1-lz]), np.array([lx, y1, z1])])
'''

bvd = np.array([np.array([0.0, 0.0, 0.0]), np.array([lx, ly, z1])])
bvn = np.array([np.array([x1-lx, y1-ly, 0.0]), np.array([x1, y1, z1])])
#bvd = np.array([np.array([0.0, 0.0, y2]), np.array([y0, y0, y0])])
#bvn = np.array([np.array([0.0, 0.0, 0.0]), np.array([y0, y0, y1])])

bvfd = np.array([np.array([bvd[0][0]-r0*lx, bvd[0][1]-r0*ly, bvd[0][2]-r0*lz]), np.array([bvd[1][0]+r0*lx, bvd[1][1]+r0*ly, bvd[1][2]+r0*lz])])
bvfn = np.array([np.array([bvn[0][0]-r0*lx, bvn[0][1]-r0*ly, bvn[0][2]-r0*lz]), np.array([bvn[1][0]+r0*lx, bvn[1][1]+r0*ly, bvn[1][2]+r0*lz])])

bvid = np.array([np.array([bvd[0][0]-r1, bvd[0][1]-r1, bvd[0][2]-r1]), np.array([bvd[1][0]+r1, bvd[1][1]+r1, bvd[1][2]+r1])])
bvin = np.array([np.array([bvn[0][0]-r1, bvn[0][1]-r1, bvn[0][2]-r1]), np.array([bvn[1][0]+r1, bvn[1][1]+r1, bvn[1][2]+r1])])

# volumes com pressao prescrita

volumes_d, inds_vols_d= get_box(M1.all_volumes, all_centroids, bvd, True)

# volumes com vazao prescrita
volumes_n, inds_vols_n = get_box(M1.all_volumes, all_centroids, bvn, True)

# volumes finos por neumann
volumes_fn = get_box(M1.all_volumes, all_centroids, bvfn, False)

# volumes finos por Dirichlet
volumes_fd = get_box(M1.all_volumes, all_centroids, bvfd, False)

volumes_f=rng.unite(volumes_fn,volumes_fd)

inds_pocos = inds_vols_d + inds_vols_n
Cent_wels = all_centroids[inds_pocos]

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)


#--------------fim dos parâmetros de entrada------------------------------------
print("")
print("INICIOU PRÉ PROCESSAMENTO")

tempo0_pre=time.time()
def Min_Max(e):
    verts = M1.mb.get_connectivity(e)
    coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
    xmin, xmax = coords[0][0], coords[0][0]
    ymin, ymax = coords[0][1], coords[0][1]
    zmin, zmax = coords[0][2], coords[0][2]
    for c in coords:
        if c[0]>xmax: xmax=c[0]
        if c[0]<xmin: xmin=c[0]
        if c[1]>ymax: ymax=c[1]
        if c[1]<ymin: ymin=c[1]
        if c[2]>zmax: zmax=c[2]
        if c[2]<zmin: zmin=c[2]
    return([xmin,xmax,ymin,ymax,zmin,zmax])

def lu_inv2(M):
    L=M.shape[0]
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        t0=time.time()
        lc=range(L)
        d=np.repeat(1,L)
        B=csc_matrix((d,(lc,lc)),shape=(L,L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            t0=time.time()
            l=range(s*i,s*(i+1))
            B=csc_matrix((d,(l,c)),shape=(L,s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=range(s*n,L)
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(L,r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))
    #print(time.time()-tinv,M.shape[0],"tempo de inversão")
    return inversa

def lu_inv3(M,lines):
    lines=np.array(lines)
    L=len(lines)
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        l=lines
        c=range(len(l))
        d=np.repeat(1,L)
        B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B,'T')).transpose()
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            l=lines[s*i:s*(i+1)]
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B,'T')).transpose()
            else:
                inversa=csc_matrix(vstack([inversa,csc_matrix(LU.solve(B,'T')).transpose()]))

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(vstack([inversa,csc_matrix(LU.solve(B,'T')).transpose()]))
    f=find(inversa)
    ll=f[0]
    c=f[1]
    d=f[2]
    pos_to_line=dict(zip(range(len(lines)),lines))
    lg=[pos_to_line[l] for l in ll]
    inversa=csc_matrix((d,(lg,c)),shape=(M.shape[0],M.shape[0]))
    #print(time.time()-tinv,L,"tempo de inversão")
    return inversa

def lu_inv4(M,lines):
    lines=np.array(lines)
    cols=lines
    L=len(lines)
    s=500
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        l=lines
        c=range(len(l))
        d=np.repeat(1,L)
        B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            l=lines[s*i:s*(i+1)]
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

    tk1=time.time()
    #f=find(inversa.tocsr())
    #l=f[0]
    #cc=f[1]
    #d=f[2]
    #pos_to_col=dict(zip(range(len(cols)),cols))
    #cg=[pos_to_col[c] for c in cc]
    lp=range(len(cols))
    cp=cols
    dp=np.repeat(1,len(cols))
    permut=csc_matrix((dp,(lp,cp)),shape=(len(cols),M.shape[0]))
    inversa=csc_matrix(inversa*permut)

    #inversa1=csc_matrix((d,(l,cg)),shape=(M.shape[0],M.shape[0]))
    #inversa=inversa1
    print(tk1-tinv,L,time.time()-tk1,len(lines),'/',M.shape[0],"tempo de inversão")
    return inversa

def ilu_inv(M,tol,fil):
    L=M.shape[0]
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.spilu(M,tol,fil)
    if L<s:
        t0=time.time()
        lc=range(L)
        d=np.repeat(1,L)
        B=csc_matrix((d,(lc,lc)),shape=(L,L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            t0=time.time()
            l=range(s*i,s*(i+1))
            B=csc_matrix((d,(l,c)),shape=(L,s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=range(s*n,L)
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(L,r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))
    #print(time.time()-tinv,M.shape[0],"tempo de inversão")
    return inversa

all_volumes=M1.all_volumes
print("Volumes:",all_volumes)
xmin, xmax, ymin, ymax, zmin, zmax=Min_Max(all_volumes[0])
dx0, dy0, dz0 = xmax-xmin, ymax-ymin, zmax-zmin # Tamanho de cada elemento na malha fina
#-------------------------------------------------------------------------------
print("definiu dimensões")
finos=[]
pocos_meshset=M1.mb.create_meshset()

M1.mb.tag_set_data(M1.finos_tag, 0,pocos_meshset)
finos=list(rng.unite(rng.unite(volumes_d,volumes_n),volumes_f))

print("definiu volumes na malha fina")

pocos=M1.mb.get_entities_by_handle(pocos_meshset)

finos_meshset = M1.mb.create_meshset()

print("definiu poços")
#-------------------------------------------------------------------------------
#Determinação das dimensões do reservatório e dos volumes das malhas intermediária e grossa
for v in M1.all_nodes:       # M1.all_nodes -> todos os vértices da malha fina
    c=M1.mb.get_coords([v])  # Coordenadas de um nó
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]

Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin  # Dimensões do reservatório
# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
# O valor 0.01 é adicionado para corrigir erros de ponto flutuante
for i in range(int(Lx/l2[0])):    lx2.append(xmin+i*l2[0])
for i in range(int(Ly/l2[1])):    ly2.append(ymin+i*l2[1])
for i in range(int(Lz/l2[2])):    lz2.append(zmin+i*l2[2])
lx2.append(Lx)
ly2.append(Ly)
lz2.append(Lz)
#-------------------------------------------------------------------------------
dirichlet_meshset = M1.mb.create_meshset()
neumann_meshset = M1.mb.create_meshset()

if M1.gravity == False:
    pressao = np.repeat(press, len(volumes_d))

# # colocar gravidade
elif M1.gravity == True:
    pressao = []
    z_elems_d = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes_d])
    delta_z = z_elems_d + Lz
    pressao = M1.gama*(delta_z) + press
###############################################
else:
    print("Defina se existe gravidade (True) ou nao (False)")

M1.mb.add_entities(dirichlet_meshset, volumes_d)
M1.mb.add_entities(neumann_meshset, volumes_n)
M1.mb.add_entities(finos_meshset, rng.unite(rng.unite(volumes_n, volumes_d),volumes_f))

#########################################################################################
#jp: modifiquei as tags para sparse
neumann=M1.mb.tag_get_handle("neumann", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
dirichlet=M1.mb.tag_get_handle("dirichlet", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
###############################################################################################

M1.mb.tag_set_data(neumann, volumes_n, np.repeat(1, len(volumes_n)))
M1.mb.tag_set_data(dirichlet, volumes_d, np.repeat(1, len(volumes_d)))

M1.mb.tag_set_data(M1.wells_neumann_tag, 0, neumann_meshset)
M1.mb.tag_set_data(M1.wells_dirichlet_tag, 0, dirichlet_meshset)
M1.mb.tag_set_data(M1.finos_tag, 0, finos_meshset)
M1.mb.tag_set_data(M1.press_value_tag, volumes_d, pressao)
M1.mb.tag_set_data(M1.press_value_tag, volumes_n, np.repeat(vazao, len(volumes_n)))
#-------------------------------------------------------------------------------
# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(l2[0]/l1[0])):   lx1.append(i*l1[0])
for i in range(int(l2[1]/l1[1])):   ly1.append(i*l1[1])
for i in range(int(l2[2]/l1[2])):   lz1.append(i*l1[2])


D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
nD_x=int((D_x+0.001)/l1[0])
nD_y=int((D_y+0.001)/l1[1])
nD_z=int((D_z+0.001)/l1[2])


lxd1=[xmin+dx0/100]
for i in range(int(Lx/l1[0])-2-nD_x):
    lxd1.append(l1[0]/2+(i+1)*l1[0])
lxd1.append(xmin+Lx-dx0/100)

lyd1=[ymin+dy0/100]
for i in range(int(Ly/l1[1])-2-nD_y):
    lyd1.append(l1[1]/2+(i+1)*l1[1])
lyd1.append(ymin+Ly-dy0/100)

lzd1=[zmin+dz0/100]

for i in range(int(Lz/l1[2])-2-nD_z):
    lzd1.append(l1[2]/2+(i+1)*l1[2])
lzd1.append(xmin+Lz-dz0/100)

#lzd1[-2]=21.5

#lzd1[0]=1.5
#lzd1[-2]=23.5
#lzd1[-3]=20.5
print("definiu planos do nível 1")
lxd2=[lxd1[0]]
for i in range(1,int(len(lxd1)*l1[0]/l2[0])-1):
    lxd2.append(lxd1[int(i*l2[0]/l1[0]+0.0001)+1])
lxd2.append(lxd1[-1])

lyd2=[lyd1[0]]
for i in range(1,int(len(lyd1)*l1[1]/l2[1])-1):
    lyd2.append(lyd1[int(i*l2[1]/l1[1]+0.00001)+1])
lyd2.append(lyd1[-1])

lzd2=[lzd1[0]]
for i in range(1,int(len(lzd1)*l1[2]/l2[2])-1):
    lzd2.append(lzd1[int(i*l2[2]/l1[2]+0.00001)+1])
lzd2.append(lzd1[-1])

print("definiu planos do nível 2")


node=M1.all_nodes[0]
coords=M1.mb.get_coords([node])
min_dist_x=coords[0]
min_dist_y=coords[1]
min_dist_z=coords[2]
t0=time.time()
# ---- Criação e preenchimento da árvore de meshsets----------------------------
# Esse bloco é executado apenas uma vez em um problema bifásico, sua eficiência
# não é criticamente importante.
L2_meshset=M1.mb.create_meshset()       # root Meshset
l2_meshset_tag = M1.mb.tag_get_handle("L2_MESHSET", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
M1.mb.tag_set_data(l2_meshset_tag, 0, L2_meshset)
###########################################################################################
#jp:modifiquei as tags abaixo para sparse
D1_tag=M1.mb.tag_get_handle("d1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
D2_tag=M1.mb.tag_get_handle("d2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################
fine_to_primal1_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
fine_to_primal2_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
AV_meshset=M1.mb.create_meshset()
primal_id_tag1 = M1.mb.tag_get_handle("PRIMAL_ID_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
primal_id_tag2 = M1.mb.tag_get_handle("PRIMAL_ID_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
nc1=0
nc2=0
D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
centroids=M1.all_centroids
sx=0
ref_dual=False
M1.mb.add_entities(AV_meshset,all_volumes)
for i in range(len(lx2)-1):
    t1=time.time()
    if i==len(lx2)-2:
        sx=D_x
    sy=0

    #################################################
    x0=lx2[i]
    x1=lx2[i+1]
    box_x=np.array([[x0-0.01,ymin,zmin],[x1+0.01,ymax,zmax]])
    vols_x=get_box(M1.all_volumes, centroids, box_x, False)
    x_centroids=np.array([M1.mtu.get_average_position([v]) for v in vols_x])
    ######################################

    for j in range(len(ly2)-1):
        if j==len(ly2)-2:
            sy=D_y
        sz=0
        #########################
        y0=ly2[j]
        y1=ly2[j+1]
        box_y=np.array([[x0-0.01,y0-0.01,zmin],[x1+0.01,y1+0.01,zmax]])
        vols_y=get_box(vols_x, x_centroids, box_y, False)
        y_centroids=np.array([M1.mtu.get_average_position([v]) for v in vols_y])
        ###############
        for k in range(len(lz2)-1):
            if k==len(lz2)-2:
                sz=D_z
            ########################################
            z0=lz2[k]
            z1=lz2[k+1]
            tb=time.time()
            box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
            vols=get_box(vols_y, y_centroids, box_dual_1, False)
            ####################
            l2_meshset=M1.mb.create_meshset()
            cont=0
            elem_por_L2=vols
            M1.mb.add_entities(l2_meshset,elem_por_L2)
            centroid_p2=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
            cx,cy,cz=centroid_p2[:,0],centroid_p2[:,1],centroid_p2[:,2]
            posx=np.where(abs(cx-lxd2[i])<=l1[0]/1.9)[0]
            posy=np.where(abs(cy-lyd2[j])<=l1[1]/1.9)[0]
            posz=np.where(abs(cz-lzd2[k])<=l1[2]/1.9)[0]
            f1a2v3=np.zeros(len(elem_por_L2),dtype=int)
            f1a2v3[posx]+=1
            f1a2v3[posy]+=1
            f1a2v3[posz]+=1
            M1.mb.tag_set_data(D2_tag, elem_por_L2, f1a2v3)
            M1.mb.tag_set_data(fine_to_primal2_classic_tag, elem_por_L2, np.repeat(nc2,len(elem_por_L2)))
            M1.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg=M1.mb.get_entities_by_handle(l2_meshset)
            print(k, len(sg), time.time()-t1)
            t1=time.time()
            M1.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)
            centroids_primal2=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
            nc2+=1
            s1x=0
            for m in range(len(lx1)):
                a=int(l2[0]/l1[0])*i+m
                if Lx-D_x==lx2[i]+lx1[m]+l1[0]:# and D_x==Lx-int(Lx/l1[0])*l1[0]:
                    s1x=D_x
                s1y=0
                for n in range(len(ly1)):
                    b=int(l2[1]/l1[1])*j+n
                    if Ly-D_y==ly2[j]+ly1[n]+l1[1]:# and D_y==Ly-int(Ly/l1[1])*l1[1]:
                        s1y=D_y
                    s1z=0

                    for o in range(len(lz1)):
                        c=int(l2[2]/l1[2])*k+o
                        if Lz-D_z==lz2[k]+lz1[o]+l1[2]:
                            s1z=D_z
                        l1_meshset=M1.mb.create_meshset()
                        box_primal1 = np.array([np.array([lx2[i]+lx1[m], ly2[j]+ly1[n], lz2[k]+lz1[o]]), np.array([lx2[i]+lx1[m]+l1[0]+s1x, ly2[j]+ly1[n]+l1[1]+s1y, lz2[k]+lz1[o]+l1[2]+s1z])])
                        elem_por_L1 = get_box(elem_por_L2, centroids_primal2, box_primal1, False)
                        M1.mb.add_entities(l1_meshset,elem_por_L1)
                        cont1=0
                        values_1=[]
                        for e in elem_por_L1:
                            cont1+=1
                            f1a2v3=0
                            M_M=Min_Max(e)
                            if (M_M[0]<lxd1[a] and M_M[1]>=lxd1[a]):
                                f1a2v3+=1
                            if (M_M[2]<lyd1[b] and M_M[3]>=lyd1[b]):
                                f1a2v3+=1
                            if (M_M[4]<lzd1[c] and M_M[5]>=lzd1[c]):
                                f1a2v3+=1
                            values_1.append(f1a2v3)
                        M1.mb.tag_set_data(D1_tag, elem_por_L1,values_1)
                        M1.mb.tag_set_data(fine_to_primal1_classic_tag, elem_por_L1, np.repeat(nc1,len(elem_por_L1)))
                        M1.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                        nc1+=1
                        M1.mb.add_child_meshset(l2_meshset,l1_meshset)
#-------------------------------------------------------------------------------

print('Criação da árvore de meshsets primais: ',time.time()-t0)
ta=time.time()
all_volumes=M1.all_volumes
vert_meshset=M1.mb.create_meshset()

#################################################################################################
# setando faces de contorno
parent_dir = os.path.dirname(os.path.abspath(__file__))
adm_mod_dir = os.path.join(parent_dir, 'ADM_mod_2-master')
parent_parent_dir = adm_mod_dir
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')

import importlib.machinery
# loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
# utpy = loader.load_module('pymoab_utils')
from utils import pymoab_utils as utpy

meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))


n_levels = 2
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
all_meshsets = [meshsets_nv1, meshsets_nv2]
t0 = time.time()

for i in range(n_levels):
    meshsets = all_meshsets[i]
    # names_tags_criadas_aqui.append(name_tag_faces_boundary_meshsets + str(i+2))
    tag_boundary = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets + str(i+2), 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
    utpy.set_faces_in_boundary_by_meshsets(M1.mb, M1.mtu, meshsets, tag_boundary)
t1 = time.time()
print('tempo faces contorno')
print(t1-t0)
all_vertex_d1=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
all_vertex_d1=np.uint64(np.array(rng.unite(all_vertex_d1,M1.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))))
mm=0

vertex_centroids=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in all_vertex_d1])
vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices=rng.unite(vertices,M1.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))
all_vertex_centroids=np.array([M1.mtu.get_average_position([v]) for v in vertices])

# volumes intermediarios por neumann
volumes_in = get_box(vertices, all_vertex_centroids, bvin, False)

# volumes intermediarios por Dirichlet
volumes_id = get_box(vertices, all_vertex_centroids, bvid, False)
intermediarios=rng.unite(volumes_id,volumes_in)
# Tag que armazena o ID do volume no nível 1
# jp: modifiquei as tags abaixo para o tipo sparse
L1_ID_tag=M1.mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# L1ID_tag=M1.mb.tag_get_handle("l1ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# Tag que armazena o ID do volume no nível 2
L2_ID_tag=M1.mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# L2ID_tag=M1.mb.tag_get_handle("l2ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# ni = ID do elemento no nível i
L3_ID_tag=M1.mb.tag_get_handle("l3_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################
# ni = ID do elemento no nível i
# volumes da malha grossa primal 1
meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))

# volumes da malha grossa primal 2
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))


for meshset in meshsets_nv2:
    nc = M1.mb.tag_get_data(primal_id_tag2, meshset, flat=True)[0]
    elems = M1.mb.get_entities_by_handle(meshset)
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))


# Gera a matriz dos coeficientes

internos=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))

M1.mb.tag_set_data(fine_to_primal1_classic_tag,vertices,np.arange(0,len(vertices)))

ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)

nni=ni
nnf=nni+nf
nne=nnf+na
nnv=nne+nv
l_elems=[internos,faces,arestas,vertices]
l_ids=[0,nni,nnf,nne,nnv]
for i, elems in enumerate(l_elems):
    M1.mb.tag_set_data(M1.ID_reordenado_tag,elems,np.arange(l_ids[i],l_ids[i+1]))

GID_tag=M1.mb.tag_get_handle("GID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(GID_tag,M1.all_volumes,range(len(M1.all_volumes)))
def set_kequiv(self,conj_faces,adjs):
    IDs1=M1.mb.tag_get_data(GID_tag,np.array(adjs[:,0]),flat=True)
    IDs2=M1.mb.tag_get_data(GID_tag,np.array(adjs[:,1]),flat=True)
    centroids=M1.all_centroids
    centroid1=centroids[IDs1]
    centroid2=centroids[IDs2]
    direction=(centroid2-centroid1)

    ADJsv=np.uint64([M1.mb.get_adjacencies(face, 0) for face in conj_faces])
    v0=ADJsv[:,0]
    v1=ADJsv[:,1]
    v2=ADJsv[:,2]
    v3=ADJsv[:,3]

    c0=M1.mb.get_coords(np.array(v0)).reshape(len(v0),3)
    c1=M1.mb.get_coords(np.array(v1)).reshape(len(v1),3)
    c2=M1.mb.get_coords(np.array(v2)).reshape(len(v2),3)
    c3=M1.mb.get_coords(np.array(v3)).reshape(len(v3),3)

    x=np.array([c0[:,0],c1[:,0],c2[:,0],c3[:,0]]).transpose()
    y=np.array([c0[:,1],c1[:,1],c2[:,1],c3[:,1]]).transpose()
    z=np.array([c0[:,2],c1[:,2],c2[:,2],c3[:,2]]).transpose()

    xi, yi, zi = x.min(axis=1), y.min(axis=1), z.min(axis=1)
    xs, ys, zs = x.max(axis=1), y.max(axis=1), z.max(axis=1)
    dx, dy, dz = xs-xi, ys-yi, zs-zi
    deltas=np.array([dx,dy,dz]).transpose()
    ord=np.sort(deltas,axis=1)
    area=ord[:,1]*ord[:,2]

    ks=self.mb.tag_get_data(self.perm_tag, self.all_volumes)
    k1 = ks[:,[0,4,8]][IDs1]
    k2 = ks[:,[0,4,8]][IDs2]

    norm_direction=np.linalg.norm(direction,axis=1)
    K1 = (k1*abs(direction)).max(axis=1)/norm_direction
    K2 = (k2*abs(direction)).max(axis=1)/norm_direction
    Kharm=2*K1*K2/(K1+K2)

    # Keq=Kharm*area/norm_direction
    Keq=Kharm
    # Keq=np.ones(len(Kharm))
    other_faces=np.setdiff1d(np.uint64(M1.all_faces),conj_faces)
    M1.mb.tag_set_data(self.k_eq_tag, conj_faces, Keq)
    M1.mb.tag_set_data(self.kharm_tag, conj_faces, Keq)

    M1.mb.tag_set_data(self.k_eq_tag, other_faces, np.zeros(len(other_faces)))
    M1.mb.tag_set_data(self.kharm_tag, other_faces, np.zeros(len(other_faces)))

    #self.mb.tag_set_data(self.kharm_tag, self.all_faces, kharm)


################################################################################
all_intern_faces=[face for face in M1.all_faces if len(M1.mb.get_adjacencies(face, 3))==2]
all_intern_adjacencies=np.array([M1.mb.get_adjacencies(face, 3) for face in all_intern_faces])
all_adjacent_volumes=[]
all_adjacent_volumes.append(M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(all_intern_adjacencies[:,0]),flat=True))
all_adjacent_volumes.append(M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(all_intern_adjacencies[:,1]),flat=True))

t1=time.time()
set_kequiv(M1,all_intern_faces,all_intern_adjacencies)
print("setou k",time.time()-t1)

kst=M1.mb.tag_get_data(M1.k_eq_tag,all_intern_faces,flat=True)

################################################################################
inter_teste=np.uint64(all_intern_faces)
def add_topology(conj_vols,tag_local,lista):
    all_fac=np.uint64(M1.mtu.get_bridge_adjacencies(conj_vols, 2 ,2))
    all_int_fac=np.uint64([face for face in all_fac if len(M1.mb.get_adjacencies(face, 3))==2])
    adjs=np.array([M1.mb.get_adjacencies(face, 3) for face in all_int_fac])
    adjs1=M1.mb.tag_get_data(tag_local,np.array(adjs[:,0]),flat=True)
    adjs2=M1.mb.tag_get_data(tag_local,np.array(adjs[:,1]),flat=True)
    adjsg1=M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(adjs[:,0]),flat=True)
    adjsg2=M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(adjs[:,1]),flat=True)
    Gids=M1.mb.tag_get_data(M1.ID_reordenado_tag,conj_vols,flat=True)
    lista.append(Gids)
    lista.append(all_int_fac)
    lista.append(adjs1)
    lista.append(adjs2)
    lista.append(adjsg1)
    lista.append(adjsg2)


local_id_int_tag = M1.mb.tag_get_handle("local_id_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
local_id_fac_tag = M1.mb.tag_get_handle("local_fac_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(local_id_int_tag, M1.all_volumes,np.repeat(len(M1.all_volumes)+1,len(M1.all_volumes)))
M1.mb.tag_set_data(local_id_fac_tag, M1.all_volumes,np.repeat(len(M1.all_volumes)+1,len(M1.all_volumes)))
sgids=0
li=[]
ci=[]
di=[]
cont=0
intern_adjs_by_dual=[]
faces_adjs_by_dual=[]
dual_1_meshset=M1.mb.create_meshset()

D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
for i in range(len(lxd1)-1):
    x0=lxd1[i]
    x1=lxd1[i+1]
    box_x=np.array([[x0-0.01,ymin,zmin],[x1+0.01,ymax,zmax]])
    vols_x=get_box(M1.all_volumes, all_centroids, box_x, False)
    x_centroids=np.array([M1.mtu.get_average_position([v]) for v in vols_x])
    for j in range(len(lyd1)-1):
        y0=lyd1[j]
        y1=lyd1[j+1]
        box_y=np.array([[x0-0.01,y0-0.01,zmin],[x1+0.01,y1+0.01,zmax]])
        vols_y=get_box(vols_x, x_centroids, box_y, False)
        y_centroids=np.array([M1.mtu.get_average_position([v]) for v in vols_y])
        for k in range(len(lzd1)-1):
            z0=lzd1[k]
            z1=lzd1[k+1]
            tb=time.time()
            box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
            vols=get_box(vols_y, y_centroids, box_dual_1, False)
            tipo=M1.mb.tag_get_data(D1_tag,vols,flat=True)
            inter=rng.Range(np.array(vols)[np.where(tipo==0)[0]])

            M1.mb.tag_set_data(local_id_int_tag,inter,range(len(inter)))
            add_topology(inter,local_id_int_tag,intern_adjs_by_dual)


            fac=rng.Range(np.array(vols)[np.where(tipo==1)[0]])
            fac_centroids=np.array([M1.mtu.get_average_position([f]) for f in fac])

            box_faces_x=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x0+lx/2,y1+ly/2,z1+lz/2]])
            box_faces_y=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y0+ly/2,z1+lz/2]])
            box_faces_z=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z0+lz/2]])

            faces_x=get_box(fac, fac_centroids, box_faces_x, False)

            faces_y=get_box(fac, fac_centroids, box_faces_y, False)
            f1=rng.unite(faces_x,faces_y)

            faces_z=get_box(fac, fac_centroids, box_faces_z, False)
            f1=rng.unite(f1,faces_z)

            if i==len(lxd1)-2:
                box_faces_x2=np.array([[x1-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_x2=get_box(fac, fac_centroids, box_faces_x2, False)
                f1=rng.unite(f1,faces_x2)

            if j==len(lyd1)-2:
                box_faces_y2=np.array([[x0-lx/2,y1-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_y2=get_box(fac, fac_centroids, box_faces_y2, False)
                f1=rng.unite(f1,faces_y2)

            if k==len(lzd1)-2:
                box_faces_z2=np.array([[x0-lx/2,y0-ly/2,z1-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_z2=get_box(fac, fac_centroids, box_faces_z2, False)
                f1=rng.unite(f1,faces_z2)

            sgids+=len(f1)
            M1.mb.tag_set_data(local_id_fac_tag,f1,range(len(f1)))
            add_topology(f1,local_id_fac_tag,faces_adjs_by_dual)

print(time.time()-t1,"criou meshset")


#meshsets_duais=M1.mb.get_child_meshsets(dual_1_meshset)

def solve_block_matrix(topology,pos_0):
    lgp=[]
    cgp=[]
    dgp=[]
    c0=0

    st=0
    ts=0
    ta=0
    tc=0

    fl=[]
    fc=[]
    fd=[]
    t_invaii=time.time()
    for cont in range(int(len(topology)/6)):
        t1=time.time()
        Gids=topology[6*cont]
        all_faces_topo=topology[6*cont+1]
        ADJs1=topology[6*cont+2]
        ADJs2=topology[6*cont+3]
        if pos_0 > 0:
            adjsg1=topology[6*cont+4]
            adjsg2=topology[6*cont+5]
            inds1=np.where(adjsg1<pos_0)[0]
            inds2=np.where(adjsg2<pos_0)[0]
            inds_elim=np.unique(np.concatenate([inds1,inds2]))
            all_faces_topo=np.delete(all_faces_topo,inds_elim)
            ADJs1=np.delete(ADJs1,inds_elim)
            ADJs2=np.delete(ADJs2,inds_elim)
        ks_all=np.array(M1.mb.tag_get_data(M1.k_eq_tag,np.array(all_faces_topo),flat=True))
        ts+=time.time()-t1
        t2=time.time()
        int1=np.where(ADJs1<len(Gids))
        int2=np.where(ADJs2<len(Gids))
        pos_int_i=np.intersect1d(int1,int2)
        pos_int_e1=np.setdiff1d(int1,pos_int_i)
        pos_int_e2=np.setdiff1d(int2,pos_int_i)

        Lid_1=ADJs1[pos_int_i]
        Lid_2=ADJs2[pos_int_i]
        ks=ks_all[pos_int_i]

        lines1=[]
        cols1=[]
        data1=[]

        lines1.append(Lid_1)
        cols1.append(Lid_2)
        data1.append(ks)

        lines1.append(Lid_2)
        cols1.append(Lid_1)
        data1.append(ks)

        lines1.append(Lid_1)
        cols1.append(Lid_1)
        data1.append(-ks)

        lines1.append(Lid_2)
        cols1.append(Lid_2)
        data1.append(-ks)

        Lid_1=ADJs1[pos_int_e1]
        ks=ks_all[pos_int_e1]
        lines1.append(Lid_1)
        cols1.append(Lid_1)
        data1.append(-ks)

        Lid_2=ADJs2[pos_int_e2]
        ks=ks_all[pos_int_e2]
        lines1.append(Lid_2)
        cols1.append(Lid_2)
        data1.append(-ks)


        lines1=np.concatenate(np.array(lines1))
        cols1=np.concatenate(np.array(cols1))
        data1=np.concatenate(np.array(data1))
        M_local=csc_matrix((data1,(lines1,cols1)),shape=(len(Gids),len(Gids)))
        ta+=time.time()-t2
        tinvert=time.time()
        try:
            inv_local=lu_inv2(M_local)
        except:
            import pdb; pdb.set_trace()

        st+=time.time()-tinvert

        t3=time.time()
        ml=find(inv_local)
        fl.append(ml[0]+c0)
        fc.append(ml[1]+c0)
        fd.append(ml[2])
        lgp.append(Gids-pos_0)
        tc+=time.time()-t3
        c0+=len(Gids)
    return(lgp,cgp,dgp,fl,fc,fd,t_invaii,st,ts,ta,tc)

def get_invAii():
    t_invaii=time.time()
    lgp,cgp,dgp,fl,fc,fd,t_invaii,st,ts,ta,tc=solve_block_matrix(intern_adjs_by_dual,0)
    fl=np.concatenate(np.array(fl))
    fc=np.concatenate(np.array(fc))
    fd=np.concatenate(np.array(fd))

    m_loc=csc_matrix((fd,(fl,fc)),shape=(ni,ni))
    lgp=np.concatenate(np.array(lgp))
    cgp=range(ni)
    dgp=np.ones(len(lgp))
    permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(ni,ni))
    invbAii=permut_g*m_loc*permut_g.transpose()
    print("inversão de Aii",time.time()-t_invaii,st,ts,ta,tc)
    return invbAii

def get_invAff():
    t_invaff=time.time()
    lgp,cgp,dgp,fl,fc,fd,t_invaii,st,ts,ta,tc=solve_block_matrix(faces_adjs_by_dual,ni)
    fl=np.concatenate(np.array(fl))
    fc=np.concatenate(np.array(fc))
    fd=np.concatenate(np.array(fd))

    m_loc=csc_matrix((fd,(fl,fc)),shape=(nf,nf))
    lgp=np.concatenate(np.array(lgp))
    cgp=range(nf)
    dgp=np.ones(len(lgp))
    permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(nf,nf))
    invbAff=permut_g*m_loc*permut_g.transpose()
    print("inversão de Aff",time.time()-t_invaff,st,ts,ta,tc)
    return invbAff

t0=time.time()
for meshset in meshsets_nv1:
    elems = M1.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, vertices)
    try:
        nc = M1.mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    except:
        import pdb; pdb.set_trace()
    M1.mb.tag_set_data(fine_to_primal1_classic_tag, elems, np.repeat(nc, len(elems)))
    M1.mb.tag_set_data(primal_id_tag1, meshset, nc)

#############################################
## malha adm 1

################################################################################
tempo0_ADM=time.time()
lines=[]
data=[]
cols=[]

lii=[]
lif=[]
lff=[]
lfe=[]
lee=[]
lev=[]
lvv=[]

cii=[]
cif=[]
cff=[]
cfe=[]
cee=[]
cev=[]
cvv=[]

dii=[]
dif=[]
dff=[]
dfe=[]
dee=[]
dev=[]
dvv=[]


#ADJs=np.array([M1.mb.get_adjacencies(face, 3) for face in M1.all_faces])
ADJs1=all_adjacent_volumes[0]
ADJs2=all_adjacent_volumes[1]
ks=M1.mb.tag_get_data(M1.k_eq_tag,all_intern_faces,flat=True)
c2=0
cont=0
for f in all_intern_faces:
    k_eq=ks[cont]
    Gid_1=ADJs1[c2]
    Gid_2=ADJs2[c2]
    c2+=1
    if Gid_1<ni and Gid_2<ni:
        lii.append(Gid_1)
        cii.append(Gid_2)
        dii.append(k_eq)

        lii.append(Gid_2)
        cii.append(Gid_1)
        dii.append(k_eq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-k_eq)

        lii.append(Gid_2)
        cii.append(Gid_2)
        dii.append(-k_eq)

    elif Gid_1<ni and Gid_2>=ni and Gid_2<ni+nf:
        lif.append(Gid_1)
        cif.append(Gid_2-ni)
        dif.append(k_eq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-k_eq)

    elif Gid_2<ni and Gid_1>=ni and Gid_1<ni+nf:
        lif.append(Gid_2)
        cif.append(Gid_1-ni)
        dif.append(k_eq)

        lii.append(Gid_2)
        cii.append(Gid_2)
        dii.append(-k_eq)

    elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni and Gid_2<ni+nf:
        lff.append(Gid_1-ni)
        cff.append(Gid_2-ni)
        dff.append(k_eq)

        lff.append(Gid_2-ni)
        cff.append(Gid_1-ni)
        dff.append(k_eq)

        lff.append(Gid_1-ni)
        cff.append(Gid_1-ni)
        dff.append(-k_eq)

        lff.append(Gid_2-ni)
        cff.append(Gid_2-ni)
        dff.append(-k_eq)

    elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni+nf and Gid_2<ni+nf+na:
        lfe.append(Gid_1-ni)
        cfe.append(Gid_2-ni-nf)
        dfe.append(k_eq)

        lff.append(Gid_1-ni)
        cff.append(Gid_1-ni)
        dff.append(-k_eq)

    elif Gid_2>=ni and Gid_2<ni+nf and Gid_1>=ni+nf and Gid_1<ni+nf+na:
        lfe.append(Gid_2-ni)
        cfe.append(Gid_1-ni-nf)
        dfe.append(k_eq)

        lff.append(Gid_2-ni)
        cff.append(Gid_2-ni)
        dff.append(-k_eq)

    elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf and Gid_2<ni+nf+na:
        lee.append(Gid_1-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(k_eq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(k_eq)

        lee.append(Gid_1-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(-k_eq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(-k_eq)

    elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf+na:
        lev.append(Gid_1-ni-nf)
        cev.append(Gid_2-ni-nf-na)
        dev.append(k_eq)

        lee.append(Gid_1-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(-k_eq)

    elif Gid_2>=ni+nf and Gid_2<ni+nf+na and Gid_1>=ni+nf+na:
        lev.append(Gid_2-ni-nf)
        cev.append(Gid_1-ni-nf-na)
        dev.append(k_eq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(-k_eq)

    elif Gid_1>=ni+nf+na and Gid_2>=ni+nf+na:
        lvv.append(Gid_1)
        cvv.append(Gid_2)
        dvv.append(k_eq)

        lvv.append(Gid_2)
        cvv.append(Gid_1)
        dvv.append(k_eq)

        lvv.append(Gid_1)
        cvv.append(Gid_1)
        dvv.append(-k_eq)

        lvv.append(Gid_2)
        cvv.append(Gid_2)
        dvv.append(-k_eq)
    cont+=1

Gid_1=ADJs1
Gid_2=ADJs2

lines=Gid_1
cols=Gid_2
data=ks
#T[Gid_1][Gid_2]=1
lines=np.concatenate([lines,Gid_2])
cols=np.concatenate([cols,Gid_1])
data=np.concatenate([data,ks])
#T[Gid_2][Gid_1]=1
lines=np.concatenate([lines,Gid_1])
cols=np.concatenate([cols,Gid_1])
data=np.concatenate([data,-ks])
#T[Gid_1][Gid_1]-=1
lines=np.concatenate([lines,Gid_2])
cols=np.concatenate([cols,Gid_2])
data=np.concatenate([data,-ks])
#T[Gid_2][Gid_2]-=1

T=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
Aii=csc_matrix((dii,(lii,cii)),shape=(ni,ni))
Aif=csc_matrix((dif,(lif,cif)),shape=(ni,nf))
Aff=csc_matrix((dff,(lff,cff)),shape=(nf,nf))
Afe=csc_matrix((dfe,(lfe,cfe)),shape=(nf,na))
Aee=csc_matrix((dee,(lee,cee)),shape=(na,na))
Aev=csc_matrix((dev,(lev,cev)),shape=(na,nv))
Avv=csc_matrix((dvv,(lvv,cvv)),shape=(nv,nv))

tomega=time.time()
tams=time.time()
ids_arestas=np.where(Aev.sum(axis=1)==0)[0]
ids_arestas_slin_m0=np.setdiff1d(range(na),ids_arestas)

Ivv=scipy.sparse.identity(nv)
invAee=lu_inv4(Aee,ids_arestas_slin_m0)
M2=-invAee*Aev
PAD=vstack([M2,Ivv])
invAff=get_invAff()
M3=-invAff*(Afe*M2)
PAD=vstack([M3,PAD])
del(M2)
invAii=get_invAii()
PAD=vstack([-invAii*(Aif*M3),PAD])
del(M3)
print("get_OP_AMS", time.time()-tams)
OP_AMS=PAD
l1=M1.mb.tag_get_data(fine_to_primal1_classic_tag, M1.all_volumes, flat=True)
c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
d1=np.ones((1,len(l1)),dtype=np.int)[0]
OR_AMS=csc_matrix((d1,(l1,c1)),shape=(nv,len(M1.all_volumes)))

##################condições de contorno#######################
ID_global=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d, flat=True)
T[ID_global]=scipy.sparse.csc_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global]=np.ones(len(ID_global))
########################## apagar para usar pressão-vazão
ID_globaln=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n, flat=True)
T[ID_globaln]=scipy.sparse.csc_matrix((len(ID_globaln),T.shape[0]))
T[ID_globaln,ID_globaln]=np.ones(len(ID_globaln))
##################################################fim_cond contorno#############

ln=[]
cn=[]
dn=[]

lines=[]
cols=[]
data=[]

IDs_globais_d=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d,flat=True)
lines_d=IDs_globais_d
cols_d=np.zeros((1,len(lines_d)),dtype=np.int32)[0]
data_d=np.repeat(press,len(lines_d))

IDs_globais_n=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n,flat=True)
lines_n=IDs_globais_n
cols_n=np.zeros(len(lines_n))
data_n=np.repeat(vazao,len(lines_n))

lines=np.concatenate([lines_d,lines_n])
cols=np.concatenate([cols_d,cols_n])
data=np.concatenate([data_d,data_n])
b=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),1))
#############################################################################
LIN_tag=M1.mb.tag_get_handle("LIN_tag", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
COL_tag=M1.mb.tag_get_handle("COL_tag", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
diag_tag=M1.mb.tag_get_handle("Diag_tag", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
RTP=OR_AMS*T*OP_AMS
diag_RTP=-RTP[range(len(vertices)),range(len(vertices))].toarray()[0]
dia=csc_matrix((-diag_RTP,(range(len(vertices)),range(len(vertices)))),shape=(len(vertices),len(vertices)))
mat=RTP-dia
ff=find(mat)
lines=ff[0]
cols=ff[1]
vals=ff[2]
negat=np.where(vals<0)[0]
posit=np.setdiff1d(range(len(vals)),negat)
lp=lines[posit]
cp=cols[posit]
ln=lines[negat]
cn=cols[negat]
mpos=csc_matrix((vals[posit],(lp,cp)),shape=(len(vertices),len(vertices)))
mneg=csc_matrix((vals[negat],(ln,cn)),shape=(len(vertices),len(vertices)))
pos=mpos.sum(axis=1).transpose()[0]
neg=mneg.sum(axis=1).transpose()[0]
rnp=-np.array(neg/pos)[0]
pos=mpos.sum(axis=0)[0]
neg=mneg.sum(axis=0)[0]
rnp2=-np.array(neg/pos)[0]
M1.mb.tag_set_data(LIN_tag,vertices,rnp)
M1.mb.tag_set_data(COL_tag,vertices,rnp2)
M1.mb.tag_set_data(diag_tag,vertices,diag_RTP)

###########################################################
GIDs=M1.mb.tag_get_data(M1.ID_reordenado_tag,M1.all_volumes,flat=True)
#####################################################################
def ADM_mesh(finos):
    tadmmesh=time.time()
    n1=-1
    n2=-1
    aux=0
    meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
    for m2 in meshset_by_L2:
        tem_poço_no_vizinho=False
        meshset_by_L1=M1.mb.get_child_meshsets(m2)
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            ver_1=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
            #ver_1=rng.unite(ver_1,M1.mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))
            if ver_1[0] in finos:
                aux=1
                tem_poço_no_vizinho=True
            else:
                #if (ar4/a9>0.02 and (ar6<40 or a9<0.05 or r_k_are_ver>10000)) or (ar4/a9>0.04 and (ar6<40 or a9<0.1 or r_k_are_ver>5000)) or (ar4/a9>0.1 and (ar6<100 and a9<0.4 and r_k_are_ver>100)) or (ar4/a9>0.2 and (ar6<300 and a9<0.3 and r_k_are_ver>1000)) or (ar4/a9>0.25 and (ar6<400 or a9<0.5 or r_k_are_ver>500)):
                if False:
                    #if ar>20 or r_k_are_ver>2000:
                    aux=1
                    tem_poço_no_vizinho=True
            if ver_1[0] in intermediarios:
                tem_poço_no_vizinho=True
            if aux==1:
                aux=0
                for elem in elem_by_L1:
                    n1+=1
                    n2+=1
                    M1.mb.tag_set_data(L1_ID_tag, elem, n1)
                    M1.mb.tag_set_data(L2_ID_tag, elem, n2)
                    M1.mb.tag_set_data(L3_ID_tag, elem, 1)
        if tem_poço_no_vizinho==False:
            elem_by_L2 = M1.mb.get_entities_by_handle(m2)
            vers=M1.mb.get_entities_by_type_and_tag(m2, types.MBHEX, np.array([D1_tag]), np.array([3]))
        if tem_poço_no_vizinho:
            for m1 in meshset_by_L1:
                elem_by_L1 = M1.mb.get_entities_by_handle(m1)
                n1+=1
                n2+=1
                t=1
                ver_1=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
                ver_1=rng.unite(ver_1,M1.mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))
                if ver_1[0] not in finos:
                    M1.mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
                    M1.mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
                    M1.mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(2,len(elem_by_L1)))
                    t=0
                n1-=t
                n2-=t
        else:
            n2+=1
            for m1 in meshset_by_L1:
                elem_by_L1 = M1.mb.get_entities_by_handle(m1)
                n1+=1
                M1.mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
                M1.mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
                M1.mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(3,len(elem_by_L1)))

    n1+=1
    n2+=1
    print(n1,n2)
    print(time.time()-tadmmesh,'definição da malha ADM')
    return(n1,n2)

def organize_OP(PAD):


    torganize=time.time()
    OP4=PAD.copy()
    mver=M1.mb.create_meshset()
    M1.mb.add_entities(mver,vertices)
    vnv1=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
    vnv2=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([3]))
    vs=rng.unite(vnv1,vnv2)
    IDs_AMS_vs=M1.mb.tag_get_data(M1.ID_reordenado_tag,vs,flat=True)
    IDs_ADM_vs=IDs_AMS_vs
    lines=IDs_ADM_vs
    cols=IDs_AMS_vs
    data=np.ones(len(lines))
    permut_elim=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),len(M1.all_volumes)))

    IDs_AMS_vert=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
    IDs_ADM_vert=M1.mb.tag_get_data(L1_ID_tag,vertices,flat=True)
    lines=IDs_AMS_vert
    cols=IDs_ADM_vert
    data=np.ones(len(lines))
    permut=csc_matrix((data,(lines,cols)),shape=(len(vertices),n1))
    vnv0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    IDs_ADM_vnv0=M1.mb.tag_get_data(M1.ID_reordenado_tag,vnv0,flat=True)
    IDs_AMS_vnv0=M1.mb.tag_get_data(L1_ID_tag,vnv0,flat=True)
    lines=IDs_ADM_vnv0
    cols=IDs_AMS_vnv0
    data=np.ones(len(lines))

    somar=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),n1))

    operador1=permut_elim*PAD*permut+somar
    #print(time.time()-torganize,"organize novo!!!")
    return operador1

def get_OR_ADM():
    l1=M1.mb.tag_get_data(L1_ID_tag, M1.all_volumes, flat=True)
    c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
    d1=np.ones(
    (1,len(l1)),dtype=np.int)[0]
    return csc_matrix((d1,(l1,c1)),shape=(n1,len(M1.all_volumes)))

def itere(xini,multip):
    titer=time.time()
    x0=xini
    ran=range(len(M1.all_volumes))
    D=T[ran,ran].toarray()[0]
    l_inv=range(len(M1.all_volumes))
    data_inv=1/D
    D_inv=csc_matrix((data_inv,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
    D=csc_matrix((D,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
    R=T-D
    x0=csc_matrix(x0).transpose()
    cont=0
    n=int(400*multip)

    for i in range(n):x0=D_inv*(b-R*x0)
    delta=(D_inv*(b-R*x0)-x0).max()
    #print(delta,delta_max,"delta e delta_max")
    cont+=n
    x0=x0.toarray().transpose()[0]
    # print(time.time()-titer,cont,delta,"tempo,  num_iterações,delta")
    return(x0,delta)

def itere2(xini,delta_max):
    titer=time.time()
    x0=xini
    ran=range(len(M1.all_volumes))
    D=T[ran,ran].toarray()[0]
    l_inv=range(len(M1.all_volumes))
    data_inv=1/D
    D_inv=csc_matrix((data_inv,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
    D=csc_matrix((D,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
    R=T-D
    x0=csc_matrix(x0).transpose()
    delta=2*delta_max
    cont=0
    n=100
    while  delta>delta_max:
        for i in range(n):x0=D_inv*(b-R*x0)
        delta=(D_inv*(b-R*x0)-x0).max()
        #print(delta,delta_max,"delta e delta_max")
        cont+=n
    x0=x0.toarray().transpose()[0]
    return(x0,delta)

def get_raz(raz_anterior,col_anterior):
    RTP=OR_ADM*T*OP_ADM
    pos_fin=M1.mb.tag_get_data(L1_ID_tag,finos,flat=True)
    diag_RTP=-RTP[range(n1),range(n1)].toarray()[0]
    dia=csc_matrix((-diag_RTP,(range(n1),range(n1))),shape=(n1,n1))
    mat=RTP-dia
    ff=find(mat)
    lines=ff[0]
    cols=ff[1]
    vals=ff[2]
    negat=np.where(vals<0)[0]
    posit=np.setdiff1d(range(len(vals)),negat)
    lp=lines[posit]
    cp=cols[posit]
    ln=lines[negat]
    cn=cols[negat]

    mpos=csc_matrix((vals[posit],(lp,cp)),shape=(n1,n1))
    mneg=csc_matrix((vals[negat],(ln,cn)),shape=(n1,n1))
    pos=mpos.max(axis=1).transpose().toarray()[0]
    neg=mneg.min(axis=1).transpose().toarray()[0]
    posic=np.where(np.array(neg)<0)[0]
    posic=np.setdiff1d(posic,pos_fin)
    pos=np.array(pos)[posic]
    neg=-np.array(neg)[posic]
    rnp=np.array(neg/pos)
    lim=np.sort(rnp)[-3]
    adms=posic[np.where(rnp>0.9*raz_anterior)[0]]
    # adms=posic[np.where(rnp>lim)[0]]
    amss=np.unique(np.array([ADM1_to_AMS[id] for id in adms],dtype=np.int))

    pos=mpos.max(axis=0).toarray()[0]
    neg=mneg.min(axis=0).toarray()[0]
    posic=np.where(np.array(neg)<0)[0]

    pos=np.array(pos)[posic]
    neg=-np.array(neg)[posic]
    rnp2=np.array(neg/pos)
    adms2=posic[np.where(rnp2>0.99*col_anterior)[0]]
    amss2=np.unique(np.array([ADM1_to_AMS[id] for id in adms2],dtype=np.int))
    #amss=np.concatenate([amss2,amss])
    amss=np.unique(amss)
    return(amss,rnp.max(),rnp2.max())
#####################################################################
IDs_vertices=range(ni+na+nf,ni+na+nf+nv)

tolC=0.3
tolL=0.8
posL=np.where(rnp>tolL)[0]
posC=np.where(rnp2>tolC)[0]
pC=np.where(rnp2>0.2)[0]
pL=np.where(rnp>0.6)[0]
p1=np.intersect1d(pC,pL)
positions=posL
positions=np.concatenate([positions,p1])
refs=np.array(vertices)[positions]
finos=np.intersect1d(np.array(vertices),finos)
finos=np.concatenate([finos,refs])
if calc_TPFA:
    SOL_TPFA=linalg.spsolve(T,b)
    print("resolveu TPFA: ",time.time()-t0)
    np.save('SOL_TPFA.npy',SOL_TPFA)
else:
    SOL_TPFA=np.load('SOL_TPFA.npy')
    print("leu TPFA: ")


finos=np.array(rng.unite(volumes_n,volumes_d))
plt.close()

n1,n2=ADM_mesh(finos)
OP_ADM=organize_OP(PAD)
OR_ADM=get_OR_ADM()

classic_ID=M1.mb.tag_get_data(fine_to_primal1_classic_tag,M1.all_volumes,flat=True)
ADM=M1.mb.tag_get_data(L1_ID_tag,M1.all_volumes,flat=True)
ADM1_to_AMS=dict(zip(ADM,classic_ID))

amss,l1,c1=get_raz(100,100)
amss,l1,c1=get_raz(l1,c1)

refs=np.array(vertices)[amss]
finos=np.concatenate([finos,refs])

n1,n2=ADM_mesh(finos)
OP_ADM=organize_OP(PAD)
OR_ADM=get_OR_ADM()

SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
SOL_ADM_1fina=OP_ADM*SOL_ADM_1
x0,delta=itere(SOL_ADM_1fina,2)
pseudo_erro=abs((SOL_ADM_1fina-x0)/x0)
# pseudo_erro_restringido=OR_AMS*pseudo_erro
pseudo_erro_restringido=pseudo_erro[IDs_vertices]
erro=abs(SOL_ADM_1fina-SOL_TPFA)/SOL_TPFA
# plt.plot(range(len(vertices)),np.sort(np.log(pseudo_erro[IDs_vertices]+0.001)),'y')
# plt.plot(range(len(vertices)),np.sort(np.log(erro[IDs_vertices]+0.001)),'b')
# plt.savefig("erro_verde,pseudo_erro_vermelho.png")
print(max(erro)*100,np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
# for i in range(5):
#     ###############################################################################
#     positions=np.where(pseudo_erro_restringido>max(pseudo_erro_restringido)/(9-i))[0]
#     refs=np.array(vertices)[positions]
#     finos=np.concatenate([finos,refs])
#     n1,n2=ADM_mesh(finos)
#     OP_ADM=organize_OP(PAD)
#     OR_ADM=get_OR_ADM()
#     SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
#     SOL_ADM_1fina=OP_ADM*SOL_ADM_1
#     x0,delta=itere(SOL_ADM_1fina,delta)
#     pseudo_erro=abs((SOL_ADM_1fina-x0)/x0)
#     pseudo_erro_restringido=OR_AMS*pseudo_erro
#     erro=abs(SOL_ADM_1fina-SOL_TPFA)/SOL_TPFA
#     plt.plot(range(len(M1.all_volumes)),np.sort(np.log(pseudo_erro+0.001)),'y')
#     plt.plot(range(len(M1.all_volumes)),np.sort(np.log(erro+0.001)),'b')
#     plt.savefig("erro_verde,pseudo_erro_vermelho.png")
#     print(max(erro)*100,np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
    #################################################################################
Pseudo_ERRO_tag=M1.mb.tag_get_handle("pseudo_erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERRO_tag=M1.mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
P_TPFA_tag=M1.mb.tag_get_handle("P_TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
P_ADM_tag=M1.mb.tag_get_handle("P_ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
perm_xx_tag=M1.mb.tag_get_handle("Perm_xx", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)

ADM=M1.mb.tag_get_data(L1_ID_tag,M1.all_volumes,flat=True)
#####################################################################
classic_ID=M1.mb.tag_get_data(fine_to_primal1_classic_tag,M1.all_volumes,flat=True)
reord_to_classic=dict(zip(GIDs,classic_ID))
ADM1_to_AMS=dict(zip(ADM,classic_ID))
psr=OR_AMS*pseudo_erro

tol=data_loaded['tol']
#tol_n2=len(M1.all_volumes)/2
tol_n2 = data_loaded['tol_n2']
tol_n2=tol_n2*len(M1.all_volumes)
cont=1
continuar=True
mpante=pseudo_erro_restringido.max()
refins=data_loaded['refins']
nr0=int(len(vertices)*refins)
perro=[]
ppseudo=[]

lg=[]
LL1=[]
LL2=[]
LL1.append(l1)
LL2.append(c1)
lg.append(100*n2/len(M1.all_volumes))
nl2=[]
nl2.append(np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA))
perro.append(erro.max())
ppseudo.append(pseudo_erro.max())

while max(pseudo_erro)>tol and n2<tol_n2 and continuar:
    ###############################################################################
    print(max(pseudo_erro[IDs_vertices]),pseudo_erro.max(),"erro_nos vértices")
    if pseudo_erro_restringido.max()>0:
        if cont<3:
            nr=2*nr0
            multip=2
        else:
            nr=nr0
            multip=1
        lim=np.sort(pseudo_erro)[len(pseudo_erro)-nr-1]
        ps=np.where(pseudo_erro>lim)[0]
        rmax=lim/pseudo_erro.max()
        classic=np.unique(np.array([reord_to_classic[v] for v in ps]))
        i=1
        while len(classic)<nr:
            lim=np.sort(pseudo_erro)[len(pseudo_erro)-nr-i-1]
            ps=np.where(pseudo_erro>lim)[0]
            rmax=lim/pseudo_erro.max()
            classic=np.unique(np.array([reord_to_classic[v] for v in ps]))
            i+=1

        positions=np.unique(classic)
        lim=np.sort(pseudo_erro_restringido)[len(pseudo_erro_restringido)-len(positions)-1]
        print(lim,"limite")
        #positions=np.where(pseudo_erro_restringido>lim)[0]
        positions_ver=np.where(pseudo_erro_restringido>lim)[0]
        rver=lim/pseudo_erro_restringido.max()

        lim=np.sort(psr)[len(psr)-len(positions)-1]
        p2=np.where(psr>lim)[0]
        rsum=lim/psr.max()
        if pseudo_erro_restringido.max()>1:
            # vec=[rmax,rver,rsum]
            vec=[rver,rmax]
        else:
            vec=[rmax]
            # vec=[rver,rmax,rsum]
        mv=min(vec)
        print(vec)

        if mv==rmax:
            positions=positions
            print('maximo!!!!!!!!!')
        if mv==rver:
            positions=positions_ver
            print('vertice!!!!!!!!!')
        if mv==rsum:
            positions=p2
            print('soma!!!!!!!!!!!!!!!')

    ADM=M1.mb.tag_get_data(L1_ID_tag,M1.all_volumes,flat=True)
    ADM1_to_AMS=dict(zip(ADM,classic_ID))
    raz_anterior=l1
    col_anterior=c1
    amss,l1,c1=get_raz(l1,c1)
    positions=np.concatenate([positions,amss])
    positions=np.unique(positions)
    print(len(positions))

    refs=np.array(vertices)[positions]
    finos=np.concatenate([finos,refs])
    n1,n2=ADM_mesh(finos)
    OP_ADM=organize_OP(PAD)
    OR_ADM=get_OR_ADM()
    # while l1>0.95*raz_anterior:
    while False:
        ADM=M1.mb.tag_get_data(L1_ID_tag,M1.all_volumes,flat=True)
        ADM1_to_AMS=dict(zip(ADM,classic_ID))
        amss,l1,c1=get_raz(l1,c1)
        refs=np.array(vertices)[amss]
        finos=np.concatenate([finos,refs])
        n1,n2=ADM_mesh(finos)
        OP_ADM=organize_OP(PAD)
        OR_ADM=get_OR_ADM()
        print(len(amss),l1,raz_anterior,"No loop do fluxo!!!")

    SOL_ADM_fina_ant=SOL_ADM_1fina
    SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
    SOL_ADM_1fina=OP_ADM*SOL_ADM_1
    sol_vers=SOL_ADM_1fina[IDs_vertices]
    sol_prol=OP_AMS*sol_vers
    x0,delta=itere(SOL_ADM_1fina,multip)
    pseudo_erro=abs((SOL_ADM_1fina-x0)/x0)

    LL1.append(l1)
    LL2.append(c1)
    lg.append(100*n2/len(M1.all_volumes))
    nl2.append(np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA))

    plt.close()
    plt.scatter(lg,np.array(LL1),3,'r')
    plt.plot(lg,np.array(LL1),'r')
    #plt.scatter(lg,np.array(LL2),1)
    plt.savefig("raz_lin.png")
    plt.close()
    plt.scatter(lg,np.array(nl2),3,'r')
    plt.plot(lg,np.array(nl2),'r')
    plt.savefig("nl2.png")
    plt.close()

    # pseudo_erro_restringido=OR_AMS*pseudo_erro
    p1=OR_AMS*pseudo_erro
    mpante=pseudo_erro_restringido.max()
    pseudo_erro_restringido=pseudo_erro[IDs_vertices]
    erro=abs(SOL_ADM_1fina-SOL_TPFA)/SOL_TPFA
    # plt.plot(range(len(vertices)),np.sort(np.log(pseudo_erro_restringido+0.001)),'y')
    # plt.plot(range(len(vertices)),np.sort(np.log(erro[IDs_vertices]+0.001)),'b')
    # plt.savefig("erro_verde,pseudo_erro_vermelho.png")
    psr=OR_AMS*pseudo_erro
    # plt.plot(range(len(psr)),np.sort(np.log(psr+0.001)),'g')
    perro.append(erro.max())
    ppseudo.append(pseudo_erro.max())
    plt.close()
    plt.scatter(lg,100*(np.array(perro)),3,'g')
    plt.scatter(lg,100*(np.array(ppseudo)),3,'y')
    plt.plot(lg,100*(np.array(perro)),'g')
    plt.plot(lg,100*(np.array(ppseudo)),'y')
    plt.savefig("psr.png")
    print(max(erro)*100,np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")

    if max(pseudo_erro)<tol:
        print("ATINGIU A TOLERÂNCIA!!!")
    if n2>tol_n2:
        print("ATINGIU O TAMANHO MÁXIMO!!!")
    ############################################
    cont+=1
    '''
    M1.mb.tag_set_data(Pseudo_ERRO_tag,M1.all_volumes,pseudo_erro[GIDs])
    M1.mb.tag_set_data(ERRO_tag,M1.all_volumes,erro[GIDs])
    M1.mb.tag_set_data(P_ADM_tag,M1.all_volumes,SOL_ADM_1fina[GIDs])
    ext_vtk = 'testes_MAD'  + str(cont) + '.vtk'
    M1.mb.write_file(ext_vtk,[av])'''

#####################################################################################
'''
diag_ad1=((OR_ADM*T*OP_ADM)[range(len(SOL_ADM_1)),range(len(SOL_ADM_1))]).toarray()[0]
mdiag=csc_matrix((1/diag_ad1,(range(len(SOL_ADM_1)),range(len(SOL_ADM_1)))),shape=(len(SOL_ADM_1),len(SOL_ADM_1)))
S1=linalg.spsolve(mdiag*(OR_ADM*T*OP_ADM),mdiag*OR_ADM*b)'''# teste de precondicionador


print("total",time.time()-tomega)
'''
if not calc_TPFA:
    SOL_TPFA=np.load('SOL_TPFA.npy')
    if len(SOL_TPFA)==len(M1.all_volumes):
        M1.mb.tag_set_data(P_TPFA_tag,M1.all_volumes,SOL_TPFA[GIDs])
    else:
        M1.mb.tag_set_data(P_TPFA_tag,M1.all_volumes,SOL_TPFA)
else:
    print("resolvendo TPFA")
    t0=time.time()
    if len(M1.all_volumes)<100000:
        SOL_TPFA=linalg.spsolve(T,b)
        print("resolveu TPFA: ",time.time()-t0)
        np.save('SOL_TPFA.npy',SOL_TPFA)
    else:
        SOL_TPFA=np.ones(len(M1.all_volumes))
        print('TPFA muito grande > 30.000 vols!!! não resolveu')'''
perms_xx=M1.mb.tag_get_data(M1.perm_tag,M1.all_volumes)[:,0]

x=np.array([lg,nl2,perro,LL1,ppseudo]).transpose()
np.savetxt('PV454545CR333limitadorefin005_homogeneo.csv',x,delimiter=',')
erro=abs(SOL_TPFA-SOL_ADM_1fina)/SOL_TPFA
erro_ABS=abs(SOL_TPFA-SOL_ADM_1fina)
#print(max(erro)*100,np.linalg.norm(erro_ABS)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
M1.mb.tag_set_data(Pseudo_ERRO_tag,M1.all_volumes,pseudo_erro[GIDs])
M1.mb.tag_set_data(ERRO_tag,M1.all_volumes,erro[GIDs])
M1.mb.tag_set_data(P_ADM_tag,M1.all_volumes,SOL_ADM_1fina[GIDs])
M1.mb.tag_set_data(P_TPFA_tag,M1.all_volumes,SOL_TPFA[GIDs])
M1.mb.tag_set_data(perm_xx_tag,M1.all_volumes,np.array(perms_xx))

n1, n2 = ADM_mesh(np.concatenate([volumes_n, volumes_d]))

av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)
M1.mb.write_file("testes_MAD.vtk",[av])

finos_0_meshset = M1.mb.create_meshset()
finos_0 = M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
M1.mb.add_entities(finos_0_meshset, finos_0)
finos_0_tag = M1.mb.tag_get_handle('finos0', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
M1.mb.tag_set_data(finos_0_tag, 0, finos_0_meshset)
intermediarios_meshset = M1.mb.create_meshset()
intermediarios = M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
M1.mb.add_entities(intermediarios_meshset, intermediarios)
intermediarios_tag = M1.mb.tag_get_handle('intermediarios', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
M1.mb.tag_set_data(intermediarios_tag, 0, intermediarios_meshset)

wells_injector_tag = M1.mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
wells_producer_tag = M1.mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
wells_injector_meshset = M1.mb.create_meshset()
wells_producer_meshset = M1.mb.create_meshset()
M1.mb.add_entities(wells_injector_meshset, volumes_n)
M1.mb.add_entities(wells_producer_meshset, volumes_d)
M1.mb.tag_set_data(wells_injector_tag, 0, wells_injector_meshset)
M1.mb.tag_set_data(wells_producer_tag, 0, wells_producer_meshset)

ext_h5m_out = input_name + '_malha_adm.h5m'
M1.mb.write_file(ext_h5m_out)
np.save('faces_adjs_by_dual', faces_adjs_by_dual)
np.save('intern_adjs_by_dual', intern_adjs_by_dual)
