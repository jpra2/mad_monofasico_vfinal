import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
from matplotlib import pyplot as plt
# import sympy
import cython
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, vstack, hstack, linalg, identity, find

__all__ = ['M1']


class MeshManager:
    def __init__(self,mesh_file, dim=3):
        self.dimension = dim
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

        self.mb.load_file(mesh_file)

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
        self.set_k_and_phi_structured_spe10()
        # self.set_k()
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
M1= MeshManager('60x220x85.msh')          # Objeto que armazenará as informações da malha
all_volumes=M1.all_volumes
# press = -20000.0
# press= 7785.0
press= 4000.0
vazao = 10000.0

testar_MPFA=False
MPFA_NO_NIVEL_2=True
calc_TPFA=False
load_TPFA=False
corrigir_pocos=False
conectar_pocos=False

iterar_mono=True
so_pressao=True
refinar_nv2=True

imprimir_a_cada_iteracao=True

rel_v2=1.2

TOL=0.0
tol_n2=0.2
Ni=5
# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)

M1.all_centroids=np.array([M1.mtu.get_average_position([v]) for v in all_volumes])
all_centroids = M1.all_centroids

nx=60
ny=220
nz=85

lx=20
ly=10
lz=2

l1=[5*lx,5*ly,3*lz]
l2=[15*lx,15*ly,9*lz]

x1=nx*lx
y1=ny*ly
z1=nz*lz
# Distância, em relação ao poço, até onde se usa malha fina
r0 = 1
# Distância, em relação ao poço, até onde se usa malha intermediária
r1 = 1
'''
bvd = np.array([np.array([0.0, y1-ly, 0.0]), np.array([x1, y1, z1])])
bvn = np.array([np.array([0.0, 0.0, 0.0]), np.array([x1, ly, z1])])'''
# bvd = np.array([np.array([x1-lx, 0.0, 0.0]), np.array([x1, y1, lz])])
# bvn = np.array([np.array([0.0, 0.0, z1-lz]), np.array([lx, y1, z1])])
bvn = np.array([np.array([x1-lx, y1-ly, z1-22*lz]), np.array([x1, y1, z1-9*lz])])
bvd = np.array([np.array([0.0, 0.0, 0.0]), np.array([lx, ly, z1])])
'''
bvn = np.array([np.array([x1-lx, y1-ly, 0.0]), np.array([x1, y1, z1])])   ############ Usar esse exemplo
bvd = np.array([np.array([0.0, 0.0, z1-27*lz]), np.array([lx, ly, z1-6*lz])])
'''
bvn2 = np.array([np.array([x1-lx, 7*ly, 0.0]), np.array([x1, 8*ly, 0*lz])])

'''
bvd = np.array([np.array([0.0, 0.0, 0.0]), np.array([lx, ly, z1])])
bvn = np.array([np.array([x1-lx, y1-ly, 0.0]), np.array([x1, y1, z1])])'''
#bvd = np.array([np.array([0.0, 0.0, y2]), np.array([y0, y0, y0])])
#bvn = np.array([np.array([0.0, 0.0, 0.0]), np.array([y0, y0, y1])])

bvfd = np.array([np.array([bvd[0][0]-r0*lx, bvd[0][1]-r0*ly, bvd[0][2]-r0*lz]), np.array([bvd[1][0]+r0*lx, bvd[1][1]+r0*ly, bvd[1][2]+r0*lz])])
bvfn = np.array([np.array([bvn[0][0]-r0*lx, bvn[0][1]-r0*ly, bvn[0][2]-r0*lz]), np.array([bvn[1][0]+r0*lx, bvn[1][1]+r0*ly, bvn[1][2]+r0*lz])])

bvid = np.array([np.array([bvd[0][0]-r1, bvd[0][1]-r1, bvd[0][2]-r1]), np.array([bvd[1][0]+r1, bvd[1][1]+r1, bvd[1][2]+r1])])
bvin = np.array([np.array([bvn[0][0]-r1, bvn[0][1]-r1, bvn[0][2]-r1]), np.array([bvn[1][0]+r1, bvn[1][1]+r1, bvn[1][2]+r1])])

# volumes com pressao prescrita

volumes_d, inds_vols_d= get_box(M1.all_volumes, all_centroids, bvd, True)

# volumes com vazao prescrita

volumes_n1, inds_vols_n = get_box(M1.all_volumes, all_centroids, bvn, True)
volumes_n2, lixo=get_box(M1.all_volumes, all_centroids, bvn2, True)
volumes_n=rng.unite(volumes_n1,volumes_n2)

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
    s=50
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
    s=10
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
M1.mb.tag_set_data(M1.press_value_tag, volumes_n, np.repeat(vazao/len(volumes_n), len(volumes_n)))
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

    Keq=Kharm*area/norm_direction
    # Keq=np.ones(len(Kharm))
    other_faces=np.setdiff1d(np.uint64(M1.all_faces),conj_faces)
    M1.mb.tag_set_data(self.k_eq_tag, conj_faces, Keq)

    M1.mb.tag_set_data(self.k_eq_tag, other_faces, np.zeros(len(other_faces)))

    #self.mb.tag_set_data(self.kharm_tag, self.all_faces, kharm)
    return deltas

################################################################################
all_intern_faces=[face for face in M1.all_faces if len(M1.mb.get_adjacencies(face, 3))==2]
all_intern_adjacencies=np.array([M1.mb.get_adjacencies(face, 3) for face in all_intern_faces])


t1=time.time()
deltas=set_kequiv(M1,all_intern_faces,all_intern_adjacencies)
print("setou k",time.time()-t1)

kst=M1.mb.tag_get_data(M1.k_eq_tag,all_intern_faces,flat=True)

if corrigir_pocos:
    if conectar_pocos:
        z_1=np.array([M1.mtu.get_average_position([v]) for v in volumes_n1])[:,2]
        z_1_min=z_1.min()
        pos1=np.where(z_1==z_1_min)[0][0]

        z_2=np.array([M1.mtu.get_average_position([v]) for v in volumes_n2])[:,2]
        z_2_max=z_2.max()
        pos2=np.where(z_2==z_2_max)[0][0]

        v1v2=np.uint64([np.array(volumes_n1)[pos1],np.array(volumes_n2)[pos2]])
        IDs_12=M1.mb.tag_get_data(M1.ID_reordenado_tag,v1v2,flat=True)
        K_12=kst.max()
        # all_intern_adjacencies=np.concatenate([all_intern_adjacencies,np.array([v1v2])])
        # kst=np.concatenate([kst,np.array([kst.max()])])

    faces_n=[]
    for v in volumes_n: faces_n.append(np.array(M1.mtu.get_bridge_adjacencies(v,3,2)))
    fc_n=np.concatenate(faces_n)
    facs_nn=[]
    for f in fc_n:
        if len(np.where(fc_n==f)[0])==2:facs_nn.append(f)
    facs_nn=np.unique(np.uint64(facs_nn))
    ks_neu=M1.mb.tag_get_data(M1.k_eq_tag,facs_nn,flat=True)
    print(len(fc_n),len(facs_nn))
    vals=np.repeat(kst.max(),len(facs_nn))
    M1.mb.tag_set_data(M1.k_eq_tag, np.uint64(facs_nn), vals)
    kst=M1.mb.tag_get_data(M1.k_eq_tag,all_intern_faces,flat=True)

#####################################################
all_adjacent_volumes=[]
all_adjacent_volumes.append(M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(all_intern_adjacencies[:,0]),flat=True))
all_adjacent_volumes.append(M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(all_intern_adjacencies[:,1]),flat=True))
###############################################################################
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
## set termo fonte

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
if so_pressao:
    data_n=np.repeat(vazao,len(lines_n))
else:
    data_n=np.repeat(-vazao/len(volumes_n),len(lines_n))

lines=np.concatenate([lines_d,lines_n])
cols=np.concatenate([cols_d,cols_n])
data=np.concatenate([data_d,data_n])
b=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),1))

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
if conectar_pocos and conectar_pocos:
    # T[IDs_12]=K_12
    lines=np.concatenate([lines,np.array([IDs_12[0]])])
    cols=np.concatenate([cols,np.array([IDs_12[1]])])
    data=np.concatenate([data,np.array([K_12])])
    cols=np.concatenate([cols,np.array([IDs_12[0]])])
    lines=np.concatenate([lines,np.array([IDs_12[1]])])
    data=np.concatenate([data,np.array([K_12])])
    lines=np.concatenate([lines,np.array([IDs_12[0]])])
    cols=np.concatenate([cols,np.array([IDs_12[0]])])
    data=np.concatenate([data,np.array([-K_12])])
    lines=np.concatenate([lines,np.array([IDs_12[1]])])
    cols=np.concatenate([cols,np.array([IDs_12[1]])])
    data=np.concatenate([data,np.array([-K_12])])

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
OP_AMS=PAD.copy()
del(PAD)
del(invAii)
del(invAff)
del(invAee)
def get_OR_AMS():
    l1=M1.mb.tag_get_data(fine_to_primal1_classic_tag, M1.all_volumes, flat=True)
    c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
    d1=np.ones((1,len(l1)),dtype=np.int)[0]
    OR_AMS=csc_matrix((d1,(l1,c1)),shape=(nv,len(M1.all_volumes)))
    return(OR_AMS)

OR_AMS=get_OR_AMS()

v=M1.mb.create_meshset()
M1.mb.add_entities(v,vertices)
inte=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))
M1.mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)

def get_OP_AMS_2():
    cols=M1.mb.tag_get_data(fine_to_primal1_classic_tag,np.concatenate([inte,fac,are,ver]),flat=True)
    lines=range(len(cols))
    data=np.ones(len(cols))

    G=csc_matrix((data,(lines,cols)),shape=(len(vertices),len(vertices)))

    T_AMS=OR_AMS*T*OP_AMS
    W_AMS=G*T_AMS*G.transpose()

    ni=nint
    nf=nfac
    na=nare
    nv=nver

    Aii=W_AMS[0:ni,0:ni]
    Aif=W_AMS[0:ni,ni:ni+nf]
    Aie=W_AMS[0:ni,ni+nf:ni+nf+na]
    Aiv=W_AMS[0:ni,ni+nf+na:ni+nf+na+nv]

    lines=[]
    cols=[]
    data=[]
    if MPFA_NO_NIVEL_2==False:
        lines=range(ni)
        data=np.array(Aie.sum(axis=1)+Aiv.sum(axis=1)).transpose()[0]
        S=csc_matrix((data,(lines,lines)),shape=(ni,ni))
        Aii += S
        del(S)

    Afi=W_AMS[ni:ni+nf,0:ni]
    Aff=W_AMS[ni:ni+nf,ni:ni+nf]
    Afe=W_AMS[ni:ni+nf,ni+nf:ni+nf+na]
    Afv=W_AMS[ni:ni+nf,ni+nf+na:ni+nf+na+nv]

    lines=range(nf)
    data_fi=np.array(Afi.sum(axis=1)).transpose()[0]
    data_fv=np.array(Afv.sum(axis=1)).transpose()[0]

    Sfi=csc_matrix((data_fi,(lines,lines)),shape=(nf,nf))
    Aff += Sfi
    if MPFA_NO_NIVEL_2==False:
        Sfv=csc_matrix((data_fv,(lines,lines)),shape=(nf,nf))
        Aff +=Sfv

    Aei=W_AMS[ni+nf:ni+nf+na,0:ni]
    Aef=W_AMS[ni+nf:ni+nf+na,ni:ni+nf]
    Aee=W_AMS[ni+nf:ni+nf+na,ni+nf:ni+nf+na]
    Aev=W_AMS[ni+nf:ni+nf+na,ni+nf+na:ni+nf+na+nv]

    lines=range(na)
    data=np.array(Aei.sum(axis=1)+Aef.sum(axis=1)).transpose()[0]
    S=csc_matrix((data,(lines,lines)),shape=(na,na))
    Aee += S

    Ivv=scipy.sparse.identity(nv)
    invAee=lu_inv2(Aee)
    M2=-csc_matrix(invAee)*Aev
    P2=vstack([M2,Ivv])

    invAff=lu_inv2(Aff)
    if MPFA_NO_NIVEL_2:
        M3=-invAff*Afe*M2-invAff*Afv
        P2=vstack([M3,P2])
    else:
        Mf=-invAff*Afe*M2
        P2=vstack([Mf,P2])
    invAii=lu_inv2(Aii)
    if MPFA_NO_NIVEL_2:
        M3=invAii*(-Aif*M3+Aie*invAee*Aev-Aiv)
        P2=vstack([M3,P2])
    else:
        P2=vstack([-invAii*Aif*Mf,P2])

    # IDs_dirichlet=M1.mb.tag_get_data(fine_to_primal1_classic_tag,volumes_n,flat=True)
    # bn=b.copy()
    # bn[IDs_dirichlet]=0

    b_ams2_wire=G*OR_AMS*b
    b_ams2_int=b_ams2_wire[0:ni,0]
    b_ams2_fac=b_ams2_wire[ni:ni+nf,0]
    b_ams2_are=b_ams2_wire[ni+nf:ni+nf+na,0]
    b_ams2_ver=b_ams2_wire[ni+nf+na:ni+nf+na+nv,0]

    corr=csc_matrix((len(ver),1))
    corr=vstack([invAee*b_ams2_are,corr])
    corr=vstack([invAff*b_ams2_fac-invAff*Afe*invAee*b_ams2_are,corr])
    corr=vstack([invAii*b_ams2_int-invAii*Aif*invAff*b_ams2_fac+invAii*(Aif*invAff*Afe*invAee-Aie*invAee)*b_ams2_are,corr])

    c2=csc_matrix((nv,ni+nf+na+nv))
    c2=vstack([hstack([hstack([csc_matrix((na,ni+nf)),invAee]),csc_matrix((na,nv))]),c2])
    c2=vstack([hstack([hstack([hstack([csc_matrix((nf,ni)),invAff]),-invAff*Afe*invAee]),csc_matrix((nf,nv))]),c2])
    c2=vstack([hstack([hstack([hstack([invAii,-invAii*Aif*invAff]),invAii*(Aif*invAff*Afe*invAee-Aie*invAee)]),csc_matrix((ni,nv))]),c2])
    c2=csc_matrix(c2)

    return G.transpose()*P2, G.transpose()*c2*G

OP_AMS_2, corr=get_OP_AMS_2()


##################condições de contorno#######################
ID_global=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d, flat=True)
T[ID_global]=scipy.sparse.csc_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global]=np.ones(len(ID_global))
########################## apagar para usar pressão-vazão
if so_pressao:
    ID_globaln=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n, flat=True)
    T[ID_globaln]=scipy.sparse.csc_matrix((len(ID_globaln),T.shape[0]))
    T[ID_globaln,ID_globaln]=np.ones(len(ID_globaln))
    ##################################################fim_cond contorno#############

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
def ADM_mesh(finos,intermediarios):
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
            if ver_1[0] in intermediarios:
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
    print(time.time()-torganize,"organize novo!!!")
    return operador1

def organize_OP_2(OP_AMS_2):
    torganize=time.time()
    OP4=OP_AMS_2.copy()
    mver=M1.mb.create_meshset()
    M1.mb.add_entities(mver,vertices)
    vnv2=M1.mb.get_entities_by_type_and_tag(mver, types.MBHEX, np.array([L3_ID_tag]), np.array([3]))
    IDs_AMS_vs=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vnv2,flat=True)
    IDs_ADM_vs=M1.mb.tag_get_data(L1_ID_tag,vnv2,flat=True)
    data=np.ones(len(IDs_AMS_vs))
    permut_elim=csc_matrix((data,(IDs_ADM_vs,IDs_AMS_vs)),shape=(n1,len(vertices)))

    IDs_AMS_vert=M1.mb.tag_get_data(fine_to_primal2_classic_tag,ver,flat=True)
    IDs_ADM_vert=M1.mb.tag_get_data(L2_ID_tag,ver,flat=True)
    lines=IDs_AMS_vert
    cols=IDs_ADM_vert
    data=np.ones(len(lines))
    permut=csc_matrix((data,(lines,cols)),shape=(len(ver),n2))
    vnv0=np.setdiff1d(vertices,vnv2)

    all_v_nv0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    vnv0=np.unique(np.concatenate([vnv0,all_v_nv0]))

    IDs_ADM_vnv0=M1.mb.tag_get_data(L1_ID_tag,vnv0,flat=True)
    IDs_AMS_vnv0=M1.mb.tag_get_data(L2_ID_tag,vnv0,flat=True)
    lines=IDs_ADM_vnv0
    cols=IDs_AMS_vnv0
    data=np.ones(len(lines))

    somar=csc_matrix((data,(lines,cols)),shape=(n1,n2))
    operador1=permut_elim*OP4*permut+somar
    print(time.time()-torganize,'organize_OP_2')
    return operador1

def get_OR_ADM():
    l1=M1.mb.tag_get_data(L1_ID_tag, M1.all_volumes, flat=True)
    c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
    d1=np.ones(
    (1,len(l1)),dtype=np.int)[0]
    return csc_matrix((d1,(l1,c1)),shape=(n1,len(M1.all_volumes)))

def get_OR_ADM_2():
    l1=M1.mb.tag_get_data(L2_ID_tag, M1.all_volumes, flat=True)
    c1=M1.mb.tag_get_data(L1_ID_tag, M1.all_volumes, flat=True)
    d1=np.ones(len(l1))
    OR_ADM_2=csc_matrix((d1,(l1,c1)),shape=(n2,n1))
    r2=find(OR_ADM_2)
    lin=r2[0]
    col=r2[1]
    dat=np.ones((1,len(lin)),dtype=np.int)[0]
    OR_ADM_2=csc_matrix((dat,(lin,col)),shape=(n2,n1))
    return OR_ADM_2

def itere2(xini,multip):
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

def itere(xini,delta_max):
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
        if cont==0:
            delta_max=delta/5
        print(delta,delta_max,"delta e delta_max")
        cont+=n
    x0=x0.toarray().transpose()[0]

    return(x0,delta)

def itere3(xini):
    titer=time.time()
    ran=range(len(M1.all_volumes))
    D=T[ran,ran].toarray()[0]
    l_inv=range(len(M1.all_volumes))
    data_inv=1/D
    D_inv=csc_matrix((data_inv,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
    D=csc_matrix((D,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
    R=T-D
    x0=csc_matrix(xini).transpose()
    n=50
    cont=0
    for i in range(n):x0=D_inv*(b-R*x0)
    delta_ant=abs((D_inv*(b-R*x0)-x0)).max()
    cont+=n
    for i in range(n):x0=D_inv*(b-R*x0)
    delta=abs((D_inv*(b-R*x0)-x0)).max()
    cont+=n
    while  delta<0.6*delta_ant:
        delta_ant=delta
        for i in range(n):x0=D_inv*(b-R*x0)
        delta=abs((D_inv*(b-R*x0)-x0)).max()
        cont+=n
    x0=x0.toarray().transpose()[0]
    print(time.time()-titer,n, "iterou ")
    return(x0)

def itere_ver(xini):
    titer=time.time()
    ifa=len(internos)+len(faces)+len(arestas)
    lc=range(ifa,len(M1.all_volumes))
    D=T[lc,lc].toarray()[0]
    l_inv=range(len(vertices))
    c_inv=lc
    data_inv=1/D
    D_inv=csc_matrix((data_inv,(l_inv,l_inv)),shape=(len(vertices),len(M1.all_volumes)))
    D=csc_matrix((D,(l_inv,c_inv)),shape=(len(vertices),len(M1.all_volumes)))
    R=T[lc]-D
    x0=csc_matrix(xini).transpose()
    cont=0
    n=100
    bv=b[IDs_vertices]
    import pdb; pdb.set_trace()
    for i in range(n):x0=D_inv*(bv-R*x0)
    delta_ant=(D_inv*(b-R*x0)-x0).max()
    cont+=n
    for i in range(n):x0=D_inv*(bv-R*x0)
    delta=(D_inv*(b-R*x0)-x0).max()
    cont+=n
    while  delta_ant>1.1*delta:
        delta_ant=delta
        for i in range(n):x0=D_inv*(bv-R*x0)
        delta=(D_inv*(b-R*x0)-x0).max()
        print(delta,delta_max,"delta e delta_max")
        cont+=n
    x0=x0.toarray().transpose()[0]
    import pdb; pdb.set_trace()
    return(x0)

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
#####################################################################
Pseudo_ERRO_tag=M1.mb.tag_get_handle("pseudo_erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERRO_tag=M1.mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
P_TPFA_tag=M1.mb.tag_get_handle("P_TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
P_ADM_tag=M1.mb.tag_get_handle("P_ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)

if calc_TPFA:
    SOL_TPFA=linalg.spsolve(T,b)
    LU=linalg.splu(T)
    sol=LU.solve(b.toarray())
    print("resolveu TPFA: ",time.time()-t0)
    np.save('SOL_TPFA.npy',SOL_TPFA)

if load_TPFA:
    SOL_TPFA=np.load('SOL_TPFA.npy')

active_nodes=[]
perro=[]
erro=[]


Nmax=tol_n2*len(M1.all_volumes)

finos=np.array(rng.unite(volumes_n,volumes_d))


pfins=np.unique(M1.mb.tag_get_data(fine_to_primal1_classic_tag,finos,flat=True))

nr=int(tol_n2*(len(vertices)-len(finos))/(Ni))

n1,n2=ADM_mesh(finos,[])

pseudo_erro=np.repeat(TOL+1,2) #iniciou pseudo_erro
t0=time.time()
cont=0
while max(pseudo_erro)>TOL and n2<Nmax and iterar_mono:
    if cont>0:
        lim=np.sort(psr)[len(psr)-nr-1]
        positions=np.where(psr>lim)[0]
        finos=np.concatenate([finos,np.array(vertices)[positions]])
        pfins=np.concatenate([pfins,positions])
        n1,n2=ADM_mesh(finos,[])
    OP_ADM=organize_OP(OP_AMS)
    OR_ADM=get_OR_ADM()
    if testar_MPFA:
        OP_ADM_2=organize_OP_2(OP_AMS_2)
        OR_ADM_2=get_OR_ADM_2()
        SOL_ADM=linalg.spsolve(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2,OR_ADM_2*OR_ADM*b)
        SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM
    else:
        SOL_ADM=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
        SOL_ADM_fina=OP_ADM*SOL_ADM
    x0=itere3(SOL_ADM_fina)
    pseudo_erro=abs((SOL_ADM_fina-x0))

    if calc_TPFA or load_TPFA:
        erro.append(abs((SOL_TPFA-SOL_ADM_fina)/SOL_TPFA).max())
    else:
        erro.append(abs(pseudo_erro/x0).max())
    psr=(OR_AMS*abs(pseudo_erro))
    psr[pfins]=0

    perro.append(abs((SOL_ADM_fina-x0)/x0).max())
    active_nodes.append(n2/len(M1.all_volumes))
    if (not calc_TPFA) and (not load_TPFA):
        SOL_TPFA=x0
    if imprimir_a_cada_iteracao:
        M1.mb.tag_set_data(Pseudo_ERRO_tag,M1.all_volumes,abs(pseudo_erro/x0)[GIDs])

        M1.mb.tag_set_data(ERRO_tag,M1.all_volumes,abs((SOL_ADM_fina-SOL_TPFA)/SOL_TPFA)[GIDs])
        M1.mb.tag_set_data(P_ADM_tag,M1.all_volumes,SOL_ADM_fina[GIDs])
        M1.mb.tag_set_data(P_TPFA_tag,M1.all_volumes,SOL_TPFA[GIDs])
        ext_vtk = 'testes_MAD'  + str(cont) + '.vtk'
        M1.mb.write_file(ext_vtk,[av])
    cont+=1


IDs_pocos=np.unique(M1.mb.tag_get_data(fine_to_primal1_classic_tag,np.concatenate([volumes_n,volumes_d]),flat=True))
if so_pressao:
    P_pocos=np.concatenate([SOL_ADM_fina[IDs_pocos],np.array([press,vazao])])
else:
    P_pocos=np.concatenate([SOL_ADM_fina[IDs_pocos],np.array([press])])
pmin=min(P_pocos)

if refinar_nv2 and iterar_mono and cont>0:
    cont+=1
    #DMP
    p2=np.where(SOL_ADM_fina[IDs_vertices]<pmin)[0]
    finos=np.concatenate([finos,np.array(vertices)[p2]])
    OP_ADM_2=organize_OP_2(OP_AMS_2)
    OR_ADM_2=get_OR_ADM_2()
    SOL_ADM=linalg.spsolve(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2,OR_ADM_2*OR_ADM*b)
    SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM
    err2=abs(SOL_ADM_fina-x0)[IDs_vertices]
    pemax=pseudo_erro.max()
    positions=np.where(err2>rel_v2*pemax)[0]
    if len(positions)>0 or len(p2)>0:
        n1,n2=ADM_mesh(finos,np.array(vertices)[positions])
        # OP_ADM=organize_OP(OP_AMS)
        # OP_ADM_2=organize_OP_2(OP_AMS_2)
        # OR_ADM=get_OR_ADM()
        # OR_ADM_2=get_OR_ADM_2()
        SOL_ADM=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
        SOL_ADM_fina=OP_ADM*SOL_ADM
        pseudo_erro=abs((SOL_ADM_fina-x0))
        perro.append(abs((SOL_ADM_fina-x0)/x0).max())
        active_nodes.append(n2/len(M1.all_volumes))
        if calc_TPFA or load_TPFA:
            erro.append(abs((SOL_TPFA-SOL_ADM_fina)/SOL_TPFA).max())
        else:
            erro.append(abs(pseudo_erro/x0).max())
    if imprimir_a_cada_iteracao:
        M1.mb.tag_set_data(Pseudo_ERRO_tag,M1.all_volumes,abs(pseudo_erro/x0)[GIDs])
        M1.mb.tag_set_data(ERRO_tag,M1.all_volumes,abs((SOL_ADM_fina-SOL_TPFA)/SOL_TPFA)[GIDs])
        M1.mb.tag_set_data(P_ADM_tag,M1.all_volumes,SOL_ADM_fina[GIDs])
        ext_vtk = 'testes_MAD'  + str(cont) + '.vtk'
        M1.mb.write_file(ext_vtk,[av])

OP_ADM_2=organize_OP_2(OP_AMS_2)
OR_ADM_2=get_OR_ADM_2()
try:
    SOL_ADM=linalg.spsolve(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2,OR_ADM_2*OR_ADM*b)
except:
    OP_ADM=organize_OP(OP_AMS)
    OR_ADM=get_OR_ADM()
    SOL_ADM=linalg.spsolve(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2,OR_ADM_2*OR_ADM*b)
SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM

print(time.time()-t0)

plt.close()
plt.scatter(np.array(active_nodes),np.log(np.array(perro)),3,'r')
plt.plot(np.array(active_nodes),np.log(np.array(perro)),'r')

plt.scatter(np.array(active_nodes),np.log(np.array(erro)),3,'g')
plt.plot(np.array(active_nodes),np.log(np.array(erro)),'g')
plt.savefig("l_inf2.png")
plt.close()

perms_xx=M1.mb.tag_get_data(M1.perm_tag,M1.all_volumes)[:,0]
Pseudo_ERRO_tag=M1.mb.tag_get_handle("pseudo_erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERRO_tag=M1.mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
P_TPFA_tag=M1.mb.tag_get_handle("P_TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
P_ADM_tag=M1.mb.tag_get_handle("P_ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
perm_xx_tag=M1.mb.tag_get_handle("Perm_xx", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)


# M1.mb.tag_set_data(Pseudo_ERRO_tag,M1.all_volumes,pseudo_erro[GIDs])
M1.mb.tag_set_data(ERRO_tag,M1.all_volumes,(100*abs((SOL_ADM_fina-SOL_TPFA)/SOL_TPFA)[GIDs]))
M1.mb.tag_set_data(P_ADM_tag,M1.all_volumes,SOL_ADM_fina[GIDs])
M1.mb.tag_set_data(P_TPFA_tag,M1.all_volumes,SOL_TPFA[GIDs])
M1.mb.tag_set_data(perm_xx_tag,M1.all_volumes,np.array(perms_xx))

print(max(abs((SOL_ADM_fina-SOL_TPFA)/SOL_TPFA))*100,np.linalg.norm(SOL_ADM_fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
##############################correção do fluxo########################
tcorr=time.time()
GID_by_primal1_tag=M1.mb.tag_get_handle("GID_primal1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
elems=[]
lens=[]
plver=[] #posição local do vertice
cont=0
for m in meshsets_nv1:
    vols=np.array(M1.mb.get_entities_by_handle(m))
    vert=np.uint64(np.intersect1d(vols,vertices))
    plver.append(np.where(vols==vert)[0])
    elems.append(vols)
    lens.append(len(vols))
    cont+=1
vols_by_primal1=np.uint64(np.concatenate(elems))
ids_p1=range(len(vols_by_primal1))
M1.mb.tag_set_data(GID_by_primal1_tag,vols_by_primal1,ids_p1)
IDs_primal1=M1.mb.tag_get_data(GID_by_primal1_tag,M1.all_volumes,flat=True)
Gp1=csc_matrix((np.ones(len(GIDs)),(IDs_primal1,GIDs)),shape=(len(GIDs),len(GIDs)))
T_primais1=(Gp1*T*Gp1.transpose()).tolil()
Di=[]
for g in range(len(vertices)):
    p0=np.int(np.array(lens[0:g]).sum())
    p1=p0+lens[g]
    Di.append(T_primais1[p0:p1,p0:p1])
D=scipy.sparse.block_diag(np.array(Di))
indep=T_primais1-D
P_by_primal=Gp1*SOL_ADM_fina
qs=indep*P_by_primal

pos_ver=M1.mb.tag_get_data(GID_by_primal1_tag,vertices,flat=True)
for i in range(len(vertices)):
    p0=np.int(np.array(lens[0:i]).sum())
    p1=p0+lens[i]
    Dl=Di[i]
    Dl[plver[i]]=0


print(time.time()-tcorr,"correção do fluxo")
###############################





SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
SOL_ADM_1fina=OP_ADM*SOL_ADM_1
erro_ADM1=abs((SOL_ADM_1fina-SOL_TPFA))
ERRO_adm1_tag=M1.mb.tag_get_handle("erro_adm1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(ERRO_adm1_tag,M1.all_volumes,erro_ADM1[GIDs])


SOL_AMS=linalg.spsolve(OR_AMS*T*OP_AMS,OR_AMS*b)
SOL_AMS_fina=OP_AMS*SOL_AMS
erro_AMS=abs((SOL_AMS_fina-SOL_TPFA))
ERRO_ams1_tag=M1.mb.tag_get_handle("erro_ams1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(ERRO_ams1_tag,M1.all_volumes,erro_AMS[GIDs])


l2=M1.mb.tag_get_data(fine_to_primal2_classic_tag,vertices,flat=True)
c2=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
d2=np.ones(len(l2))

OR_AMS_2=csc_matrix((d2,(l2,c2)),shape=(len(ver),len(vertices)))
SOL_AMS_2=linalg.spsolve(OR_AMS_2*OR_AMS*T*OP_AMS*OP_AMS_2,OR_AMS_2*OR_AMS*b)

aa=-(OR_AMS*T*OP_AMS)*corr*OR_AMS*b+corr*OR_AMS*b

SOL_AMS_fina_2=OP_AMS*(OP_AMS_2*SOL_AMS_2+aa.toarray().transpose()[0])
erro_AMS2=((SOL_AMS_fina_2-SOL_TPFA))
ERRO_ams2_tag=M1.mb.tag_get_handle("erro_ams2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(ERRO_ams2_tag,M1.all_volumes,erro_AMS2[GIDs])



M1.mb.write_file("testes_MAD.vtk",[av])
M1.mb.write_file("Dirichlet_MAD.vtk",[dirichlet_meshset])
M1.mb.write_file("Neumann_MAD.vtk",[neumann_meshset])
import pdb; pdb.set_trace()





import pdb; pdb.set_trace()


######################################################################
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
# finos=np.intersect1d(np.array(vertices),finos)
finos=np.concatenate([finos,refs])
if calc_TPFA:
    SOL_TPFA=linalg.spsolve(T,b)
    print("resolveu TPFA: ",time.time()-t0)
    np.save('SOL_TPFA.npy',SOL_TPFA)
'''desapagar
else:
    SOL_TPFA=np.load('SOL_TPFA.npy')
    print("leu TPFA: ")'''


finos=np.array(rng.unite(volumes_n,volumes_d))
# finos=np.setdiff1d(finos,rng.unite(volumes_n,volumes_d))
plt.close()

n1,n2=ADM_mesh(finos)
OP_ADM=organize_OP(OP_AMS)
OR_ADM=get_OR_ADM()

classic_ID=M1.mb.tag_get_data(fine_to_primal1_classic_tag,M1.all_volumes,flat=True)
ADM=M1.mb.tag_get_data(L1_ID_tag,M1.all_volumes,flat=True)
ADM1_to_AMS=dict(zip(ADM,classic_ID))

amss,l1,c1=get_raz(100,100)
amss,l1,c1=get_raz(l1,c1)

refs=np.array(vertices)[amss]
finos=np.concatenate([finos,refs])

n1,n2=ADM_mesh(finos)
OP_ADM=organize_OP(OP_AMS)
OR_ADM=get_OR_ADM()

SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
SOL_ADM_1fina=OP_ADM*SOL_ADM_1
x0,delta=itere2(SOL_ADM_1fina,8)

grs=(abs(abs(SOL_ADM_1fina-x0)[ADJs1]-abs(SOL_ADM_1fina-x0)[ADJs2])/deltas.max(axis=1))
locv1=np.where(ADJs1>=ni+nf+na)[0]
locv2=np.where(ADJs2>=ni+nf+na)[0]
locv=np.concatenate([locv1,locv2])
grv=grs[locv]
ver1=ADJs1[locv1]-ni-nf-na
ver2=ADJs2[locv2]-ni-nf-na
lines=np.concatenate([ver1,ver2])
cols=range(len(lines))
data=grv
mgv=csc_matrix((data,(lines,cols)),shape=(len(vertices),len(lines)))
grad_ver_res=mgv.max(axis=1).transpose().toarray()

# pseudo_erro=(OP_AMS*csc_matrix(grad_ver_res).transpose()).transpose().toarray()[0]


# pseudo_erro=abs((SOL_ADM_1fina-x0)/x0)
pseudo_erro=abs((SOL_ADM_1fina-x0))
#pseudo_erro=np.transpose(abs((T*csc_matrix(SOL_ADM_1fina).transpose()).toarray()))[0]
# pseudo_erro_restringido=OR_AMS*pseudo_erro
pseudo_erro_restringido=pseudo_erro[IDs_vertices]
if calc_TPFA:
    erro=abs(SOL_ADM_1fina-SOL_TPFA)/SOL_TPFA
# plt.plot(range(len(vertices)),np.sort(np.log(pseudo_erro[IDs_vertices]+0.001)),'y')
# plt.plot(range(len(vertices)),np.sort(np.log(erro[IDs_vertices]+0.001)),'b')
# plt.savefig("erro_verde,pseudo_erro_vermelho.png")
if calc_TPFA:
    print(max(erro)*100,np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
else:
    print(pseudo_erro.max()*100,np.linalg.norm(SOL_ADM_1fina-x0)/np.linalg.norm(x0),"erro max percentual e norma l2")
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

tol=0.0
#tol_n2=len(M1.all_volumes)/2
tol_n2=0.1*len(M1.all_volumes)
cont=1
continuar=True
mpante=pseudo_erro_restringido.max()
refins=0.05
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
if calc_TPFA:
    nl2.append(np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA))
    perro.append(erro.max())
else:
    nl2.append(np.linalg.norm(SOL_ADM_1fina-x0)/np.linalg.norm(x0))
    perro.append(pseudo_erro.max())
ppseudo.append(pseudo_erro.max())

while max(pseudo_erro)>tol and n2<tol_n2 and continuar:
    ###############################################################################
    print(max(pseudo_erro[IDs_vertices]),pseudo_erro.max(),"erro_nos vértices")
    finos=np.intersect1d(finos,np.array(vertices))
    if pseudo_erro_restringido.max()>0:
        if cont<3:
            nr=int(refins*(len(vertices)-len(finos))+1)
            # multip=(pseudo_erro*x0).sum()/(100*len(pseudo_erro))
            multip=(pseudo_erro).sum()/(10*len(pseudo_erro))
        else:
            nr=int(refins*(len(vertices)-len(finos))+1)
            # multip=(pseudo_erro*x0).sum()/(100*len(pseudo_erro))
            multip=(pseudo_erro).sum()/(10*len(pseudo_erro))
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
            vec=[rmax,rver,rsum]
            # vec=[rver,rmax]
            # vec=[rsum,rmax]
            # vec=[rmax]
        else:
            # vec=[rsum,rmax]
            # vec=[rmax]
            vec=[rver,rmax,rsum]
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
    # positions=np.concatenate([positions,amss])
    positions=np.unique(positions)
    print(len(positions))

    refs=np.array(vertices)[positions]
    finos=np.concatenate([finos,refs])
    n1,n2=ADM_mesh(finos)
    OP_ADM=organize_OP(OP_AMS)
    OR_ADM=get_OR_ADM()
    # while l1>0.95*raz_anterior:
    while False:
    # while l1>0.99*raz_anterior:
        ADM=M1.mb.tag_get_data(L1_ID_tag,M1.all_volumes,flat=True)
        ADM1_to_AMS=dict(zip(ADM,classic_ID))
        amss,l1,c1=get_raz(l1,c1)
        refs=np.array(vertices)[amss]
        finos=np.concatenate([finos,refs])
        n1,n2=ADM_mesh(finos)
        OP_ADM=organize_OP(OP_AMS)
        OR_ADM=get_OR_ADM()
        print(len(amss),l1,raz_anterior,"No loop do fluxo!!!")

    SOL_ADM_fina_ant=SOL_ADM_1fina
    SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
    SOL_ADM_1fina=OP_ADM*SOL_ADM_1
    sol_vers=SOL_ADM_1fina[IDs_vertices]
    sol_prol=OP_AMS*sol_vers
    x0,delta=itere(SOL_ADM_1fina,multip)

    grs=(abs(abs(SOL_ADM_1fina-x0)[ADJs1]-abs(SOL_ADM_1fina-x0)[ADJs2])/deltas.max(axis=1))
    locv1=np.where(ADJs1>=ni+nf+na)[0]
    locv2=np.where(ADJs2>=ni+nf+na)[0]
    locv=np.concatenate([locv1,locv2])
    grv=grs[locv]
    ver1=ADJs1[locv1]-ni-nf-na
    ver2=ADJs2[locv2]-ni-nf-na
    lines=np.concatenate([ver1,ver2])
    cols=range(len(lines))
    data=grv
    mgv=csc_matrix((data,(lines,cols)),shape=(len(vertices),len(lines)))
    grad_ver_res=mgv.max(axis=1).transpose().toarray()[0]
    grv_tag=M1.mb.tag_get_handle("grv_erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    M1.mb.tag_set_data(grv_tag,vertices,grad_ver_res)

    # pseudo_erro=(OP_AMS*csc_matrix(grad_ver_res).transpose()).transpose().toarray()[0]
    # pseudo_erro=abs((SOL_ADM_1fina-x0)/x0)
    pseudo_erro=abs((SOL_ADM_1fina-x0))


    # pseudo_erro=np.transpose(abs((T*csc_matrix(SOL_ADM_1fina).transpose()).toarray()))[0]

    LL1.append(l1)
    LL2.append(c1)
    lg.append(100*n2/len(M1.all_volumes))
    if calc_TPFA:
        nl2.append(np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA))
    else:
        nl2.append(np.linalg.norm(SOL_ADM_1fina-x0)/np.linalg.norm(x0))

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
    if calc_TPFA:
        erro=abs(SOL_ADM_1fina-SOL_TPFA)/SOL_TPFA
    else:
        erro=pseudo_erro
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
    if calc_TPFA:
        print(max(erro)*100,np.linalg.norm(SOL_ADM_1fina-SOL_TPFA)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
    else:
        print(max(pseudo_erro)*100,np.linalg.norm(SOL_ADM_1fina-x0)/np.linalg.norm(x0),"erro max percentual e norma l2")

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
np.savetxt('PVspe10CR553limitadorefin005.csv',x,delimiter=',')
if calc_TPFA:
    erro=abs(SOL_TPFA-SOL_ADM_1fina)/SOL_TPFA
    erro_ABS=abs(SOL_TPFA-SOL_ADM_1fina)
else:
    erro=abs(x0-SOL_ADM_1fina)/x0
    erro_abs=abs(SOL_ADM_1fina-x0)
    SOL_TPFA=x0
#print(max(erro)*100,np.linalg.norm(erro_ABS)/np.linalg.norm(SOL_TPFA),"erro max percentual e norma l2")
M1.mb.tag_set_data(Pseudo_ERRO_tag,M1.all_volumes,pseudo_erro[GIDs])
M1.mb.tag_set_data(ERRO_tag,M1.all_volumes,erro[GIDs])
M1.mb.tag_set_data(P_ADM_tag,M1.all_volumes,SOL_ADM_1fina[GIDs])
M1.mb.tag_set_data(P_TPFA_tag,M1.all_volumes,SOL_TPFA[GIDs])
M1.mb.tag_set_data(perm_xx_tag,M1.all_volumes,np.array(perms_xx))


av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)
M1.mb.write_file("testes_MAD.vtk",[av])



import pdb; pdb.set_trace()
