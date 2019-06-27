import numpy as np
import scipy.sparse as sp
from utils.others_utils import OtherUtils as oth
from utils import prolongation_ams as prolongation
from pymoab import types, rng
import time


class SolAdm:

    def __init__(self, mb, wirebasket_elems, wirebasket_numbers, tags, all_volumes, faces_adjs_by_dual, intern_adjs_by_dual, mv, mtu, mesh):
        self.mb = mb
        self.wirebasket_elems = wirebasket_elems
        self.wirebasket_numbers = wirebasket_numbers
        self.tags = tags
        self.vertices = wirebasket_elems[0][3]
        self.ver = wirebasket_elems[1][3]
        self.all_volumes = all_volumes
        self.faces_adjs_by_dual = faces_adjs_by_dual
        self.intern_adjs_by_dual = intern_adjs_by_dual
        self.ni = len(self.wirebasket_elems[0][0])
        self.nf = len(self.wirebasket_elems[0][1])
        self.wirebasket_elems_nv0 = np.concatenate(wirebasket_elems[0])
        self.mv = mv
        self.mtu = mtu
        self.gravity = mesh.gravity
        self.all_boundary_faces = mesh.all_boundary_faces
        self.bound_faces_nv = mesh.bound_faces_nv
        self.all_faces_in = mesh.all_faces_in
        self.av = mesh.vv

        meshset_vertices_nv2 = mb.create_meshset()
        mb.add_entities(meshset_vertices_nv2, wirebasket_elems[1][3])
        self.mv2 = meshset_vertices_nv2

        self.get_AMS_to_ADM_dict()
        self.get_COL_TO_ADM_2()
        self.get_G_nv1()
        self.get_OR1_AMS()
        self.get_OR2_AMS()
        self.get_n1adm_and_n2adm_and_finos0()

    def get_AMS_to_ADM_dict(self):

        ID_ADM = self.mb.tag_get_data(self.tags['l1_ID'], self.vertices, flat=True)
        ID_AMS = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL1_CLASSIC'], self.vertices, flat=True)
        self.AMS_TO_ADM = dict(zip(ID_AMS, ID_ADM))

    def get_COL_TO_ADM_2(self):

        ID_AMS = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL2_CLASSIC'], self.ver, flat=True)
        ID_ADM = self.mb.tag_get_data(self.tags['l2_ID'], self.ver, flat=True)
        self.COL_TO_ADM_2 = dict(zip(ID_AMS, ID_ADM))

    def get_G_nv1(self):
        nint = len(self.wirebasket_elems[1][0])
        inte = self.wirebasket_elems[1][0]
        nfac = len(self.wirebasket_elems[1][1])
        fac = self.wirebasket_elems[1][1]
        nare = len(self.wirebasket_elems[1][2])
        are = self.wirebasket_elems[1][2]
        nver = len(self.wirebasket_elems[1][3])
        ver = self.wirebasket_elems[1][3]
        fine_to_primal1_classic_tag = self.tags['FINE_TO_PRIMAL1_CLASSIC']
        nv = len(self.vertices)

        lines = []
        cols = []
        data = []

        for i in range(nint):
            v=inte[i]
            ID_AMS = int(self.mb.tag_get_data(fine_to_primal1_classic_tag,int(v)))
            lines.append(i)
            cols.append(ID_AMS)
            data.append(1)

        i=0
        for i in range(nfac):
            v=fac[i]
            ID_AMS=int(self.mb.tag_get_data(fine_to_primal1_classic_tag,int(v)))
            lines.append(nint+i)
            cols.append(ID_AMS)
            data.append(1)
            #G[nint+i][ID_AMS]=1
        i=0
        for i in range(nare):
            v=are[i]
            ID_AMS=int(self.mb.tag_get_data(fine_to_primal1_classic_tag,int(v)))
            lines.append(nint+nfac+i)
            cols.append(ID_AMS)
            data.append(1)
            #G[nint+nfac+i][ID_AMS]=1
        i=0

        for i in range(nver):
            v=ver[i]
            ID_AMS=int(self.mb.tag_get_data(fine_to_primal1_classic_tag,int(v)))
            lines.append(nint+nfac+nare+i)
            cols.append(ID_AMS)
            data.append(1)
            #G[nint+nfac+nare+i][ID_AMS]=1
        self.G = sp.csc_matrix((data,(lines,cols)),shape=(nv, nv))

    def get_OR2_AMS(self):

        ID_AMS_1 = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL1_CLASSIC'], self.vertices, flat=True)
        ID_AMS_2 = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL2_CLASSIC'], self.vertices, flat=True)
        lines = ID_AMS_2
        cols = ID_AMS_1
        data = np.ones(len(self.vertices))
        nver = self.wirebasket_numbers[1][3]
        nv = self.wirebasket_numbers[0][3]
        self.OR2_AMS = sp.csc_matrix((data,(lines,cols)),shape=(nver,nv))

    def get_OR1_AMS(self):

        elem_Global_ID = self.mb.tag_get_data(self.tags['ID_reord_tag'], self.all_volumes, flat=True)
        AMS_ID = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL1_CLASSIC'], self.all_volumes, flat=True)
        lines = AMS_ID
        cols = elem_Global_ID
        data = np.ones(len(self.all_volumes))
        self.OR1_AMS = sp.csc_matrix((data,(lines,cols)),shape=(len(self.vertices),len(self.all_volumes)))

    def get_As(self, Tf):
        self.As = oth.get_Tmod_by_sparse_wirebasket_matrix(Tf, self.wirebasket_numbers[0])

    def get_OP1_AMS(self):
        self.OP1_AMS = prolongation.get_op_AMS_TPFA_top(self.mb, self.faces_adjs_by_dual, self.intern_adjs_by_dual, self.ni, self.nf, self.tags['MOBI_IN_FACES'], self.As)

    def get_n1adm_and_n2adm_and_finos0(self):
        tags_1 = self.tags
        all_volumes = self.all_volumes

        self.n1_adm = self.mb.tag_get_data(tags_1['l1_ID'], all_volumes, flat=True).max() + 1
        self.n2_adm = self.mb.tag_get_data(tags_1['l2_ID'], all_volumes, flat=True).max() + 1
        self.finos0 = self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([self.tags['l3_ID']]), np.array([1]))

    def get_n1adm_and_n2adm(self):
        tags_1 = self.tags
        all_volumes = self.all_volumes

        self.n1_adm = self.mb.tag_get_data(tags_1['l1_ID'], all_volumes, flat=True).max() + 1
        self.n2_adm = self.mb.tag_get_data(tags_1['l2_ID'], all_volumes, flat=True).max() + 1

    def organize_OP1_ADM(self):
        OP1_AMS = self.OP1_AMS
        all_volumes = self.all_volumes
        dict_tags = self.tags
        mb = self.mb
        AMS_TO_ADM = self.AMS_TO_ADM
        n1_adm = self.n1_adm

        OP1_AMS = OP1_AMS.tolil()
        nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['l3_ID']]), np.array([1]))

        ta1 = time.time()
        OP1 = OP1_AMS.copy()
        OP1 = OP1.tolil()
        ID_global1 = mb.tag_get_data(dict_tags['ID_reord_tag'],nivel_0, flat=True)
        OP1[ID_global1]=sp.csc_matrix((1,OP1.shape[1]))
        gids_nv1_adm = mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True)

        IDs_ADM1=mb.tag_get_data(dict_tags['l1_ID'],nivel_0, flat=True)

        m=sp.find(OP1)
        l1=m[0]
        c1=m[1]
        d1=m[2]
        lines=ID_global1
        cols=IDs_ADM1
        data=np.repeat(1,len(lines))

        # ID_ADM1=[AMS_TO_ADM[str(k)] for k in c1]
        ID_ADM1=[AMS_TO_ADM[k] for k in c1]
        lines=np.concatenate([lines,l1])
        cols=np.concatenate([cols,ID_ADM1])
        data=np.concatenate([data,d1])

        opad1=sp.csc_matrix((data,(lines,cols)),shape=(len(all_volumes),n1_adm))
        print("opad1",time.time()-ta1)
        OP_ADM=opad1

        print("set_nivel 0")

        # lines = []
        # cols = []
        # data = []

        # for v in all_volumes:
        #     elem_Global_ID = int(mb.tag_get_data(dict_tags['ID_reord_tag'], v, flat=True))
        #     elem_ID1 = int(mb.tag_get_data(dict_tags['l1_ID'], v, flat=True))
        #     lines.append(elem_ID1)
        #     cols.append(elem_Global_ID)
        #     data.append(1)
        #     #OR_ADM[elem_ID1][elem_Global_ID]=1
        cols = mb.tag_get_data(dict_tags['ID_reord_tag'], all_volumes, flat=True)
        lines = mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True)
        data = np.ones(len(lines))
        OR_ADM=sp.csc_matrix((data,(lines,cols)),shape=(n1_adm,len(all_volumes)))

        # return OP_ADM, OR_ADM
        self.OP1_ADM = OP_ADM
        self.OR1_ADM = OR_ADM

    def organize_OP1_ADM_v2(self):
        PAD = self.OP1_AMS

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

    def organize_OP2_ADM(self, OP2_AMS):
        t0 = time.time()

        mb = self.mb
        all_volumes = self.all_volumes
        dict_tags = self.tags
        n1_adm = self.n1_adm
        n2_adm = self.n2_adm


        lines=[]
        cols=[]
        data=[]

        lines_or=[]
        cols_or=[]
        data_or=[]

        My_IDs_2=[]
        for v in all_volumes:
            ID_global=int(mb.tag_get_data(dict_tags['l1_ID'],v))
            if ID_global not in My_IDs_2:
                My_IDs_2.append(ID_global)
                ID_ADM=int(mb.tag_get_data(dict_tags['l2_ID'],v))
                nivel=mb.tag_get_data(dict_tags['l3_ID'],v)
                d1=mb.tag_get_data(dict_tags['d2'],v)
                ID_AMS = int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL1_CLASSIC'], v))
                # nivel<3 refere-se aos volumes na malha fina (nivel=1) e intermédiária (nivel=2)
                # d1=3 refere-se aos volumes que são vértice na malha dual de grossa
                if nivel<3:
                    lines.append(ID_global)
                    cols.append(ID_ADM)
                    data.append(1)
                    lines_or.append(ID_ADM)
                    cols_or.append(ID_global)
                    data_or.append(1)
                    #OP_ADM_2[ID_global][ID_ADM]=1
                else:
                    lines_or.append(ID_ADM)
                    cols_or.append(ID_global)
                    data_or.append(1)
                    for i in range(OP2_AMS[ID_AMS].shape[1]):
                        p=OP2_AMS[ID_AMS, i]
                        if p>0:
                            # id_ADM=self.COL_TO_ADM_2[str(i)]
                            id_ADM=self.COL_TO_ADM_2[i]
                            lines.append(ID_global)
                            cols.append(id_ADM)
                            data.append(float(p))
                            #OP_ADM_2[ID_global][id_ADM]=p

        print(time.time()-t0,"organize OP_ADM_2_______________________________::::::::::::")
        OP_ADM_2 = sp.csc_matrix((data,(lines,cols)),shape=(n1_adm,n2_adm))
        OR_ADM_2 = sp.csc_matrix((data_or,(lines_or,cols_or)),shape=(n2_adm,n1_adm))

        return OP_ADM_2, OR_ADM_2

    def get_OPs2_ADM(self):
        n1_adm = self.n1_adm
        n2_adm = self.n2_adm

        if n1_adm == n2_adm:
            OP2_ADM = sp.identity(n1_adm)
            OR2_ADM = sp.identity(n1_adm)

        else:
            T1_AMS = self.OR1_AMS.dot(self.As['Tf'])
            T1_AMS = T1_AMS.dot(self.OP1_AMS)
            W_AMS = self.G.dot(T1_AMS)
            W_AMS = W_AMS.dot(self.G.transpose())
            OP2_AMS = oth.get_op_by_wirebasket_Tf_wire_coarse(W_AMS, self.wirebasket_numbers[1])
            OP2_AMS = self.G.transpose().dot(OP2_AMS)
            OP2_ADM, OR2_ADM = self.organize_OP2_ADM(OP2_AMS)

        self.OP2_ADM = OP2_ADM
        self.OR2_ADM = OR2_ADM

    def get_ops_ADM_seq(self):
        self.get_OP1_AMS()
        self.organize_OP1_ADM()
        self.get_OPs2_ADM()

    def solucao_pressao(self, Tf2, b2, loop, Tf):
        self.get_As(Tf)
        self.get_ops_ADM_seq()
        self.solve_adm_system(Tf2, b2)

        return self.Pf

    def solve_adm_system(self, Tf2, b2):
        OR1_ADM = self.OR1_ADM
        OP1_ADM = self.OP1_ADM
        OR2_ADM = self.OR2_ADM
        OP2_ADM = self.OP2_ADM
        mb = self.mb
        tags = self.tags
        wirebasket_elems_nv0 = self.wirebasket_elems_nv0

        T1_ADM = OR1_ADM.dot(Tf2)
        T1_ADM = T1_ADM.dot(OP1_ADM)
        b1_ADM = OR1_ADM.dot(b2)
        T1_ADM = T1_ADM.tocsc()
        T2_ADM = OR2_ADM.dot(T1_ADM)
        T2_ADM = T2_ADM.dot(OP2_ADM)
        b2_ADM = OR2_ADM.dot(b1_ADM)
        T2_ADM = T2_ADM.tocsc()
        PC2_ADM = oth.get_solution(T2_ADM, b2_ADM)
        Pms2 = OP2_ADM.dot(PC2_ADM)
        Pms2 = OP1_ADM.dot(Pms2)
        self.Pf = Pms2
        mb.tag_set_data(tags['PMS2'], wirebasket_elems_nv0, Pms2)

    def calculate_total_flux(self, ids0, ids1, mobi_in_faces, s_grav_f, Pf, fw_in_faces, volumes, gravity):
        tags_1 = self.tags
        mb = self.mb
        meshset_vertices = self.mv
        meshset_vertices_nv2 = self.mv2

        elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([1]))
        vertices_nv1 = mb.get_entities_by_type_and_tag(meshset_vertices, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([2]))
        vertices_nv2 = mb.get_entities_by_type_and_tag(meshset_vertices_nv2, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([3]))
        k = 0
        self.calc_pcorr_nv(vertices_nv1, k, tags_1['FINE_TO_PRIMAL1_CLASSIC'])
        k = 1
        self.calc_pcorr_nv(vertices_nv2, k, tags_1['FINE_TO_PRIMAL2_CLASSIC'])

        

        fluxos = np.zeros(len(self.all_volumes))
        fluxos_w = np.zeros(len(self.all_volumes))
        fluxo_grav_volumes = np.zeros(len(self.all_volumes))
        flux_in_faces = np.zeros(len(self.all_faces_in))
        flux_w_in_faces = np.zeros(len(self.all_faces_in))

        import pdb; pdb.set_trace()

        return fluxos, fluxos_w, flux_in_faces, flux_w_in_faces, fluxo_grav_volumes

    def calc_pcorr_nv(self, vertices_nv, k, tag_primal_classic):
        tags_1 = self.tags
        mb = self.mb
        mtu = self.mtu
        boundary_faces = self.all_boundary_faces
        bound_faces_nv = self.bound_faces_nv

        for vert in vertices_nv:
            t00 = time.time()
            primal_id = mb.tag_get_data(tag_primal_classic, vert, flat=True)[0]
            elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tag_primal_classic]), np.array([primal_id]))
            faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
            faces = rng.subtract(faces, boundary_faces)
            faces_boundary = rng.intersect(faces, bound_faces_nv[k])
            if len(faces_boundary) < 1:
                import pdb; pdb.set_trace()
            t01 = time.time()
            t02 = time.time()
            self.calculate_pcorr(elems_in_meshset, vert, faces_boundary, faces, k)
            # bif_utils.calculate_pcorr_v4(elems_in_meshset, tags_1['PCORR2'], tags_1)
            # t03 = time.time()
            # bif_utils.set_flux_pms_meshsets(elems_in_meshset, faces, faces_boundary, tags_1['PMS2'], tags_1['PCORR2'])
            # t04 = time.time()
            # dt0 = t01 - t00
            # dt1= t03 - t02
            # dt2= t04 - t03
            # dtt = t04 - t01

    def calculate_pcorr(self, elems_in_meshset, vertice, faces_boundary, faces, k):

        mb = self.mb
        mobi_in_faces_tag = self.tags['MOBI_IN_FACES']
        pms_tag = self.tags['PMS2']
        s_grav_tag = self.tags['S_GRAV']
        pcorr_tag = self.tags['PCORR2']

        allmobis = mb.tag_get_data(mobi_in_faces_tag, faces, flat=True)
        try:
            vols3 = self.mtu.get_bridge_adjacencies(faces_boundary, 2, 3)
        except:
            import pdb; pdb.set_trace()

        vols_inter = rng.subtract(vols3, elems_in_meshset)
        pms_vols3 = self.mb.tag_get_data(pms_tag, vols3, flat=True)
        map_pms_vols3 = dict(zip(vols3, pms_vols3))
        del pms_vols3

        volumes_2 = self.mtu.get_bridge_adjacencies(elems_in_meshset, 2, 3)
        if self.gravity:
            s_gravs = mb.tag_get_data(s_grav_tag, faces, flat=True)
        else:
            s_gravs = np.zeros(len(faces))
        n = len(elems_in_meshset)

        map_local = dict(zip(elems_in_meshset, range(n)))
        lines = []
        cols = []
        data = []
        b = np.zeros(n)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]
        faces_in = rng.subtract(faces, faces_boundary)
        map_id_faces = dict(zip(faces, range(len(faces))))

        for face in faces_in:
            id_face = map_id_faces[face]
            mobi = allmobis[id_face]
            s_g = -s_gravs[id_face]
            id_face = map_id_faces[face]
            elem0 = Adjs[id_face][0]
            elem1 = Adjs[id_face][1]
            id0 = map_local[elem0]
            id1 = map_local[elem1]
            b[id0] += s_g
            b[id1] -= s_g
            lines += [id0, id1]
            cols += [id1, id0]
            data += [mobi, mobi]

        for face in faces_boundary:
            id_face = map_id_faces[face]
            mobi = allmobis[id_face]
            s_g = -s_gravs[id_face]
            elem0 = Adjs[id_face][0]
            elem1 = Adjs[id_face][1]
            vvv = True
            try:
                id = map_local[elem0]
            except KeyError:
                id = map_local[elem1]
                vvv = False
            flux = -(map_pms_vols3[elem1] - map_pms_vols3[elem0])*mobi
            if vvv:
                b[id] += s_g + flux
            else:
                b[id] -= s_g + flux

        T = sp.csc_matrix((data,(lines,cols)),shape=(n, n))
        T = T.tolil()
        d1 = np.array(T.sum(axis=1)).reshape(1, n)[0]*(-1)
        T.setdiag(d1)

        d_vols = rng.Range(vertice)
        map_values = dict(zip(d_vols, mb.tag_get_data(pms_tag, d_vols, flat=True)))
        T, b = oth.set_boundary_dirichlet_matrix(map_local, map_values, b, T)
        x = oth.get_solution(T, b)
        mb.tag_set_data(pcorr_tag, elems_in_meshset, x)
