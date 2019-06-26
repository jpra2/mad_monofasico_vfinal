import numpy as np
import scipy.sparse as sp
from utils.others_utils import OtherUtils as oth
from utils import prolongation_ams as prolongation
from pymoab import types, rng


class SolAdm:
    def __init__(self, mb, wirebasket_elems, wirebasket_numbers, tags, all_volumes, faces_adjs_by_dual, intern_adjs_by_dual):
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

        self.get_AMS_to_ADM_dict()
        self.get_COL_TO_ADM_2()
        self.get_G_nv1()
        self.get_OR1_AMS()
        self.get_OR2_AMS()

    def get_AMS_to_ADM_dict(self):

        ID_ADM = self,mb.tag_get_data(self.tags['l1_ID'], self.vertices, flat=True)
        ID_AMS = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL1_CLASSIC'], self.vertices, flat=True)
        self.AMS_TO_ADM = dict(zip(ID_AMS, ID_ADM))

    def get_COL_TO_ADM_2(self):

        ID_AMS = mb.tag_get_data(self.tags['FINE_TO_PRIMAL2_CLASSIC'], self.ver, flat=True)
        ID_ADM = mb.tag_get_data(self.tags['l2_ID'], self.ver, flat=True)
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
            ID_AMS = int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
            lines.append(i)
            cols.append(ID_AMS)
            data.append(1)

        i=0
        for i in range(nfac):
            v=fac[i]
            ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
            lines.append(nint+i)
            cols.append(ID_AMS)
            data.append(1)
            #G[nint+i][ID_AMS]=1
        i=0
        for i in range(nare):
            v=are[i]
            ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
            lines.append(nint+nfac+i)
            cols.append(ID_AMS)
            data.append(1)
            #G[nint+nfac+i][ID_AMS]=1
        i=0

        for i in range(nver):
            v=ver[i]
            ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
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
        self.OR2_AMS = sp.csc_matrix((data,(lines,cols)),shape=(nver,self.nv))

    def get_OR1_AMS(self):

        for v in all_volumes:
        elem_Global_ID = self.mb.tag_get_data(self.tags['ID_reord_tag'], self.all_volumes, flat=True)
        AMS_ID = self.mb.tag_get_data(self.tags['FINE_TO_PRIMAL1_CLASSIC'], self.all_volumes, flst=True)
        lines = AMS_ID
        cols = elem_Global_ID
        data = np.ones(len(self.all_volumes))
        self.OR1_AMS = sp.csc_matrix((data,(lines,cols)),shape=(len(self.vetices),len(self.all_volumes)))

    def get_As(self, Tf):
        self.As = oth.get_Tmod_by_sparse_wirebasket_matrix(Tf, self.wirebasket_numbers[0])

    def get_OP1_AMS(self):
        self.OP1_AMS = prolongation. get_op_AMS_TPFA_top(self.mb, self.faces_adjs_by_dual, self.intern_adjs_by_dual, self.ni, self.nf, self.tags['MOBI_IN_FACES'], self.As)

    def get_OP2_AMS(self):

    def get_n1adm_and_n2adm_and_elemsnv0(self):
        self.n1_adm = self.mb.tag_get_data(tags_1['l1_ID'], all_volumes, flat=True).max() + 1
        self.n2_adm = self.mb.tag_get_data(tags_1['l2_ID'], all_volumes, flat=True).max() + 1
        self.elems_nv0 = self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([self.tags['l3_ID']]), np.array([1]))

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

        m=find(OP1)
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

        opad1=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),n1_adm))
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
        OR_ADM=csc_matrix((data,(lines,cols)),shape=(n1_adm,len(all_volumes)))

        # return OP_ADM, OR_ADM
        self.OP1_ADM = OP_ADM
        self.OR1_ADM = OR_ADM

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
                            id_ADM=self.COL_TO_ADM_2[str(i)]
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

    def solve_ADM_system(self, Tf2, b2):
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
        mb.tag_set_data(tags['PMS2'], wirebasket_elems_nv0, Pms2)
