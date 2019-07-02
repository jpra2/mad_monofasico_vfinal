import numpy as np
import scipy.sparse as sp
from functions import f1
import time
from pymoab import types, rng
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
bifasico_sol_direta_dir = os.path.join(bifasico_dir, 'sol_direta')
bifasico_sol_multiescala_dir = os.path.join(bifasico_dir, 'sol_multiescala')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

class BifasicElems:

    def __init__(self, data_loaded, Adjs, all_centroids, all_faces_in, all_kharm, all_volumes, injectors, producers, tags, mb, volumes_d, volumes_n, ler_anterior, mtu, wirebasket_elems_nv1, mesh):
        self.cfl_ini = 0.5
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
        # ler_anterior = data_loaded['ler_anterior']
        self.Adjs = Adjs
        self.tags = tags
        self.all_centroids = mb.tag_get_data(tags['CENT'], all_volumes)
        self.all_faces_in = all_faces_in
        self.all_kharm = all_kharm
        self.map_volumes = dict(zip(all_volumes, range(len(all_volumes))))
        self.map_faces_in = dict(zip(all_faces_in, range(len(all_faces_in))))
        self.all_volumes = all_volumes
        self.wells_injector = injectors
        self.wells_producer = producers
        self.ids0 = mb.tag_get_data(tags['ID_reord_tag'], np.array(Adjs[:, 0]), flat=True)
        self.ids1 = mb.tag_get_data(tags['ID_reord_tag'], np.array(Adjs[:, 1]), flat=True)
        self.ids_volsd = mb.tag_get_data(tags['ID_reord_tag'], volumes_d, flat=True)
        self.values_d = mb.tag_get_data(tags['P'], volumes_d, flat=True)
        self.ids_volsn = mb.tag_get_data(tags['ID_reord_tag'], volumes_n, flat=True)
        self.values_n = mb.tag_get_data(tags['Q'], volumes_n, flat=True)
        self.phis = mb.tag_get_data(tags['PHI'], all_volumes, flat=True)
        self.faces_volumes = [rng.intersect(mtu.get_bridge_adjacencies(v, 3, 2), all_faces_in) for v in all_volumes]
        self.mb = mb
        self.ids_reord = self.mb.tag_get_data(self.tags['ID_reord_tag'], self.all_volumes, flat=True)
        self.map_global = dict(zip(self.all_volumes, self.ids_reord))
        self.wirebasket_elems_nv1 = wirebasket_elems_nv1
        self.mtu = mtu
        self.av = mesh.vv
        self.ADM = mesh.ADM

        v0 = all_volumes[0]
        points = mtu.get_bridge_adjacencies(v0, 3, 0)
        coords = mb.tag_get_data(tags['NODES'], points)
        maxs = coords.max(axis=0)
        mins = coords.min(axis=0)
        # hs = maxs - mins
        hs = np.array([6.0959998, 3.0479999, 0.60959998])
        yyy = mb.tag_get_data(tags['KHARM'], all_faces_in, flat=True)
        # import pdb; pdb.set_trace()
        self.hs = hs
        # self.Areas = np.array([hs[1]*hs[2], hs[0]*hs[2], hs[0]*hs[1]])
        vol = hs[0]*hs[1]*hs[2]
        mb.tag_set_data(tags['VOLUME'], all_volumes, np.repeat(vol, len(all_volumes)))

        historico = np.array(['vpi','tempo_decorrido', 'prod_agua', 'prod_oleo', 'wor', 'dt', 'loop'])
        np.save('historico', historico)
        self.delta_t = 0.0
        self.Vs = mb.tag_get_data(tags['VOLUME'], all_volumes, flat=True)
        self.V_total = float((self.Vs*self.phis).sum())
        self.vpi = 0.0
        self.hist2 = []

        self.load_sats_ini(mb, tags['SAT'])

        if ler_anterior:
            self.load_infos()
        else:
            self.set_lamb()
            self.set_mobi_faces_ini()
        pass

        phis0 = self.mb.tag_get_data(self.tags['PHI'], np.array([Adjs[:,0]]), flat=True)
        phis1 = self.mb.tag_get_data(self.tags['PHI'], np.array([Adjs[:,1]]), flat=True)
        self.phis_tempo = np.array([phis0, phis1]).max(axis=0)

    def load_sats_ini(self, mb, sat_tag):
        self.all_sats = mb.tag_get_data(sat_tag, self.all_volumes, flat=True)
        self.all_sats_ant = self.all_sats.copy()

    def pol_interp(self, S):
        # S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
        # krw = (S_temp)**(self.nw)
        # kro = (1 - S_temp)**(self.no)
        if S > (1 - self.Sor) and S <= 1:
            krw = 1.0
            kro = 0.0
        elif S < self.Swc and S >= 0:
            krw = 0.0
            kro = 1.0
        else:
            S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
            if S_temp < 0 or S_temp > 1:
                print('erro S_temp')
                import pdb; pdb.set_trace()
            krw = (S_temp)**(self.nw)
            kro = (1 - S_temp)**(self.no)

        return krw, kro

    def set_lamb(self):
        """
        seta o lambda
        """
        all_sats = self.all_sats
        all_lamb_w = np.zeros(len(self.all_volumes))
        all_lamb_o = all_lamb_w.copy()
        all_lbt = all_lamb_w.copy()
        all_fw = all_lamb_w.copy()
        all_gamav = all_lamb_w.copy()

        for i, sat in enumerate(all_sats):
            # volume = all_volumes[i]
            krw, kro = self.pol_interp(sat)
            all_lamb_w[i] = krw/self.mi_w
            all_lamb_o[i] = kro/self.mi_o
            all_lbt[i] = all_lamb_o[i] + all_lamb_w[i]
            all_fw[i] = all_lamb_w[i]/float(all_lbt[i])
            gama = (self.gama_w*all_lamb_w[i] + self.gama_o*all_lamb_o[i])/(all_lbt[i])
            all_gamav[i] = gama

        self.all_lamb_w = all_lamb_w
        self.all_lamb_o = all_lamb_o
        self.all_lbt = all_lbt
        self.all_fw = all_fw
        self.all_gamav = all_gamav

        self.mb.tag_set_data(self.tags['LAMB_W'], self.all_volumes, self.all_lamb_w)
        self.mb.tag_set_data(self.tags['LAMB_O'], self.all_volumes, self.all_lamb_o)
        self.mb.tag_set_data(self.tags['LBT'], self.all_volumes, self.all_lbt)
        self.mb.tag_set_data(self.tags['FW'], self.all_volumes, self.all_fw)
        self.mb.tag_set_data(self.tags['GAMAV'], self.all_volumes, self.all_gamav)

    def set_mobi_faces_ini(self):
        lim = 1e-5

        all_lbt = self.all_lbt
        all_centroids = self.all_centroids
        all_fw = self.all_fw
        all_sats = self.all_sats
        all_gamav = self.all_gamav
        all_kharm = self.all_kharm
        all_faces_in = self.all_faces_in
        map_volumes = self.map_volumes
        set_wells_injector = set(self.wells_injector)

        all_mobi_in_faces = np.zeros(len(all_faces_in))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        s_grav_volumes = np.zeros(len(self.all_volumes))
        Adjs = self.Adjs

        for i, face in enumerate(all_faces_in):
            elems = Adjs[i]
            kharm = all_kharm[i]
            id0 = map_volumes[elems[0]]
            id1 = map_volumes[elems[1]]
            lbt0 = all_lbt[id0]
            lbt1 = all_lbt[id1]
            fw0 = all_fw[id0]
            fw1 = all_fw[id1]
            sat0 = all_sats[id0]
            sat1 = all_sats[id1]
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            if abs(sat0-sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            if set([elems[0]]) & set_wells_injector:
                all_mobi_in_faces[i] = kharm*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
            elif set([elems[1]]) & set_wells_injector:
                all_mobi_in_faces[i] = kharm*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
            else:
                all_mobi_in_faces[i] = kharm*(lbt0 + lbt1)/2.0
                all_fw_in_face[i] = (fw0 + fw1)/2.0
                gamaf = (gama0 + gama1)/2.0
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf
            s_grav_volumes[id0] += all_s_gravs[i]
            s_grav_volumes[id1] -= all_s_gravs[i]

        self.all_mobi_in_faces = all_mobi_in_faces
        self.all_s_gravs = all_s_gravs
        self.all_fw_in_face = all_fw_in_face
        self.all_dfds = all_dfds
        self.all_gamaf = all_gamaf
        self.s_grav_volumes = s_grav_volumes
        self.fluxo_grav_volumes = s_grav_volumes

        volumes = self.all_volumes

        self.mb.tag_set_data(self.tags['MOBI_IN_FACES'], self.all_faces_in, self.all_mobi_in_faces)
        self.mb.tag_set_data(self.tags['S_GRAV'], self.all_faces_in, self.all_s_gravs)
        self.mb.tag_set_data(self.tags['FW_IN_FACES'], self.all_faces_in, self.all_fw_in_face)
        self.mb.tag_set_data(self.tags['DFDS'], self.all_faces_in, self.all_dfds)
        self.mb.tag_set_data(self.tags['GAMAF'], self.all_faces_in, self.all_gamaf)
        self.mb.tag_set_data(self.tags['S_GRAV_VOLUME'], volumes, s_grav_volumes)

        # s_grav2 = np.zeros(len(self.all_volumes))
        # cont = 0
        # passados = set()
        #
        # for v in self.all_volumes:
        #     v = int(v)
        #     faces0 = self.mtu.get_bridge_adjacencies(v, 3, 2)
        #     cent0 = self.mb.tag_get_data(self.tags['CENT'], int(v), flat=True)
        #     adjs = self.mtu.get_bridge_adjacencies(v, 2, 3)
        #     for adj in adjs:
        #         adj = int(adj)
        #         if set([adj]) & passados:
        #             continue
        #         faces1 = self.mtu.get_bridge_adjacencies(adj, 3, 2)
        #         cent1 = self.mb.tag_get_data(self.tags['CENT'], adj, flat=True)
        #         f = rng.intersect(faces0, faces1)
        #         gamaf = self.mb.tag_get_data(self.tags['GAMAF'], f, flat=True)[0]
        #         mobi = self.mb.tag_get_data(self.tags['MOBI_IN_FACES'], f, flat=True)[0]
        #         s_g = gamaf*mobi*(cent1[2] - cent0[2])
        #         s_grav2[map_volumes[v]] += s_g
        #         s_grav2[map_volumes[adj]] -= s_g
        #     passados.add(v)

    def set_mobi_faces(self, finos0=None):

        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        # finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.01
        # finos = self.mb.create_meshset()
        # self.mb.tag_set_data(self.finos_tag, 0, finos)
        # if finos0 == None:
        #     self.mb.add_entities(finos, self.wells_injector)
        #     self.mb.add_entities(finos, self.wells_producer)
        # else:
        #     self.mb.add_entities(finos, finos0)

        # map_volumes = dict(zip(volumes, range(len(volumes))))
        # all_lbt = self.mb.tag_get_data(self.lbt_tag, volumes, flat=True)
        # all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        # all_fws = self.mb.tag_get_data(self.fw_tag, volumes, flat=True)
        # all_centroids = self.mb.tag_get_data(self.cent_tag, volumes)
        # all_gamav = self.mb.tag_get_data(self.gamav_tag, volumes, flat=True)
        #
        # all_flux_in_faces = self.mb.tag_get_data(self.flux_in_faces_tag, faces, flat=True)
        # all_kharm = self.mb.tag_get_data(self.kharm_tag, faces, flat=True)
        # all_mobi_in_faces = np.zeros(len(faces))
        # all_s_gravs = all_mobi_in_faces.copy()
        # all_fw_in_face = all_mobi_in_faces.copy()
        # all_dfds = all_mobi_in_faces.copy()
        # all_gamaf = all_mobi_in_faces.copy()

        all_lbt = self.all_lbt
        all_centroids = self.all_centroids
        all_fws = self.all_fw
        all_sats = self.all_sats
        all_gamav = self.all_gamav
        all_kharm = self.all_kharm
        all_faces_in = self.all_faces_in
        map_volumes = self.map_volumes
        all_faces_in = self.all_faces_in
        all_flux_in_faces = self.flux_in_faces
        map_volumes = self.map_volumes
        Adjs = self.Adjs
        faces = self.all_faces_in
        set_wells_injector = set(self.wells_injector)

        all_mobi_in_faces = np.zeros(len(all_faces_in))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        s_grav_volumes = np.zeros(len(self.all_volumes))

        for i, face in enumerate(faces):
            elems = Adjs[i]
            kharm = all_kharm[i]
            # lbt0 = self.mb.tag_get_data(self.lbt_tag, elems[0], flat=True)[0]
            # lbt1 = self.mb.tag_get_data(self.lbt_tag, elems[1], flat=True)[0]
            id0 = map_volumes[elems[0]]
            id1 = map_volumes[elems[1]]
            lbt0 = all_lbt[id0]
            lbt1 = all_lbt[id1]
            fw0 = all_fws[id0]
            fw1 = all_fws[id1]
            sat0 = all_sats[id0]
            sat1 = all_sats[id1]
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            if abs(sat0 - sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            # if abs(sat0 - sat1) > lim_sat:
            #     self.mb.add_entities(finos, elems)

            flux_in_face = all_flux_in_faces[i]
            if set([elems[0]]) & set_wells_injector:
                all_mobi_in_faces[i] = kharm*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
                # continue
            elif set([elems[1]]) & set_wells_injector:
                all_mobi_in_faces[i] = kharm*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
                # continue
            elif flux_in_face < 0:
                all_mobi_in_faces[i] = kharm*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
            else:
                all_mobi_in_faces[i] = kharm*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
            # all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf
            s_grav_volumes[id0] += all_s_gravs[i]
            s_grav_volumes[id1] -= all_s_gravs[i]

        self.all_mobi_in_faces = all_mobi_in_faces
        self.all_s_gravs = all_s_gravs
        self.all_fw_in_face = all_fw_in_face
        self.all_dfds = all_dfds
        self.all_gamaf = all_gamaf
        self.s_grav_volumes = s_grav_volumes
        self.fluxo_grav_volumes = s_grav_volumes

        volumes = self.all_volumes

        self.mb.tag_set_data(self.tags['MOBI_IN_FACES'], self.all_faces_in, self.all_mobi_in_faces)
        self.mb.tag_set_data(self.tags['S_GRAV'], self.all_faces_in, self.all_s_gravs)
        self.mb.tag_set_data(self.tags['FW_IN_FACES'], self.all_faces_in, self.all_fw_in_face)
        self.mb.tag_set_data(self.tags['DFDS'], self.all_faces_in, self.all_dfds)
        self.mb.tag_set_data(self.tags['GAMAF'], self.all_faces_in, self.all_gamaf)
        self.mb.tag_set_data(self.tags['S_GRAV_VOLUME'], volumes, s_grav_volumes)

    def get_Tf_and_b(self):
        self.Tf = f1.get_Tf(self.all_mobi_in_faces, self.ids0, self.ids1, self.all_volumes)

        if self.gravity:
            self.b = -1*self.mb.tag_get_data(self.tags['S_GRAV_VOLUME'], self.wirebasket_elems_nv1, flat=True)
        else:
            self.b = np.zeros(len(self.all_volumes))

        if self.ADM:
            self.b3 = np.zeros(len(self.all_volumes))
        else:
            self.b3 = self.b.copy()
        self.Tf2, self.b2 = f1.set_boundary_dirichlet(self.Tf, self.b3, self.ids_volsd, self.values_d)
        if len(self.ids_volsn) > 0:
            self.b2 = f1.set_boundary_neuman(b, self.ids_volsn, self.values_n)

    def set_boundary_dirichlet(self, Tf, b, ids_volsd, values):
        Tf2 = Tf.copy().tolil()
        b2 = b.copy()
        t = Tf2.shape[0]
        n = len(ids_volsd)

        Tf2[ids_volsd] = sp.lil_matrix((n, t))
        Tf2[ids_volsd, ids_volsd] = np.ones(n)
        b2[ids_volsd] = values

        return Tf2, b2

    def calc_cfl_dep0(self):
        """
        cfl usando fluxo em cada volume
        """

        lim_sup = 1e40
        self.cfl = self.cfl_ini
        qs = self.flux_in_faces
        dfdss = self.all_dfds
        Adjs = self.Adjs
        all_volumes = self.all_volumes
        faces_volumes = self.faces_volumes

        delta_ts = np.zeros(len(self.all_volumes))
        phis = self.phis
        Vs = self.Vs
        map_faces = self.map_faces_in

        for i, v in enumerate(all_volumes):
            V = Vs[i]
            phi = phis[i]
            if phi == 0:
                delta_ts[i] = lim_sup
                continue
            faces = faces_volumes[i]
            ids_faces = [map_faces[f] for f in faces]
            q_faces = qs[ids_faces]
            dfdss_faces = dfdss[ids_faces]
            qmax = q_faces.max()
            ind = np.where(q_faces == qmax)[0]
            dfds = dfdss_faces[ind][0]
            if dfds == 0.0:
                dt1 = lim_sup
            else:
                qmax = abs(qmax)
                dt1 = self.cfl*(phi*V)/float(qmax*dfds)
                if dt1 < 0:
                    print('erro')
                    import pdb; pdb.set_trace()

            dfds_max = dfdss_faces.max()
            if dfds_max == 0:
                dt2 = dt1
            else:
                ind = np.where(dfdss_faces == dfds_max)[0]
                q2 = abs(q_faces[ind][0])
                dt2 = self.cfl*(phi*V)/float(q2*dfds_max)
                if dt2 < 0:
                    print('erro')
                    import pdb; pdb.set_trace()

            delta_ts[i] = min([dt1, dt2])

        self.delta_t = delta_ts.min()
        import pdb; pdb.set_trace()
        # self.flux_total_producao = self.mb.tag_get_data(self.tags['TOTAL_FLUX'], self.wells_producer, flat=True).sum()

    def calc_cfl(self):
        """
        cfl usando fluxo em cada volume
        """

        lim_sup = 1e40
        self.cfl = self.cfl_ini
        qs = self.flux_in_faces
        dfdss = self.all_dfds
        Adjs = self.Adjs
        all_volumes = self.all_volumes
        faces_volumes = self.faces_volumes

        phis = self.phis
        Vs = self.Vs[0]
        map_faces = self.map_faces_in

        q_dfds = np.absolute(qs*dfdss)
        phis = self.phis_tempo
        dts = self.cfl*(phis*Vs)/(q_dfds)

        self.delta_t = dts.min()

    def set_solutions1(self):

        self.mb.tag_set_data(self.tags['TOTAL_FLUX'], self.all_volumes, self.fluxos)
        self.mb.tag_set_data(self.tags['FLUX_W'], self.all_volumes, self.fluxos_w)
        self.mb.tag_set_data(self.tags['FLUX_IN_FACES'], self.all_faces_in, self.flux_in_faces)
        # self.mb.tag_set_data(self.tags['S_GRAV_VOLUME'], self.all_volumes, self.fluxo_grav_volumes)
        self.mb.tag_set_data(self.tags['PF'], self.all_volumes, self.Pf)

    def verificar_cfl(self, loop):
        print('entrou verificar cfl')
        erro_cfl = 1
        cfl = self.cfl
        contagem = 0

        while erro_cfl != 0:
            erro_cfl = self.calculate_sat(loop)
            if erro_cfl != 0:
                if erro_cfl == 1:
                    cfl = self.rec_cfl(cfl)
                    self.cfl = cfl

                contagem += 1
                # if cfl < cfl_min:
                #     cfl = cfl*0.1
                #     self.cfl = cfl
                #     erro_cfl = self.calculate_sat(volumes, loop)
                    # erro_cfl = False
                # else:
                #     self.cfl = cfl
            if contagem > 1000:
                print('cfl nao converge ')
                print(self.delta_t)
                print(cfl)
                import pdb; pdb.set_trace()

        print('saiu de verificar cfl \n')

    def calculate_sat(self, loop):
        """
        calcula a saturacao do passo de tempo corrente
        """
        #self.loop = loop
        volumes = self.all_volumes
        delta_sat = 0.001
        max_delta_sat = 0.6
        lim = 1e-10
        lim_qw = 9e-8
        all_qw = self.fluxos_w
        all_fis = self.phis
        all_sats = self.all_sats
        self.mb.tag_set_data(self.tags['SAT_LAST'], volumes, all_sats)
        self.all_sats_ant = all_sats.copy()
        all_volumes = self.Vs
        all_fw = self.all_fw
        all_total_flux = self.fluxos
        # all_Vs = self.mb.tag_get_data(self.volume_tag, volumes, flat=True)

        sats_2 = np.zeros(len(volumes))

        set_wells_producer = set(self.wells_producer)
        set_wells_injector = set(self.wells_injector)

        for i, volume in enumerate(volumes):
            sat1 = all_sats[i]
            V = all_volumes[i]
            if set([volume]) & set_wells_injector or sat1 == 0.8:
                sats_2[i] = sat1
                continue
            qw = all_qw[i]
            # if qw < 0 and abs(qw) < lim_qw:
            #     qw = 0.0

            # if abs(qw) < lim:
            #     sats_2[i] = sat1
            #     continue
            # if qw < -lim:
            #     print('abs(qw) > lim')
            #     print(qw)
            #     print('i')
            #     print(i)
            #     print('loop')
            #     print(loop)
            #     print('\n')
            #     import pdb; pdb.set_trace()
            #     return 1

            # else:
            #     pass

            fi = all_fis[i]
            if fi == 0.0:
                sats_2[i] = sat1
                continue
            if set([volume]) & set_wells_producer:
                fw = all_fw[i]
                flux = all_total_flux[i]
                qw_out = flux*fw
            else:
                qw_out = 0.0
                flux = None
                fw = None

            w_in = (qw - qw_out)*self.delta_t
            if w_in < 0 and abs(w_in < lim_qw):
                w_in = 0.0
            sat = sat1 + w_in/(fi*V)
            # sat = sat1 + (qw - qw_out)*(self.delta_t/(fi*V))
            # sat = sat1 + qw*(self.delta_t/(self.fimin*self.Vmin))
            # if sat1 > sat + lim:
            #     print('erro na saturacao')
            #     print('sat1 > sat')
            #     return True
            if sat > (1-self.Sor) - delta_sat and sat < ((1-self.Sor)) + delta_sat:
                sat = 1-self.Sor
            elif sat > self.Swc - delta_sat and sat < self.Swc + delta_sat:
                sat = self.Swc

            elif abs(sat - sat1) > max_delta_sat:
                return 1

            elif sat > 1-self.Sor:
                #sat = 1 - self.Sor
                print("Sat > 0.8")
                print(sat)
                print('i')
                print(i)
                print('loop')
                print(loop)
                print('\n')
                # self.delta_t = fi*V*(0.8 - sat1)/qw
                # self.mb.tag_set_data(self.sat_tag, volume, 0.8)
                # return 2
                return 1

            # elif sat > sat1 + 0.2:
            #     print('sat > sat1 + 0.2')
            #     print(f'sat: {sat}')
            #     print(f'sat1: {sat1}\n')
            #     return 1

            elif sat < self.Swc:

                print('erro2')
                return 1

            #elif sat < 0 or sat > (1 - self.Sor):
            elif sat < 0 or sat > 1:
                print('Erro: saturacao invalida')
                print('Saturacao: {0}'.format(sat))
                print('Saturacao anterior: {0}'.format(sat1))
                print('i: {0}'.format(i))
                print('fi: {0}'.format(fi))
                # print('V: {0}'.format(V))
                print('delta_t: {0}'.format(delta_t))
                print('loop: {0}'.format(loop))
                pdb.set_trace()
                return 1

            else:
                pass

            sats_2[i] = sat

        self.mb.tag_set_data(self.tags['SAT'], volumes, sats_2)
        self.all_sats = sats_2.copy()
        return 0

    def rec_cfl(self, cfl):
        k = 0.5
        cfl = k*cfl
        print('novo cfl', cfl)
        # qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, self.all_faces_in, flat=True)).max()
        # dfdsmax = self.mb.tag_get_data(self.dfds_tag, self.all_faces_in, flat=True).max()
        # self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.delta_t *= k
        # vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        # self.vpi += vpi
        return cfl

    def get_hist(self, t, dt, loop):
        self.flux_total_prod = self.mb.tag_get_data(self.tags['TOTAL_FLUX'], self.wells_producer, flat=True)
        fws = self.mb.tag_get_data(self.tags['FW'], self.wells_producer, flat=True)

        qw = (self.flux_total_prod*fws).sum()*self.delta_t
        qo = (self.flux_total_prod.sum() - qw)*self.delta_t
        wor = qw/float(qo)
        vpi = (self.flux_total_prod.sum()*self.delta_t)/self.V_total
        self.vpi += vpi
        self.hist = np.array([self.vpi, t, qw, qo, wor, dt, loop])
        self.hist2.append(self.hist)

    def print_hist(self, loop):
        name = 'historico_' + str(loop)
        np.save(name, self.hist)
        name2 = 'historico2_' + str(loop)
        np.save(name2, np.array(self.hist2))
        self.hist2 = []

    def load_infos(self):
        self.all_lamb_w = self.mb.tag_get_data(self.tags['LAMB_W'], self.all_volumes, flat=True)
        self.all_lamb_o = self.mb.tag_get_data(self.tags['LAMB_O'], self.all_volumes, flat=True)
        self.all_lbt = self.mb.tag_get_data(self.tags['LBT'], self.all_volumes, flat=True)
        self.all_fw = self.mb.tag_get_data(self.tags['FW'], self.all_volumes, flat=True)
        self.all_gamav = self.mb.tag_get_data(self.tags['GAMAV'], self.all_volumes, flat=True)
        self.all_mobi_in_faces = self.mb.tag_get_data(self.tags['MOBI_IN_FACES'], self.all_faces_in, flat=True)
        self.all_s_gravs = self.mb.tag_get_data(self.tags['S_GRAV'], self.all_faces_in, flat=True)
        self.all_fw_in_face = self.mb.tag_get_data(self.tags['FW_IN_FACES'], self.all_faces_in, flat=True)
        self.all_dfds = self.mb.tag_get_data(self.tags['DFDS'], self.all_faces_in, flat=True)
        self.all_gamaf = self.mb.tag_get_data(self.tags['GAMAF'], self.all_faces_in, flat=True)
        self.s_grav_volumes = self.mb.tag_get_data(self.tags['S_GRAV_VOLUME'], self.all_volumes, flat=True)
