from utils.others_utils import OtherUtils as oth
import numpy as np
import scipy.sparse as sp

class SolDireta:

    def __init__(self):
        pass

    def solucao_pressao(self, Tf2, b2, loop, Tf):
        self.Pf = oth.get_solution(Tf2, b2)
        return self.Pf

    def calculate_total_flux(self, ids0, ids1, mobi_in_faces, s_grav_f, Pf, fw_in_faces, volumes, gravity):

        n = len(volumes)
        n2 = len(mobi_in_faces)

        if gravity:
            all_sgravs = s_grav_f
        else:
            all_sgravs = np.zeros(n2)

        # fluxo nas faces
        flux_in_faces = (Pf[ids1] - Pf[ids0])*(mobi_in_faces) + all_sgravs
        fw_in_faces = flux_in_faces*fw_in_faces

        lines = []
        cols = []
        data = []

        # fluxo total nos volumes
        lines.append(ids0)
        cols.append(np.zeros(n2))
        data.append(flux_in_faces)
        lines.append(ids1)
        cols.append(np.zeros(n2))
        data.append(-flux_in_faces)
        lines = np.concatenate(lines)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        flux_volumes = np.array(sp.csc_matrix((data, (lines, cols)), shape=(n, 1)).todense()).flatten()

        # fluxo de agua nos volumes
        data = []
        data.append(fw_in_faces)
        data.append(-fw_in_faces)
        data = np.concatenate(data)

        fluxos_w_volumes = np.array(sp.csc_matrix((data, (lines, cols)), shape=(n, 1)).todense()).flatten()

        # # fluxo de gravidade no volumes
        # data = []
        # data.append(s_grav_f)
        # data.append(-s_grav_f)
        # data = np.concatenate(data)
        #
        # s_grav_volumes = np.array(sp.csc_matrix((data, (lines, cols)), shape=(n, 1)).todense()).flatten()

        return flux_volumes, fluxos_w_volumes, flux_in_faces, fw_in_faces
