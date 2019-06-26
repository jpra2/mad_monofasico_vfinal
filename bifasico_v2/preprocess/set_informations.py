from pymoab import types, rng


def injector_producer_press(mb, gama_w, gama_o, gravity, all_nodes, volumes_d, tags):

    press_tag = tags['P']
    values = mb.tag_get_data(press_tag, volumes_d, flat=True)
    wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    tags['WELLS_INJECTOR'] = wells_injector_tag
    tags['WELLS_PRODUCER'] = wells_producer_tag
    wells_injector_meshset = mb.create_meshset()
    wells_producer_meshset = mb.create_meshset()
    m = values.mean()
    injectors = []
    producers = []
    for i, v in enumerate(values):
        if v > m:
            injectors.append(volumes_d[i])
        else:
            producers.append(volumes_d[i])

    producers = rng.Range(producers)
    injectors = rng.Range(injectors)
    mb.add_entities(wells_producer_meshset, producers)
    mb.add_entities(wells_injector_meshset, injectors)
    mb.tag_set_data(wells_injector_tag, 0, wells_injector_meshset)
    mb.tag_set_data(wells_producer_tag, 0, wells_producer_meshset)

    if gravity:
        set_p_with_gravity(mb, press_tag, all_nodes, injectors, producers, gama_w, gama_o, tags)

    return injectors, producers

def load_injectors_producers(mb, gama_w, gama_o, gravity, all_nodes, volumes_d, tags):
    wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR')
    wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER')
    tags['WELLS_INJECTOR'] = wells_injector_tag
    tags['WELLS_PRODUCER'] = wells_producer_tag
    injectors = mb.tag_get_data(wells_injector_tag, 0, flat=True)[0]
    producers = mb.tag_get_data(wells_producer_tag, 0, flat=True)[0]
    injectors = mb.get_entities_by_handle(injectors)
    producers = mb.get_entities_by_handle(producers)

    if gravity:
        set_p_with_gravity(mb, tags['P'], all_nodes, injectors, producers, gama_w, gama_o, tags)

    return injectors, producers

def set_p_with_gravity(mb, press_tag, all_nodes, injectors, producers, gama_w, gama_o, tags):

    coords = mb.tag_get_data(tags['NODES'], all_nodes)
    coords = coords.reshape([len(all_nodes), 3])
    maxs = coords.max(axis=0)
    Lz = maxs[2]

    values = mb.tag_get_data(press_tag, injectors, flat=True)
    z_elems = -1*mb.tag_get_data(tags['CENT'], injectors)[:,2]
    delta_z = z_elems + Lz
    pressao = gama_w*(delta_z) + values
    mb.tag_set_data(press_tag, injectors, pressao)

    values = mb.tag_get_data(press_tag, producers, flat=True)
    z_elems = -1*mb.tag_get_data(tags['CENT'], producers)[:,2]
    delta_z = z_elems + Lz
    pressao = gama_o*(delta_z) + values
    mb.tag_set_data(press_tag, producers, pressao)

def set_p_with_gravity_2(mb, press_tag, all_nodes, injectors, producers, gama_w, gama_o, tags):

    coords = mb.tag_get_data(tags['NODES'], all_nodes)
    coords = coords.reshape([len(all_nodes), 3])
    maxs = coords.max(axis=0)
    Lz = maxs[2]

    values = mb.tag_get_data(press_tag, injectors, flat=True)
    z_elems = -1*mb.tag_get_data(tags['CENT'], injectors)[:,2]
    delta_z = z_elems + Lz
    pressao = gama_w*(delta_z) + values
    mb.tag_set_data(press_tag, injectors, pressao)

    values = mb.tag_get_data(press_tag, producers, flat=True)
    z_elems = -1*mb.tag_get_data(tags['CENT'], producers)[:,2]
    delta_z = z_elems + Lz
    pressao = gama_o*(delta_z) + values
    mb.tag_set_data(press_tag, producers, pressao)

def injector_producer(mb, gama_w, gama_o, gravity, all_nodes, volumes_d, volumes_n, tags):
    neuman_tag = tags['Q']
    press_tag = tags['P']
    wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_injector_meshset = mb.create_meshset()
    wells_producer_meshset = mb.create_meshset()
    mb.add_entities(wells_producer_meshset, volumes_d)
    mb.add_entities(wells_injector_meshset, volumes_n)
    mb.tag_set_data(wells_injector_tag, 0, wells_injector_meshset)
    mb.tag_set_data(wells_producer_tag, 0, wells_producer_meshset)

    if gravity:
        set_p_with_gravity(mb, tags['P'], all_nodes, volumes_n, volumes_d, gama_w, gama_o, tags)

    return volumes_n, volumes_d

def convert_to_SI(mb, tags, all_volumes, all_faces, all_nodes, volumes_d, volumes_n):
    from preprocess import conversao as conv
    import numpy as np
    k_pe_to_m = 1.0
    k_md_to_m2 = 1.0
    k_psi_to_pa = 1.0
    k_bbldia_to_m3seg = 1.0
    k_pe_to_m = conv.pe_to_m(k_pe_to_m)
    k_md_to_m2 = conv.milidarcy_to_m2(k_md_to_m2)
    k_psi_to_pa = conv.psi_to_Pa(k_psi_to_pa)
    k_bbldia_to_m3seg = conv.bbldia_to_m3seg(k_bbldia_to_m3seg)

    p_tag = tags['P']
    k_harm_tag = tags['KHARM']
    cent_tag = tags['CENT']

    press_values = mb.tag_get_data(tags['P'], volumes_d, flat=True)
    press_values *= k_psi_to_pa
    mb.tag_set_data(p_tag, volumes_d, press_values)

    if len(volumes_n) > 0:
        q_values = mb.tag_get_data(tags['Q'], volumes_n, flat=True)
        q_values *= k_bbldia_to_m3seg
        mb.tag_set_data(q_tag, volumes_q, q_values)

    # k_harms = mb.tag_get_data(tags['KHARM'], all_faces, flat=True)
    faces_with_kharm = mb.get_entities_by_type_and_tag(0, types.MBQUAD, np.array([k_harm_tag]), np.array([None]))
    k_harms = mb.tag_get_data(tags['KHARM'], faces_with_kharm, flat=True)

    k_harms *= k_md_to_m2*k_pe_to_m
    # mb.tag_set_data(k_harm_tag, all_faces, k_harms)
    mb.tag_set_data(k_harm_tag, faces_with_kharm, k_harms)

    centroids = (k_pe_to_m)*mb.tag_get_data(cent_tag, all_volumes)
    mb.tag_set_data(cent_tag, all_volumes, centroids)

    coords = mb.tag_get_data(tags['NODES'], all_nodes)
    mb.tag_set_data(tags['NODES'], all_nodes, coords*(k_pe_to_m))
