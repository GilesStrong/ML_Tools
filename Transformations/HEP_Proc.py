import pandas
import numpy as np

def move_to_cartesian(in_data, particle, z=True, drop=False):
    try:
        pt = in_data.loc[in_data.index[:], particle + "_pT"]
        pt_name = particle + "_pT"
    except KeyError:
        pt = in_data.loc[in_data.index[:], particle + "_pt"]
        pt_name = particle + "_pt"

    if z: 
        eta = in_data.loc[in_data.index[:], particle + "_eta"]  

    phi = in_data.loc[in_data.index[:], particle + "_phi"]

    in_data[particle + '_px'] = pt*np.cos(phi)
    in_data[particle + '_py'] = pt*np.sin(phi)
    if z: 
        in_data[particle + '_pz'] = pt*np.sinh(eta)

    if drop:
        in_data.drop(columns=[pt_name, particle + "_phi"], inplace=True)
        if z:
            in_data.drop(columns=[particle + "_eta"], inplace=True)
        
def move_to_pt_eta_phi(in_data, particle):
    px = in_data.loc[in_data.index[:], particle + "_px"]
    py = in_data.loc[in_data.index[:], particle + "_py"]
    if 'mPT' not in particle: 
        pz = in_data.loc[in_data.index[:], particle + "_pz"]  

    in_data[particle + '_pT'] = np.sqrt(np.square(px)+np.square(py))

    if 'mPT' not in particle: 
        in_data[particle + '_eta'] = np.arcsinh(pz/in_data.loc[in_data.index[:], particle + '_pT'])

    in_data[particle + '_phi'] = np.arcsin(py/in_data.loc[in_data.index[:], particle + '_pT'])
    in_data.loc[(in_data[particle + "_px"] < 0) & (in_data[particle + "_py"] > 0), particle + '_phi'] = \
            np.pi - in_data.loc[(in_data[particle + "_px"] < 0) & (in_data[particle + "_py"] > 0), particle + '_phi']
    in_data.loc[(in_data[particle + "_px"] < 0) & (in_data[particle + "_py"] < 0), particle + '_phi'] = \
            -1 * (np.pi + in_data.loc[(in_data[particle + "_px"] < 0) & (in_data[particle + "_py"] < 0), particle + '_phi'])         
    in_data.loc[(in_data[particle + "_px"] < 0) & (in_data[particle + "_py"] == 0), particle + '_phi'] = \
            np.random.choice([-1*np.pi, np.pi], in_data[(in_data[particle + "_px"] < 0) & (in_data[particle + "_py"] == 0)].shape[0])
    
def delta_phi(a, b):
    return np.sign(b-a)*(np.pi - np.abs(np.abs(a-b) - np.pi))

def twist(dphi, deta):
    return np.arctan(np.abs(dphi/deta))

def add_abs_mom(in_data, particle, z=True):
    if z:
        in_data[particle + '_|p|'] = np.sqrt(np.square(in_data.loc[in_data.index[:], particle + '_px']) +
                                            np.square(in_data.loc[in_data.index[:], particle + '_py']) +
                                            np.square(in_data.loc[in_data.index[:], particle + '_pz']))
    else:
        in_data[particle + '_|p|'] = np.sqrt(np.square(in_data.loc[in_data.index[:], particle + '_px']) +
                                            np.square(in_data.loc[in_data.index[:], particle + '_py']))

def add_energy(in_data, particle):
    if particle + '_|p|' not in in_data.columns:
        add_abs_mom(in_data, particle)

    in_data[particle + '_E'] = np.sqrt(np.square(in_data.loc[in_data.index[:], particle + '_mass']) +
                                      np.square(in_data.loc[in_data.index[:], particle + '_|p|']))

def add_mt(in_data, pT, phi, name):
    in_data[name + '_mT'] = np.sqrt(2 * pT * in_data['mPT_pT'] * (1 - np.cos(delta_phi(phi, in_data['mPT_phi']))))