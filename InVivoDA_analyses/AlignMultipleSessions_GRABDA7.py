import os

import numpy as np
import scipy as sp

import suite2p as s2p
from suite2p.registration import bidiphase, utils, rigid, nonrigid

import TwoPUtils




if __name__=="__main__":
    # set paths for mouse
    base_folder = os.path.join('/mnt', 'BigDisk', '2P_scratch')
    mouse = 'GRABDA7'

    file_list = ({'date': '08_12_2020', 'scene': 'SingleMorph', 'session': 1, 'scan_number': 4},
                 {'date': '10_12_2020', 'scene': 'NeuroMods_LocationA', 'session': 3, 'scan_number': 1},
                 {'date': '12_12_2020', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan_number': 2},
                 {'date': '13_12_2020', 'scene': 'NeuroMods_Day6Image', 'session': 2, 'scan_number': 8},
                 {'date': '14_12_2020', 'scene': 'NeuroMods_LocationB', 'session': 2, 'scan_number': 5},
                 {'date': '15_12_2020', 'scene': 'NeuroMods_Day8Image', 'session': 1, 'scan_number': 2},
                 {'date': '16_12_2020', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan_number': 2},
                 {'date': '17_12_2020', 'scene': 'NM_MorphBlocks', 'session': 1, 'scan_number': 5},
                 {'date': '19_12_2020', 'scene': 'NM_RandomMorphs', 'session': 2, 'scan_number': 5},
                 {'date': '21_12_2020', 'scene': 'NM_RandomMorphs', 'session': 2, 'scan_number': 2},
                 {'date': '22_12_2020', 'scene': 'NM_Morph1ToDreamLand', 'session': 2, 'scan_number': 5},
                 {'date': '24_12_2020', 'scene': 'NM_Morph1ToDreamLand', 'session': 2, 'scan_number': 3},
                 {'date': '26_12_2020', 'scene': 'NM_DreamLandToPizzaLand', 'session': 1, 'scan_number': 3},
                 {'date': '28_12_2020', 'scene': 'NM_PizzaLandOnly', 'session': 2, 'scan_number': 2})
    for file in file_list:
        file['mouse'] = mouse

    sessions = []
    meanImgs = []
    meanImgs_chan2 = []
    for f in file_list:
        f['scanner'] = "NLW"
        f['VR_only'] = False
        f['prompt_for_keys'] = False

        scan_str = "%s_%03d_%03d" % (f['scene'], f['session'], f['scan_number'])
        try:
            source_folder = os.path.join('/media/mplitt', 'Backup Plus')
            source_stem = os.path.join(source_folder, f['mouse'], f['date'], f['scene'], scan_str)
            info = TwoPUtils.scanner_tools.sbx_utils.loadmat(source_stem + '.mat')
        except:
            source_folder = os.path.join('/media/mplitt', 'Backup Plus1', '2P_Data')
            source_stem = os.path.join(source_folder, f['mouse'], f['date'], f['scene'], scan_str)
            info = TwoPUtils.scanner_tools.sbx_utils.loadmat(source_stem + '.mat')

        f['scan_file'] = source_stem + '.sbx'
        f['scanheader_file'] = source_stem + '.mat'
        f['vr_filename'] = os.path.join("/home/mplitt/VR_scratch", f['mouse'], f['date'],
                                        "%s_%d.sqlite" % (f['scene'], f['session']))
        if f['mouse'] == 'GRABDA6':
            f['s2p_path'] = os.path.join("/home/mplitt/2P_scratch", f['mouse'], f['date'], f['scene'], scan_str,
                                         'suite2p')
        else:
            f['s2p_path'] = os.path.join("/mnt/BigDisk/2P_scratch", f['mouse'], f['date'], f['scene'], scan_str,
                                         'suite2p')

        sess = TwoPUtils.sess.Session(**f)
        sess.load_scan_info()
        sess.load_suite2p_data()
        sessions.append(sess)

        # stack mean images into vector
        meanImgs.append(sess.s2p_ops['meanImg'])
        meanImgs_chan2.append(sess.s2p_ops['meanImg_chan2'])

    comb_path = os.path.join(base_folder, mouse, "all", "suite2p", "plane0")
    os.makedirs(comb_path, exist_ok=True)
    data_bin_path = os.path.join(comb_path, "data.bin")
    data2_bin_path = os.path.join(comb_path, "data_chan2.bin")

    overwrite = False

    if ~os.path.exists(data_bin_path) and overwrite:

        with open(data_bin_path, 'wb') as f:
            pass

        with open(data2_bin_path, 'wb') as f:
            pass

        for i, (sess, dy, dx, dy1, dx1) in enumerate(zip(sessions, ymax, xmax, ymax1, xmax1)):
            print(i)

            # open single session binaries
            _data_bin = s2p.io.binary.BinaryFile(ops['Ly'], ops['Lx'], sess.s2p_ops['reg_file'])
            _data2_bin = s2p.io.binary.BinaryFile(ops['Ly'], ops['Lx'], sess.s2p_ops['reg_file_chan2'])

            # for each frame in binary
            dy1 = dy1[np.newaxis, :]
            dx1 = dx1[np.newaxis, :]

            with open(data_bin_path, 'ab') as f:
                for index, red_frames in _data_bin.iter_frames(batch_size=4000):
                    # append data to combined file
                    f.write(bytearray(red_frames.astype(np.int16)))

            with open(data2_bin_path, 'ab') as f:
                for index, green_frames in _data2_bin.iter_frames(batch_size=4000):
                    print("data2", i, index)
                    # append data to combined file
                    f.write(bytearray(green_frames.astype(np.int16)))
    else:
        pass

    data_bin = s2p.io.BinaryFile(ops['Ly'], ops['Lx'], data_bin_path)
    data2_bin = s2p.io.BinaryFile(ops['Ly'], ops['Lx'], data2_bin_path)

    combined_ops = {'data_path': [],
                    'reg_file': data_bin_path,
                    'reg_file_chan2': data_bin_path,
                    'save_path0': os.path.join(base_folder, mouse, 'all'),
                    'save_folder': "suite2p",
                    'save_path': os.path.join(base_folder, mouse, 'all', 'suite2p', 'plane0'),
                    'ops_path': os.path.join(base_folder, mouse, 'all', 'suite2p', 'plane0', 'ops.npy'),
                    'do_registration': 2,
                    'two_step_registration': True,
                    'nimg_init': 10000,
                    'align_by_chan': 2,
                    'block_size': [64, 64],
                    'maxregshiftNR': 10,
                    'nchannels': 2,
                    'tau': .7,
                    'functional_chan': 2,
                    'fs': 15.46,
                    'roidetect': True,
                    'input_format': "h5",
                    'h5py_key': 'data',
                    'sparse_mode': True,
                    'threshold_scaling': 1.,
                    'nchannels': 2,
                    'nframes': data_bin.n_frames,
                    'Ly': data_bin.shape[1],
                    'Lx': data_bin.shape[2],
                    'xrange': [50, 462],
                    'yrange': [50, 745],
                    'nbinned': 10000,
                    'meanImg': data_bin.sampled_mean(),
                    'meanImag_chan2': data2_bin.sampled_mean(),
                    }

    ops = {**s2p.default_ops(), **combined_ops}
    np.save(ops['ops_path'], ops)

    ops = s2p.run_s2p(ops=ops)

