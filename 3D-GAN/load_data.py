# importing library
import glob
import numpy as np
import scipy.io as io
import scipy.ndimage as nd

def get3DImages(data_dir):
    all_files = np.random.choice(glob.glob(data_dir), size=10)
    all_volumes = np.asarray([getVoxelsFromMat(file) for file in all_files], dtype=np.bool)

    return all_volumes


def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    # Note: loaded 3D image is 30x30x30 but Our network requires images of shape 64x64x64.
    # NumPy's pad() method to increase the size of the 3D image to 32x32x32
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    if cube_len != 32 and cube_len == 64:
        # zoom() function from the scipy.ndimage module to convert the 3D image to a 3D image with dimensions of 64x64x64
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

    return voxels

if __name__ == '__main__':
    # Loading data
    object_name = 'chair'
    data_dir = 'data/volumetric_data/{}/30/train/*.mat'.format(object_name)

    print("Loading data...")
    volumes = get3DImages(data_dir=data_dir)
    volumes = volumes[..., np.newaxis].astype(np.float)
    print("Data loaded...")