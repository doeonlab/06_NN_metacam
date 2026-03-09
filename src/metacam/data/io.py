import numpy as np
import scipy.io

try:
    import mat73
except ImportError:
    mat73 = None

try:
    import h5py
except ImportError:
    h5py = None


def _load_hdf5_mat_file(file_path):
    if h5py is None:
        raise ImportError("h5py is not installed")

    data = {}
    with h5py.File(file_path, "r") as f:
        for key, value in f.items():
            if key.startswith("#"):
                continue
            if isinstance(value, h5py.Dataset):
                arr = np.array(value)
                if arr.ndim >= 2:
                    arr = np.transpose(arr)
                data[key] = arr

    if not data:
        raise ValueError("No top-level numeric datasets found in v7.3 .mat file.")

    return data


def load_mat_file(file_path):
    """Load a MATLAB .mat file across v7.2 and v7.3+ formats."""
    try:
        return scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        try:
            return _load_hdf5_mat_file(file_path)
        except Exception:
            pass

        if mat73 is None:
            print("MATLAB 7.3+ loader not available. Install `h5py` or `mat73`.")
            return None

        try:
            return mat73.loadmat(file_path)
        except Exception as e:
            print(f"An error occurred while reading the .mat file with mat73: {e}")
            return None
    except Exception as e:
        print(f"An error occurred while reading the .mat file: {e}")
        return None


def load_matfile(path):
    import scipy.io as sio

    data = sio.loadmat(path)

    return data


def batch_generator_2d(data, batch_size, data2=None, data3=None):

    s, a, b = data.shape
    s_idx = np.arange(s)
    np.random.shuffle(s_idx)
    data = data[s_idx, :, :].reshape(-1, batch_size, 1, a, b)

    if data2 is not None:
        if data2.shape[-1] == 1:
            data2 = data2[s_idx, :, :].reshape(-1, batch_size, 1, 1, 1)
        else:
            data2 = data2[s_idx, :, :].reshape(-1, batch_size, 1, a, b)

        if data3 is not None:
            data3 = data3[s_idx, :, :].reshape(-1, batch_size, 1, a, b)

            return data, data2, data3

        return data, data2

    return data


def data_augmentation(data, rotation_p=0.5, flip_p=0.5, data2=None, data3=None):
    N, Nx, Ny = np.shape(data)

    # rotate data
    if data2 is None:
        data = data_rotation(data, rotation_p, None)
        data = data_flipping(data, flip_p, flip_dim=0, data2=None)  # vertical flipping
        data = data_flipping(
            data, flip_p, flip_dim=1, data2=None
        )  # horizontal flipping

        return data
    elif data3 is None:
        data, data2 = data_rotation(data, rotation_p, data2)
        data, data2 = data_flipping(
            data, flip_p, flip_dim=0, data2=data2
        )  # vertical flipping
        data, data2 = data_flipping(
            data, flip_p, flip_dim=1, data2=data2
        )  # horizontal flipping

        return data, data2

    else:
        data, data2, data3 = data_rotation(data, rotation_p, data2, data3)
        data, data2, data3 = data_flipping(
            data, flip_p, flip_dim=0, data2=data2, data3=data3
        )  # vertical flipping
        data, data2, data3 = data_flipping(
            data, flip_p, flip_dim=1, data2=data2, data3=data3
        )  # horizontal flipping

        return data, data2, data3


def data_flipping(data, p=0.5, flip_dim=0, data2=None, data3=None):

    N, Nx, Ny = np.shape(data)

    r_limit = int(N * p)

    r_idx = np.arange(N)
    np.random.shuffle(r_idx)

    for idx, s_idx in enumerate(r_idx):

        if idx >= r_limit:
            break
        data[s_idx, :, :] = np.flip(data[s_idx, :, :], flip_dim)

        if data2 is not None:
            data2[s_idx, :, :] = np.flip(data2[s_idx, :, :], flip_dim)

        if data3 is not None:
            data3[s_idx, :, :] = np.flip(data3[s_idx, :, :], flip_dim)

    if data2 is None:
        return data
    elif data3 is None:
        return data, data2
    else:
        return data, data2, data3


def data_rotation(data, p=0.5, data2=None, data3=None):
    N, Nx, Ny = np.shape(data)

    r_limit = int(N * p)

    r_idx = np.arange(N)
    np.random.shuffle(r_idx)

    for idx, s_idx in enumerate(r_idx):

        if idx >= r_limit:
            break

        rot_times = np.random.randint(-3, 4)
        data[s_idx, :, :] = np.rot90(data[s_idx, :, :], rot_times)

        if data2 is not None:
            data2[s_idx, :, :] = np.rot90(data2[s_idx, :, :], rot_times)

        if data3 is not None:
            data3[s_idx, :, :] = np.rot90(data3[s_idx, :, :], rot_times)

    if data2 is None:
        return data
    elif data3 is None:
        return data, data2
    else:
        return data, data2, data3


def Load_data(path, folder=None, transform=None):
    import torchvision

    if folder is None:
        data = load_matfile(path=path)

        return data

    else:
        data = torchvision.datasets.ImageFolder(root=path, transform=transform)

        return data
