from scipy.spatial.transform import Rotation as R

## a set of helpr functions

def rotation_with_quaternion(Xx, Yy, Zz, coords, verbose=False):
    """Xx : angle around x axis
    Yy: angle around y axis
    Zz: angle around z axis
    coords: matrix Nx3 of N atoms in 3 dimensions
    Quaternions are cool"""

    # makes a quaternion
    r = R.from_euler('zyx',
    [Zz, Yy, Xx], degrees=True)
    if verbose:
        print(f'Quaternion is {r.as_quat()}')
    return r.apply(coords)