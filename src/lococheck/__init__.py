import numpy as np
from typing import List


def check_anymal_d_obs(x: np.ndarray) -> List[bool]:
    """
    params:
    -------
    x: np.ndarray
        State vector of shape (n_rows,36)

    idx:
    ----
    0 to 2: base linear velocity (m/s)
    3 to 5: roll, pitch, yaw (rad/s)
    6 to 8: gravity unit vector (x_dir, y_dir, z_dir)
    9 to 11: velocity command (x_dir, y_dir, z_yaw)
    12 to 23: joint positions (in absolute radians)
    24 to 35: joint velocities (in absolute radians per second)
    """

    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    elif hasattr(x, 'numpy'):
        x = x.numpy()

    assert x.shape[1] == 36, "State vector must have 36 columns"
    assert x.ndim == 2, "State vector must be 2D"

    # check that base linear velocity is reasonable
    assert np.all(x[:, 0:3] < 5.0), "Base linear velocity is too high"
    assert np.all(x[:, 0:3] > -5.0), "Base linear velocity is too low"

    # check that roll, pitch yaw velocities are reasonable
    assert np.all(x[:, 3:6] < 5.0), "Roll, pitch, yaw velocities are too high"
    assert np.all(x[:, 3:6] > -5.0), "Roll, pitch, yaw velocities are too low"

    # check that gravity unit vector is normalized
    gravity_vector = x[:, 6:9]
    assert np.allclose(np.linalg.norm(gravity_vector, axis=1), 1.0, atol=1e-6), (
        "Gravity vector is not normalized"
    )
    # check that 80% of the z-component is less than -0.8 (i.e., gravity is pointing downward)
    assert np.mean(gravity_vector[:, 2] < -0.8) > 0.8, (
        "Gravity vector is not pointing downward more than 80% of the time"
    )

    # check that velocity commands are reasonable
    assert np.all(x[:, 9:12] < 5.0), "Velocity commands are too high"
    assert np.all(x[:, 9:12] > -5.0), "Velocity commands are too low"

    # check that joint positions are reasonable
    assert np.all(x[:, 12:24] < 5.0), "Joint positions are too high"
    assert np.all(x[:, 12:24] > -5.0), "Joint positions are too low"

    # check that joint velocities are reasonable
    assert np.all(x[:, 24:36] < 5.0), "Joint velocities are too high"
    assert np.all(x[:, 24:36] > -5.0), "Joint velocities are too low"

    return [True]
