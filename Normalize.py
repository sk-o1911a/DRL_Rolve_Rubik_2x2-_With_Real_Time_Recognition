import numpy as np

U, R, F, D, L, B = 0, 1, 2, 3, 4, 5


def rotate_face_clockwise(face):
    return np.rot90(face, -1)

def rotate_face_counter_clockwise(face):
    return np.rot90(face, 1)

def rotate_face_180(face):
    return np.rot90(face, 2)

def deep_copy_cube(cube):
    return np.copy(cube)


# ══════════════════════════════════════════════════════════════
# WHOLE-CUBE ROTATIONS
# ══════════════════════════════════════════════════════════════

def whole_x_cw(cube):
    c = deep_copy_cube(cube)
    c[U] = cube[F].copy()
    c[F] = cube[D].copy()
    c[D] = rotate_face_180(cube[B])
    c[B] = rotate_face_180(cube[U])
    c[R] = rotate_face_clockwise(cube[R])
    c[L] = rotate_face_counter_clockwise(cube[L])
    return c

def whole_x_ccw(cube):
    c = deep_copy_cube(cube)
    c[U] = rotate_face_180(cube[B])
    c[F] = cube[U].copy()
    c[D] = cube[F].copy()
    c[B] = rotate_face_180(cube[D])
    c[R] = rotate_face_counter_clockwise(cube[R])
    c[L] = rotate_face_clockwise(cube[L])
    return c

def whole_x2(cube):
    c = deep_copy_cube(cube)
    c[U] = rotate_face_180(cube[D])
    c[D] = rotate_face_180(cube[U])
    c[F] = rotate_face_180(cube[B])
    c[B] = rotate_face_180(cube[F])
    c[R] = rotate_face_180(cube[R])
    c[L] = rotate_face_180(cube[L])
    return c

def whole_y_cw(cube):
    c = deep_copy_cube(cube)
    c[F] = cube[R].copy()
    c[L] = cube[F].copy()
    c[B] = cube[L].copy()
    c[R] = cube[B].copy()
    c[U] = rotate_face_clockwise(cube[U])
    c[D] = rotate_face_counter_clockwise(cube[D])
    return c

def whole_y_ccw(cube):
    c = deep_copy_cube(cube)
    c[F] = cube[L].copy()
    c[R] = cube[F].copy()
    c[B] = cube[R].copy()
    c[L] = cube[B].copy()
    c[U] = rotate_face_counter_clockwise(cube[U])
    c[D] = rotate_face_clockwise(cube[D])
    return c

def whole_y2(cube):
    c = deep_copy_cube(cube)
    c[F] = cube[B].copy()
    c[B] = cube[F].copy()
    c[R] = cube[L].copy()
    c[L] = cube[R].copy()
    c[U] = rotate_face_180(cube[U])
    c[D] = rotate_face_180(cube[D])
    return c

def whole_z_cw(cube):
    c = deep_copy_cube(cube)
    c[U] = rotate_face_clockwise(cube[L])
    c[R] = rotate_face_clockwise(cube[U])
    c[D] = rotate_face_clockwise(cube[R])
    c[L] = rotate_face_clockwise(cube[D])
    c[F] = rotate_face_clockwise(cube[F])
    c[B] = rotate_face_counter_clockwise(cube[B])
    return c

def whole_z_ccw(cube):
    c = deep_copy_cube(cube)
    c[U] = rotate_face_counter_clockwise(cube[R])
    c[L] = rotate_face_counter_clockwise(cube[U])
    c[D] = rotate_face_counter_clockwise(cube[L])
    c[R] = rotate_face_counter_clockwise(cube[D])
    c[F] = rotate_face_counter_clockwise(cube[F])
    c[B] = rotate_face_clockwise(cube[B])
    return c


def normalize_dlb(cube):
    """
    Normalize cube về orientation chuẩn:
    D[1,0] = 3, L[1,0] = 4, B[1,1] = 5
    """
    for face_up in range(6):
        if   face_up == 0: base = deep_copy_cube(cube)
        elif face_up == 1: base = whole_z_ccw(cube)
        elif face_up == 2: base = whole_x_cw(cube)
        elif face_up == 3: base = whole_x2(cube)
        elif face_up == 4: base = whole_z_cw(cube)
        else:              base = whole_x_ccw(cube)

        for yr in range(4):
            if   yr == 0: c = deep_copy_cube(base)
            elif yr == 1: c = whole_y_cw(base)
            elif yr == 2: c = whole_y2(base)
            else:         c = whole_y_ccw(base)

            if (c[D, 1, 0] == 3 and
                c[L, 1, 0] == 4 and
                c[B, 1, 1] == 5):
                return c

    return cube
