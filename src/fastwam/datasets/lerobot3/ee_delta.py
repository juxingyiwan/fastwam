"""Mesh-free kinematics and physical, base-frame end-effector delta encoding.

This module deliberately depends only on NumPy and the standard library so a
prepared dataset can carry an independent, executable copy of its conversion.
"""

import xml.etree.ElementTree as ET

import numpy as np

ACTION_NAMES = [
    "ee.delta_x_m",
    "ee.delta_y_m",
    "ee.delta_z_m",
    "ee.delta_rx_rad",
    "ee.delta_ry_rad",
    "ee.delta_rz_rad",
    "gripper.command_open_fraction",
]
STATE_NAMES = [
    "ee.x_m",
    "ee.y_m",
    "ee.z_m",
    "ee.rx_rad",
    "ee.ry_rad",
    "ee.rz_rad",
    "gripper.measured_open_fraction",
]
REPRESENTATION = "base_frame_ee_delta_v2"


def rotation_exp(vector):
    """SO(3) exponential for axis-angle vectors in radians, including zero."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[-1:] != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite axis-angle vectors of length three")
    x, y, z = np.moveaxis(vector, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    angle = np.linalg.norm(vector, axis=-1)[..., None, None]
    return (
        np.eye(3)
        + np.sinc(angle / np.pi) * skew
        + 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * (skew @ skew)
    )


def rotation_log(matrix):
    """Principal axis-angle vector; robust at zero and pi, never Euler angles.

    The scalar-positive quaternion convention gives angles in [0, pi]. At pi
    either axis sign represents the same rotation; no temporal unwrapping is
    applied to the absolute orientation state.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite rotation matrices")
    if not np.allclose(matrix @ matrix.swapaxes(-1, -2), np.eye(3), atol=1e-6):
        raise ValueError("Rotation matrices must be orthogonal")
    if not np.allclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("Rotation matrices must have determinant one")
    m = matrix.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    squared = np.maximum(
        np.stack(
            [
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ],
            -1,
        ),
        0,
    )
    a = m[:, 2, 1] - m[:, 1, 2]
    b = m[:, 0, 2] - m[:, 2, 0]
    c = m[:, 1, 0] - m[:, 0, 1]
    d = m[:, 0, 1] + m[:, 1, 0]
    e = m[:, 0, 2] + m[:, 2, 0]
    f = m[:, 1, 2] + m[:, 2, 1]
    candidates = np.stack(
        [
            np.stack([squared[:, 0], a, b, c], -1),
            np.stack([a, squared[:, 1], d, e], -1),
            np.stack([b, d, squared[:, 2], f], -1),
            np.stack([c, e, f, squared[:, 3]], -1),
        ],
        -2,
    )
    # The largest component is well-conditioned even for a half turn.
    quat = candidates[np.arange(len(m)), squared.argmax(-1)]
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.where(quat[:, :1] < 0, -quat, quat)
    sine = np.linalg.norm(quat[:, 1:], axis=-1)
    angle = 2 * np.arctan2(sine, quat[:, 0])
    factor = np.divide(angle, sine, out=np.full_like(angle, 2), where=sine > 1e-12)
    return (quat[:, 1:] * factor[:, None]).reshape(*matrix.shape[:-2], 3)


def compose_delta(position, rotation, action):
    """Compose ONE physical delta with its current measured pose.

    Do not compose an entire future chunk with one frozen initial pose. A robot
    controller must supply fresh feedback for each executed action and perform
    its own IK, limits and gripper command conversion.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1:] != (7,) or not np.isfinite(action).all():
        raise ValueError("Expected finite seven-dimensional physical actions")
    return (
        np.asarray(position) + action[..., :3],
        rotation_exp(action[..., 3:6]) @ np.asarray(rotation),
    )


def _vector(element, attribute, default):
    value = np.asarray(
        (element.get(attribute, default) if element is not None else default).split(),
        dtype=np.float64,
    )
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"URDF {attribute} must contain three finite numbers")
    return value


class URDFKinematics:
    """Forward kinematics for a fixed/revolute/continuous base-to-tool chain."""

    def __init__(self, urdf_path, base_link, tool_link, joint_names):
        root = ET.parse(urdf_path).getroot()
        links = {link.attrib["name"] for link in root.findall("link")}
        if base_link not in links or tool_link not in links or base_link == tool_link:
            raise ValueError("Expected distinct base and tool links in the URDF")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Joint names must be nonempty and distinct")
        by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            if child in by_child:
                raise ValueError(f"Multiple parent joints for {child}")
            by_child[child] = joint
        chain, visited = [], set()
        link = tool_link
        while link != base_link:
            if link in visited or link not in by_child:
                raise ValueError("No acyclic base-to-tool chain")
            visited.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.joint_names = tuple(joint_names)
        self.chain = []
        active = []
        for joint in reversed(chain):
            kind, name = joint.attrib["type"], joint.attrib["name"]
            if (
                kind not in {"fixed", "revolute", "continuous"}
                or joint.find("mimic") is not None
            ):
                raise ValueError(f"Unsupported chain joint: {name} ({kind})")
            origin = joint.find("origin")
            xyz, rpy = _vector(origin, "xyz", "0 0 0"), _vector(origin, "rpy", "0 0 0")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            # URDF fixed-axis roll/pitch/yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
            rotations = rotation_exp(np.eye(3) * rpy[:, None])
            transform[:3, :3] = rotations[2] @ rotations[1] @ rotations[0]
            index, axis = None, None
            if kind != "fixed":
                active.append(name)
                if name not in joint_names:
                    raise ValueError(f"Missing joint: {name}")
                index = joint_names.index(name)
                axis = _vector(joint.find("axis"), "xyz", "1 0 0")
                norm = np.linalg.norm(axis)
                if norm == 0:
                    raise ValueError(f"Zero joint axis: {name}")
                axis /= norm
            self.chain.append((transform, index, axis))
        if set(active) != set(joint_names):
            raise ValueError("Joint names must exactly match the selected chain")

    def __call__(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if (
            joints.shape[-1:] != (len(self.joint_names),)
            or not np.isfinite(joints).all()
        ):
            raise ValueError("Expected finite joint vectors in URDF radians")
        pose = np.broadcast_to(np.eye(4), (*joints.shape[:-1], 4, 4)).copy()
        for origin, index, axis in self.chain:
            pose = pose @ origin
            if index is not None:
                pose[..., :3, :3] = pose[..., :3, :3] @ rotation_exp(
                    joints[..., index, None] * axis
                )
        return pose


class EEDeltaCodec:
    """Encode measured motor degrees and recorded pre-driver commands.

    The first six features follow the configured URDF joint order. The seventh
    is a gripper motor coordinate. Gripper fractions are continuous proxies,
    not calibrated jaw widths; recorded targets and feedback are never clipped.
    """

    def __init__(self, urdf_path, config):
        self.config = config
        self.fk = URDFKinematics(
            urdf_path, config["base_link"], config["tool_link"], config["joint_names"]
        )
        self.scale = np.asarray(config["command_to_motor_scale"], dtype=np.float64)
        self.offset = np.asarray(config["motor_to_urdf_offset_rad"], dtype=np.float64)
        if (
            len(config["joint_names"]) != 6
            or self.scale.shape != (7,)
            or self.offset.shape != (6,)
        ):
            raise ValueError("Expected six arm joints and one gripper")
        if (
            not np.isfinite(self.scale).all()
            or not np.isfinite(self.offset).all()
            or np.any(self.scale == 0)
        ):
            raise ValueError("Invalid motor mapping")
        self.closed, self.open = map(float, config["gripper_command_endpoints"])
        if not np.isfinite([self.closed, self.open]).all() or self.closed == self.open:
            raise ValueError("Gripper command endpoints must be finite and distinct")

    def _check(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape[-1:] != (7,) or not np.isfinite(value).all():
            raise ValueError("Expected finite seven-dimensional recorded vectors")
        return value

    def state(self, motor_state):
        """Encode feedback at training and inference with the identical mapping."""
        motor_state = self._check(motor_state)
        pose = self.fk(np.deg2rad(motor_state[..., :6]) + self.offset)
        fraction = (motor_state[..., 6:7] / self.scale[6] - self.closed) / (
            self.open - self.closed
        )
        return np.concatenate(
            [
                pose[..., :3, 3],
                rotation_log(pose[..., :3, :3]),
                fraction,
            ],
            -1,
        ).astype(np.float32)

    def encode(self, motor_state, command):
        motor_state, command = self._check(motor_state), self._check(command)
        if motor_state.shape != command.shape:
            raise ValueError("State and action must describe the same rows")
        measured = self.fk(np.deg2rad(motor_state[..., :6]) + self.offset)
        target = self.fk(np.deg2rad(command[..., :6] * self.scale[:6]) + self.offset)
        delta = rotation_log(
            target[..., :3, :3] @ measured[..., :3, :3].swapaxes(-1, -2)
        )
        fraction = (command[..., 6:7] - self.closed) / (self.open - self.closed)
        action = np.concatenate(
            [target[..., :3, 3] - measured[..., :3, 3], delta, fraction], -1
        )
        return self.state(motor_state), action.astype(np.float32)

    def gripper_command(self, fraction):
        """Inverse of the continuous opening fraction, before driver scaling."""
        return np.asarray(fraction) * (self.open - self.closed) + self.closed
