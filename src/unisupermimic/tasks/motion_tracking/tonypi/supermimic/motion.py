"""Reference TonyPi motions resampled to 50 Hz with tracked-body forward kinematics."""
from __future__ import annotations
import copy
import csv
import math
import os
from collections.abc import Sequence
import torch
from unisupermimic.rl.algos.supermimic.geometry import mat_to_6d, matmul, matvec, quat_slerp, quat_to_mat, so3_log, yaw_mat
from unisupermimic.rl.contracts.supermimic import State
from .kinematics import TonyPiKinematics
CONTROL_DT = 0.02
BIN_FRAMES = 50

class MotionLib:

    def __init__(self, files: Sequence[str], kin: TonyPiKinematics, device: str | torch.device, src_fps: float=30.0):
        if not files:
            raise ValueError('MotionLib requires at least one motion file')
        self.device = torch.device(device)
        self.kin = kin
        self.compact = False
        if any((not str(file).endswith('.csv') for file in files)):
            raise ValueError('TonyPi requires CSV motion files')
        self.names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        positions, quaternions, joint_positions = ([], [], [])
        lengths = []
        for file in files:
            raw = _load_csv(file)
            if raw.shape[0] < 2 or raw.shape[1] != 23:
                raise ValueError(f'{file} must have at least two rows and 23 columns')
            position, quat_xyzw, joints = (raw[:, :3], raw[:, 3:7], raw[:, 7:23])
            quaternion = torch.cat((quat_xyzw[:, 3:4], quat_xyzw[:, :3]), -1)
            source_frames = raw.shape[0]
            duration = (source_frames - 1) / src_fps
            native_rate = math.isclose(src_fps * CONTROL_DT, 1.0, abs_tol=1e-09)
            num_frames = source_frames if native_rate else math.floor(duration / CONTROL_DT) + 1
            sample = torch.arange(num_frames, dtype=torch.float32)
            if not native_rate:
                sample = sample * CONTROL_DT * src_fps
            lower = torch.clamp(sample.floor().long(), max=source_frames - 2)
            alpha = (sample - lower.float()).clamp(0, 1)
            position = position[lower] * (1 - alpha[:, None]) + position[lower + 1] * alpha[:, None]
            joints = joints[lower] * (1 - alpha[:, None]) + joints[lower + 1] * alpha[:, None]
            quaternion = quat_slerp(quaternion[lower], quaternion[lower + 1], alpha)
            positions.append(position)
            quaternions.append(quaternion)
            joint_positions.append(joints)
            lengths.append(num_frames)
        self.clip_len = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_start = torch.cumsum(torch.tensor([0, *lengths[:-1]], dtype=torch.long, device=self.device), 0)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = sum(lengths)
        self.clip_id = torch.cat([torch.full((length,), index, dtype=torch.long) for index, length in enumerate(lengths)]).to(self.device)
        root_p, root_R, root_v, root_w, joint_q, joint_qd = ([], [], [], [], [], [])
        for position, quaternion, joints in zip(positions, quaternions, joint_positions):
            rotation = quat_to_mat(quaternion)
            linear_velocity = _gradient(position, CONTROL_DT)
            joint_velocity = _gradient(joints, CONTROL_DT)
            angular_velocity = torch.zeros_like(position)
            angular_velocity[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            angular_velocity[0] = so3_log(rotation[1] @ rotation[0].transpose(-1, -2)) / CONTROL_DT
            angular_velocity[-1] = so3_log(rotation[-1] @ rotation[-2].transpose(-1, -2)) / CONTROL_DT
            root_p.append(position)
            root_R.append(rotation)
            root_v.append(linear_velocity)
            root_w.append(angular_velocity)
            joint_q.append(joints)
            joint_qd.append(joint_velocity)
        self._root_p, self._root_R, self._root_v, self._root_w = (torch.cat(values).to(self.device) for values in (root_p, root_R, root_v, root_w))
        self._q, self._qd = (torch.cat(values).to(self.device) for values in (joint_q, joint_qd))
        with torch.no_grad():
            body_p, body_R, body_v, body_w = ([], [], [], [])
            for start in range(0, self.num_frames, 65536):
                end = min(start + 65536, self.num_frames)
                output = kin.forward(self.root_p[start:end], self.root_R[start:end], self.root_v[start:end], self.root_w[start:end], self.q[start:end], self.qd[start:end])
                body_p.append(output[0])
                body_R.append(output[1])
                body_v.append(output[2])
                body_w.append(output[3])
            self._body_p, self._body_R, self._body_v, self._body_w = (torch.cat(values) for values in (body_p, body_R, body_v, body_w))
        root_rotation_t = self.root_R.transpose(-1, -2)
        self._root_v_local = (root_rotation_t @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self._root_w_local = (root_rotation_t @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self._root_gz = self.root_R[:, 2, 2]
        self._build_indices()

    def _build_indices(self) -> None:
        clips = torch.arange(len(self.names), dtype=torch.long, device=self.device)
        self.clip_id = torch.repeat_interleave(clips, self.clip_len)
        bins_per_clip = (self.clip_len + BIN_FRAMES - 1) // BIN_FRAMES
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.bin_clip = torch.repeat_interleave(clips, bins_per_clip)
        local_bin = torch.arange(len(self.bin_clip), device=self.device) - self.clip_bin_offset[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local_bin * BIN_FRAMES
        self.num_bins = len(self.bin_clip)
        frames = torch.arange(self.num_frames, device=self.device)
        self.frame_bin = self.clip_bin_offset[self.clip_id] + (frames - self.clip_start[self.clip_id]) // BIN_FRAMES

    @property
    def root_p(self) -> torch.Tensor:
        return self._root_p

    @property
    def root_R(self) -> torch.Tensor:
        return self._root_R

    @property
    def root_v(self) -> torch.Tensor:
        return self._root_v

    @property
    def root_w(self) -> torch.Tensor:
        return self._root_w

    @property
    def q(self) -> torch.Tensor:
        return self._q

    @property
    def qd(self) -> torch.Tensor:
        return self._qd

    @property
    def root_v_local(self) -> torch.Tensor:
        return self._root_v_local
        rotation_t = self.root_R.transpose(-1, -2)
        return (rotation_t @ self.root_v.unsqueeze(-1)).squeeze(-1)

    @property
    def root_w_local(self) -> torch.Tensor:
        return self._root_w_local
        rotation_t = self.root_R.transpose(-1, -2)
        return (rotation_t @ self.root_w.unsqueeze(-1)).squeeze(-1)

    @property
    def root_gz(self) -> torch.Tensor:
        return self.root_R[:, 2, 2]

    @property
    def body_p(self) -> torch.Tensor:
        return self._body_p

    @property
    def body_R(self) -> torch.Tensor:
        return self._body_R

    @property
    def body_v(self) -> torch.Tensor:
        return self._body_v

    @property
    def body_w(self) -> torch.Tensor:
        return self._body_w

    def body_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self._body_p[frames], self._body_R[frames], self._body_v[frames], self._body_w[frames])
        reference = self._ref[frames].float()
        return self.kin.forward(reference[..., :3], quat_to_mat(reference[..., 3:7]), reference[..., 23:38], reference[..., 38:29], reference[..., 7:23], reference[..., 29:45])

    def to(self, device: str | torch.device) -> MotionLib:
        """Return a copy on ``device``; the original instance is left untouched."""
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        new.kin = self.kin.to(device)
        return new

    def future_frames(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Return frames+1 through frames+K, clamped at each clip's last frame."""
        end = self.clip_end[self.clip_id[frames]]
        future = frames.unsqueeze(-1) + torch.arange(1, num_frames + 1, device=frames.device)
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Build future reference features in the current reference heading frame."""
        future = self.future_frames(frames, num_frames)
        current_rotation = self.root_R[frames]
        root_p = self.root_p[future]
        root_R = self.root_R[future]
        root_v_local = self.root_v_local[future]
        root_w_local = self.root_w_local[future]
        q, qd = (self.q[future], self.qd[future])
        current_p = self.root_p[frames]
        heading_t = yaw_mat(current_rotation).transpose(-1, -2).unsqueeze(1)
        position_delta = matvec(heading_t, root_p - current_p.unsqueeze(1))
        rotation = mat_to_6d(matmul(heading_t, root_R))
        features = torch.cat((position_delta, rotation, root_v_local, root_w_local, q, qd * 0.1), -1)
        return features.reshape(frames.shape[0], -1)

    def ref_state(self, frames: torch.Tensor) -> State:
        return (self.root_p[frames], self.root_R[frames], self.root_v[frames], self.root_w[frames], self.q[frames], self.qd[frames])

def _load_csv(path: str) -> torch.Tensor:
    with open(path, encoding='utf-8', newline='') as stream:
        rows = [[float(value) for value in row] for row in csv.reader(stream) if row and (not row[0].lstrip().startswith('#'))]
    if not rows:
        raise ValueError(f'{path} is empty')
    return torch.tensor(rows, dtype=torch.float32)

def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
    gradient = torch.zeros_like(value)
    gradient[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
    gradient[0] = (value[1] - value[0]) / dt
    gradient[-1] = (value[-1] - value[-2]) / dt
    return gradient
