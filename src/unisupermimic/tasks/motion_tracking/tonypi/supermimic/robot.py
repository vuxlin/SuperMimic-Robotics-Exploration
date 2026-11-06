"""TonyPi 16 calibrated body servos; head/grippers fixed at zero."""
TonyPi_JOINT_NAMES = ['l_ankle_roll', 'l_ankle_pitch', 'l_knee', 'l_hip_pitch', 'l_hip_roll', 'l_elbow', 'l_shoulder_roll', 'l_shoulder_pitch', 'r_ankle_roll', 'r_ankle_pitch', 'r_knee', 'r_hip_pitch', 'r_hip_roll', 'r_elbow', 'r_shoulder_roll', 'r_shoulder_pitch']
TRACKED_BODIES = ['torso', 'l_hip_roll', 'l_knee', 'l_ankle_roll', 'l_shoulder_pitch', 'l_shoulder_roll', 'l_elbow', 'r_hip_roll', 'r_knee', 'r_ankle_roll', 'r_shoulder_pitch', 'r_shoulder_roll', 'r_elbow']
COLLISION_GEOM_NAMES = ('torso_col', 'head_pan_col', 'head_tilt_col', 'r_hip_roll_col', 'r_hip_pitch_col', 'r_knee_col', 'r_ankle_pitch_col', 'r_ankle_roll_sole', 'r_shoulder_pitch_col', 'r_shoulder_roll_col', 'r_elbow_col', 'r_gripper_col', 'r_gripper_pad1', 'r_gripper_pad2', 'r_gripper_pad3', 'l_hip_roll_col', 'l_hip_pitch_col', 'l_knee_col', 'l_ankle_pitch_col', 'l_ankle_roll_sole', 'l_shoulder_pitch_col', 'l_shoulder_roll_col', 'l_elbow_col', 'l_gripper_col', 'l_gripper_pad1', 'l_gripper_pad2', 'l_gripper_pad3')
DEFAULT_Q = [0.01875064648857445, -0.3849503844767196, 1.1121744066406232, -0.7563127319545233, -0.0020991075354709885, 0.38103031547881133, -0.29657327118856713, 0.2566046361638208, 0.002972646412584089, -0.35059588356956256, 1.1040688957990337, -0.7811889728852173, 0.028746708631733127, -0.3142774208619077, 0.2920059883103311, 0.25249205696898286]
ACTION_SCALES = {'l_ankle_roll': 0.3, 'l_ankle_pitch': 0.3, 'l_knee': 0.3, 'l_hip_pitch': 0.3, 'l_hip_roll': 0.3, 'l_elbow': 0.5, 'l_shoulder_roll': 0.5, 'l_shoulder_pitch': 0.5, 'r_ankle_roll': 0.3, 'r_ankle_pitch': 0.3, 'r_knee': 0.3, 'r_hip_pitch': 0.3, 'r_hip_roll': 0.3, 'r_elbow': 0.5, 'r_shoulder_roll': 0.5, 'r_shoulder_pitch': 0.5}
def action_scale(joint_name):
    return ACTION_SCALES[joint_name]
def default_joint_pos_list():
    return list(DEFAULT_Q)
