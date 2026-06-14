import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 80012026
SEEDS = list(range(7))
INTENTS = ["push", "pull", "rotate", "lift", "place", "handoff"]
METHODS = [
    "raw_trajectory_knn",
    "body_normalized_knn",
    "velocity_contact_classifier",
    "object_affordance_classifier",
    "style_invariant_logistic",
    "physical_intent_disambiguation",
    "oracle_intent_upper_bound",
]
TRAIN_EPISODES_PER_INTENT = 36
TEST_EPISODES_PER_SPLIT_SEED = 42
STRESS_EPISODES_PER_SEED = 24
T = 24

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


@dataclass
class Episode:
    split: str
    seed: int
    episode_id: int
    intent: str
    morphology: str
    arm_length: float
    shoulder_width: float
    handedness: int
    camera_scale: float
    style_speed: float
    curvature: float
    hesitation: float
    occlusion: float
    noise: float
    object_class: str
    affordance: np.ndarray
    shoulder: np.ndarray
    elbow: np.ndarray
    hand: np.ndarray
    object_pos: np.ndarray
    object_delta: np.ndarray
    rotation_delta: float
    contact_time: float
    target: np.ndarray
    true_param: np.ndarray


def stable_rng(*parts):
    acc = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            for ch in part:
                acc = (acc * 131 + ord(ch)) % (2**32 - 1)
        else:
            acc = (acc * 131 + int(part)) % (2**32 - 1)
    return np.random.default_rng(acc)


def ci95(vals):
    vals = list(vals)
    if len(vals) <= 1:
        return 0.0
    mean = float(np.mean(vals))
    sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
    return 1.96 * sd / math.sqrt(len(vals))


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def safe_standardize(x, mean, std):
    return (x - mean) / np.maximum(std, 1e-6)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(labels):
    y = np.zeros((len(labels), len(INTENTS)))
    for i, label in enumerate(labels):
        y[i, INTENTS.index(label)] = 1.0
    return y


def train_softmax_classifier(features, labels, iterations=360, lr=0.12, reg=0.006):
    x = np.asarray(features, dtype=float)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    xs = safe_standardize(x, mean, std)
    xs = np.c_[np.ones(len(xs)), xs]
    y = one_hot(labels)
    w = np.zeros((xs.shape[1], len(INTENTS)))
    for _ in range(iterations):
        p = softmax(xs @ w)
        grad = xs.T @ (p - y) / len(xs)
        grad[1:] += reg * w[1:]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std}


def predict_softmax(model, feature):
    x = safe_standardize(np.asarray(feature, dtype=float), model["mean"], model["std"])
    x = np.r_[1.0, x]
    prob = softmax((x @ model["w"])[None, :])[0]
    idx = int(np.argmax(prob))
    return INTENTS[idx], float(prob[idx]), prob


def morphology_params(split, rng, stress=0.0):
    if split in {"train", "seen_morphology_clean", "style_shift_same_intent", "object_affordance_ambiguity"}:
        arm = rng.normal(1.0, 0.055)
        shoulder = rng.normal(0.40, 0.025)
        camera = rng.normal(1.0, 0.035)
        morph = "seen"
    elif split == "morphology_shift":
        arm = rng.choice([rng.normal(0.74, 0.035), rng.normal(1.33, 0.045)])
        shoulder = rng.choice([rng.normal(0.30, 0.018), rng.normal(0.54, 0.020)])
        camera = rng.choice([rng.normal(0.74, 0.025), rng.normal(1.28, 0.030)])
        morph = "shift"
    elif split == "combined_hard_shift":
        arm = rng.choice([rng.normal(0.70, 0.040), rng.normal(1.38, 0.050)])
        shoulder = rng.choice([rng.normal(0.28, 0.020), rng.normal(0.57, 0.025)])
        camera = rng.choice([rng.normal(0.70, 0.030), rng.normal(1.35, 0.035)])
        morph = "shift"
    elif split == "stress_morphology":
        gap = stress
        arm = rng.choice([0.95 - 0.28 * gap + rng.normal(0, 0.025), 1.05 + 0.38 * gap + rng.normal(0, 0.030)])
        shoulder = rng.choice([0.39 - 0.12 * gap + rng.normal(0, 0.012), 0.41 + 0.16 * gap + rng.normal(0, 0.012)])
        camera = rng.choice([1.0 - 0.30 * gap + rng.normal(0, 0.018), 1.0 + 0.38 * gap + rng.normal(0, 0.022)])
        morph = "stress"
    else:
        arm = rng.normal(1.0, 0.06)
        shoulder = rng.normal(0.40, 0.025)
        camera = rng.normal(1.0, 0.035)
        morph = "seen"
    return morph, float(max(0.55, arm)), float(max(0.20, shoulder)), float(max(0.45, camera))


def style_params(split, rng, stress=0.0):
    if split == "train" or split in {"seen_morphology_clean", "morphology_shift", "object_affordance_ambiguity"}:
        speed = rng.normal(1.0, 0.08)
        curvature = rng.normal(0.18, 0.05)
        hesitation = rng.uniform(0.00, 0.08)
        occlusion = rng.uniform(0.00, 0.06)
        noise = rng.uniform(0.004, 0.012)
    elif split == "style_shift_same_intent":
        speed = rng.choice([rng.normal(0.58, 0.05), rng.normal(1.55, 0.08)])
        curvature = rng.normal(0.42, 0.10)
        hesitation = rng.uniform(0.08, 0.22)
        occlusion = rng.uniform(0.03, 0.12)
        noise = rng.uniform(0.010, 0.026)
    elif split == "combined_hard_shift":
        speed = rng.choice([rng.normal(0.52, 0.06), rng.normal(1.70, 0.09)])
        curvature = rng.normal(0.48, 0.12)
        hesitation = rng.uniform(0.10, 0.28)
        occlusion = rng.uniform(0.12, 0.32)
        noise = rng.uniform(0.018, 0.040)
    elif split == "stress_style":
        speed = rng.choice([1.0 - 0.45 * stress + rng.normal(0, 0.03), 1.0 + 0.72 * stress + rng.normal(0, 0.04)])
        curvature = 0.16 + 0.48 * stress + rng.normal(0, 0.04)
        hesitation = 0.02 + 0.25 * stress
        occlusion = 0.03 + 0.22 * stress
        noise = 0.006 + 0.034 * stress
    else:
        speed = rng.normal(1.0, 0.09)
        curvature = rng.normal(0.20, 0.06)
        hesitation = rng.uniform(0.00, 0.08)
        occlusion = rng.uniform(0.00, 0.08)
        noise = rng.uniform(0.006, 0.014)
    return float(max(0.35, speed)), float(abs(curvature)), float(hesitation), float(occlusion), float(noise)


def choose_object(intent, split, rng, stress=0.0):
    ambiguous = split in {"object_affordance_ambiguity", "combined_hard_shift", "stress_affordance"}
    object_classes = {
        "push": ["box", "sliding_box", "cart"],
        "pull": ["drawer_handle", "tethered_box", "sliding_tray"],
        "rotate": ["knob", "valve", "jar_lid"],
        "lift": ["mug", "block", "tool"],
        "place": ["cup_to_bowl", "peg_to_slot", "block_to_bin"],
        "handoff": ["tool_handoff", "cup_handoff", "part_handoff"],
    }
    if ambiguous:
        shared = {
            "push": "sliding_box",
            "pull": "sliding_box",
            "rotate": "round_handle",
            "lift": "round_handle",
            "place": "small_cup",
            "handoff": "small_cup",
        }
        object_class = shared[intent]
    else:
        object_class = str(rng.choice(object_classes[intent]))
    affordance = np.zeros(6)
    # slide_x, slide_y, rotatable, liftable, receptacle_target, recipient_target
    if intent in {"push", "pull"}:
        affordance[:2] = unit(np.array([rng.normal(1.0, 0.15), rng.normal(0.1, 0.08), 0.0]))[:2]
        if ambiguous and rng.random() < 0.45 + 0.35 * stress:
            affordance[:2] *= -1.0
    if intent == "rotate" or object_class == "round_handle":
        affordance[2] = 1.0
    if intent == "lift" or object_class in {"mug", "block", "tool", "round_handle"}:
        affordance[3] = 1.0
    if intent == "place" or object_class in {"small_cup", "cup_to_bowl", "peg_to_slot", "block_to_bin"}:
        affordance[4] = 1.0
    if intent == "handoff" or object_class in {"small_cup", "tool_handoff", "cup_handoff", "part_handoff"}:
        affordance[5] = 1.0
    return object_class, affordance


def intent_geometry(intent, rng):
    obj = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.18, 0.28), rng.uniform(0.0, 0.08)])
    if rng.random() < 0.5:
        obj[0] *= -1
    body_to_obj = unit(obj)
    lateral = unit(np.array([-body_to_obj[1], body_to_obj[0], 0.0]))
    if intent == "push":
        direction = body_to_obj
        delta = 0.26 * direction
        rot = 0.0
        target = obj + delta
        param = unit(delta)
    elif intent == "pull":
        direction = -body_to_obj
        delta = 0.24 * direction
        rot = 0.0
        target = obj + delta
        param = unit(delta)
    elif intent == "rotate":
        sign = float(rng.choice([-1, 1]))
        delta = np.zeros(3)
        rot = sign * rng.uniform(0.75, 1.15)
        target = obj + lateral * 0.18
        param = np.array([sign, abs(rot), 0.0])
    elif intent == "lift":
        delta = np.array([0.02 * rng.normal(), 0.02 * rng.normal(), rng.uniform(0.26, 0.38)])
        rot = 0.0
        target = obj + delta
        param = np.array([0.0, 0.0, 1.0])
    elif intent == "place":
        target = np.array([rng.uniform(-0.45, 0.45), rng.uniform(0.40, 0.68), rng.uniform(0.02, 0.10)])
        delta = target - obj
        delta[2] = rng.uniform(0.02, 0.08)
        rot = 0.0
        param = unit(delta)
    elif intent == "handoff":
        target = np.array([rng.uniform(-0.60, 0.60), rng.uniform(0.70, 0.95), rng.uniform(0.35, 0.55)])
        delta = target - obj
        rot = 0.0
        param = unit(delta)
    else:
        raise ValueError(intent)
    return obj, target, delta, rot, param


def make_trajectory(intent, shoulder, arm_length, handedness, object_pos, target, object_delta, rotation_delta, speed, curvature, hesitation, rng):
    start = shoulder + np.array([0.18 * handedness * arm_length, -0.30 * arm_length, 0.02])
    contact_idx = int(np.clip(round((0.42 + 0.12 * hesitation) * T / max(0.55, speed)), 7, 16))
    hand = np.zeros((T, 3))
    pre_normal = unit(np.array([-(object_pos - start)[1], (object_pos - start)[0], 0.0]))
    for t in range(T):
        if t <= contact_idx:
            a = t / max(1, contact_idx)
            ease = a * a * (3 - 2 * a)
            curve = math.sin(math.pi * a) * curvature * arm_length
            hand[t] = (1 - ease) * start + ease * object_pos + curve * pre_normal
        else:
            a = (t - contact_idx) / max(1, T - contact_idx - 1)
            ease = a * a * (3 - 2 * a)
            if intent == "rotate":
                radius = 0.16 * arm_length
                theta0 = math.atan2(hand[contact_idx, 1] - object_pos[1], hand[contact_idx, 0] - object_pos[0])
                theta = theta0 + rotation_delta * ease
                hand[t] = object_pos + np.array([math.cos(theta), math.sin(theta), 0.08]) * radius
            elif intent in {"lift", "handoff"}:
                hand[t] = object_pos + ease * object_delta
                hand[t, 2] += 0.10 * math.sin(math.pi * ease)
            elif intent == "place":
                lift_arc = np.array([0.0, 0.0, 0.16 * math.sin(math.pi * ease)])
                hand[t] = object_pos + ease * object_delta + lift_arc
            else:
                hand[t] = object_pos + ease * object_delta + curvature * 0.30 * math.sin(math.pi * ease) * pre_normal
    wobble = rng.normal(0.0, 0.012 + 0.018 * curvature, size=hand.shape)
    hand += wobble
    elbow = shoulder[None, :] + 0.52 * (hand - shoulder[None, :])
    elbow[:, 1] += 0.09 * handedness * np.sin(np.linspace(0.0, math.pi, T))
    shoulder_seq = np.repeat(shoulder[None, :], T, axis=0)
    return shoulder_seq, elbow, hand, contact_idx / (T - 1)


def apply_camera_and_noise(arr, camera_scale, occlusion, noise, rng):
    observed = arr.copy() * camera_scale
    observed += rng.normal(0.0, noise, size=observed.shape)
    if occlusion > 0:
        mask = rng.random(observed.shape[0]) < occlusion
        for idx in np.where(mask)[0]:
            if 0 < idx < observed.shape[0] - 1:
                observed[idx] = 0.5 * (observed[idx - 1] + observed[idx + 1])
            observed[idx] += rng.normal(0.0, 2.5 * noise, size=observed.shape[1])
    return observed


def make_episode(split, seed, episode_id, intent=None, stress=0.0):
    rng = stable_rng("episode", split, seed, episode_id, int(1000 * stress))
    if intent is None:
        intent = INTENTS[episode_id % len(INTENTS)]
    morphology, arm_length, shoulder_width, camera_scale = morphology_params(split, rng, stress)
    speed, curvature, hesitation, occlusion, noise = style_params(split, rng, stress)
    handedness = int(rng.choice([-1, 1]))
    if split in {"morphology_shift", "combined_hard_shift", "stress_morphology"} and rng.random() < 0.55:
        handedness *= -1
    object_class, affordance = choose_object(intent, split, rng, stress)
    object_pos, target, object_delta, rotation_delta, param = intent_geometry(intent, rng)
    shoulder = np.array([handedness * shoulder_width / 2.0, -0.52 * arm_length, 0.86 * arm_length])
    shoulder_seq, elbow, hand, contact_time = make_trajectory(
        intent,
        shoulder,
        arm_length,
        handedness,
        object_pos,
        target,
        object_delta,
        rotation_delta,
        speed,
        curvature,
        hesitation,
        rng,
    )
    shoulder_obs = apply_camera_and_noise(shoulder_seq, camera_scale, occlusion * 0.35, noise * 0.50, rng)
    elbow_obs = apply_camera_and_noise(elbow, camera_scale, occlusion * 0.60, noise * 0.85, rng)
    hand_obs = apply_camera_and_noise(hand, camera_scale, occlusion, noise, rng)
    obj_obs = object_pos * camera_scale + rng.normal(0.0, noise, size=3)
    delta_obs = object_delta * camera_scale + rng.normal(0.0, noise * 1.4, size=3)
    return Episode(
        split=split,
        seed=seed,
        episode_id=episode_id,
        intent=intent,
        morphology=morphology,
        arm_length=arm_length,
        shoulder_width=shoulder_width,
        handedness=handedness,
        camera_scale=camera_scale,
        style_speed=speed,
        curvature=curvature,
        hesitation=hesitation,
        occlusion=occlusion,
        noise=noise,
        object_class=object_class,
        affordance=affordance,
        shoulder=shoulder_obs,
        elbow=elbow_obs,
        hand=hand_obs,
        object_pos=obj_obs,
        object_delta=delta_obs,
        rotation_delta=float(rotation_delta + rng.normal(0.0, noise * 2.0)),
        contact_time=float(np.clip(contact_time + rng.normal(0.0, noise), 0.05, 0.95)),
        target=target * camera_scale + rng.normal(0.0, noise, size=3),
        true_param=param,
    )


def estimate_arm_length(ep):
    upper = np.linalg.norm(ep.elbow - ep.shoulder, axis=1)
    lower = np.linalg.norm(ep.hand - ep.elbow, axis=1)
    return float(np.median(upper + lower))


def estimate_handedness(ep):
    rel = ep.hand[0] - ep.shoulder[0]
    return -1 if rel[0] < 0 else 1


def resample_flat(points, dims=3):
    idx = np.linspace(0, len(points) - 1, 12).round().astype(int)
    return points[idx, :dims].reshape(-1)


def summary_kinematics(ep, normalized=True, mirror=True, include_object=True, include_affordance=True, include_contact=True, include_style=True):
    arm = estimate_arm_length(ep) if normalized else 1.0
    hand = ep.hand.copy()
    shoulder = ep.shoulder.copy()
    obj = ep.object_pos.copy()
    delta = ep.object_delta.copy()
    target = ep.target.copy()
    if normalized:
        hand = (hand - shoulder[0]) / max(arm, 1e-6)
        obj = (obj - shoulder[0]) / max(arm, 1e-6)
        delta = delta / max(arm, 1e-6)
        target = (target - shoulder[0]) / max(arm, 1e-6)
    if mirror:
        sign = estimate_handedness(ep)
        hand[:, 0] *= sign
        obj[0] *= sign
        delta[0] *= sign
        target[0] *= sign
    vel = np.diff(hand, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    disp = hand[-1] - hand[0]
    contact_idx = int(np.clip(round(ep.contact_time * (T - 1)), 1, T - 2))
    post = hand[-1] - hand[contact_idx]
    pre = hand[contact_idx] - hand[0]
    curvature = float(np.mean(np.linalg.norm(np.diff(vel, axis=0), axis=1)))
    feats = [
        *resample_flat(hand, dims=3),
        *disp,
        *pre,
        *post,
        float(np.mean(speed)),
        float(np.std(speed)),
        curvature,
        float(np.max(hand[:, 2]) - np.min(hand[:, 2])),
    ]
    if include_object:
        feats += [*obj, *delta, float(ep.rotation_delta)]
        feats += [float(np.linalg.norm(delta[:2])), float(delta[2]), float(np.linalg.norm(target - obj))]
    if include_affordance:
        feats += list(ep.affordance)
    if include_contact:
        feats += [ep.contact_time, float(np.linalg.norm(hand[contact_idx] - obj))]
    if include_style:
        feats += [float(np.max(speed)), float(np.argmin(np.abs(np.arange(T) - contact_idx)) / T)]
    return np.asarray(feats, dtype=float)


def factored_physical_features(ep, ablation=None):
    normalized = ablation != "minus_body_scale_normalization"
    mirror = ablation != "minus_handedness_mirroring"
    include_affordance = ablation != "minus_object_affordance"
    include_contact = ablation != "minus_contact_timing"
    include_style = ablation != "minus_style_speed_normalization"

    arm = estimate_arm_length(ep) if normalized else 1.0
    shoulder = ep.shoulder[0]
    hand = (ep.hand - shoulder) / max(arm, 1e-6)
    obj = (ep.object_pos - shoulder) / max(arm, 1e-6)
    delta = ep.object_delta / max(arm, 1e-6)
    target = (ep.target - shoulder) / max(arm, 1e-6)
    if mirror:
        handed = estimate_handedness(ep)
        hand[:, 0] *= handed
        obj[0] *= handed
        delta[0] *= handed
        target[0] *= handed
    contact_idx = int(np.clip(round(ep.contact_time * (T - 1)), 1, T - 2))
    vel = np.diff(hand, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    pre = hand[contact_idx] - hand[0]
    post = hand[-1] - hand[contact_idx]
    target_rel = target - obj
    xy_delta = float(np.linalg.norm(delta[:2]))
    body_axis = unit(np.r_[obj[:2], 0.0])[:2]
    delta_axis = unit(np.r_[delta[:2], 0.0])[:2]
    signed_motion = float(np.dot(body_axis, delta_axis))
    feats = [
        *obj,
        *delta,
        *target_rel,
        float(ep.rotation_delta),
        xy_delta,
        float(delta[2]),
        signed_motion,
        float(np.max(hand[:, 2]) - np.min(hand[:, 2])),
        *pre,
        *post,
    ]
    if include_affordance:
        feats += list(ep.affordance)
    if include_contact:
        feats += [ep.contact_time, float(np.linalg.norm(hand[contact_idx] - obj))]
    if include_style:
        feats += [float(np.mean(speed)), float(np.std(speed))]
    return np.asarray(feats, dtype=float)


def feature_vector(ep, method, ablation=None):
    if method == "raw_trajectory_knn":
        return np.r_[resample_flat(ep.shoulder), resample_flat(ep.elbow), resample_flat(ep.hand), ep.object_pos, ep.object_delta]
    if method == "body_normalized_knn":
        return summary_kinematics(ep, normalized=True, mirror=True, include_object=False, include_affordance=False, include_contact=False)
    if method == "velocity_contact_classifier":
        return summary_kinematics(ep, normalized=False, mirror=False, include_object=True, include_affordance=False, include_contact=True, include_style=True)[-22:]
    if method == "object_affordance_classifier":
        return np.r_[ep.object_delta, ep.rotation_delta, ep.affordance, ep.contact_time, np.linalg.norm(ep.target - ep.object_pos)]
    if method == "style_invariant_logistic":
        return summary_kinematics(ep, normalized=True, mirror=True, include_object=True, include_affordance=False, include_contact=True, include_style=False)
    if method == "physical_intent_disambiguation":
        return factored_physical_features(ep, ablation=ablation)
    if method == "oracle_intent_upper_bound":
        return np.zeros(3)
    raise ValueError(method)


def train_prototype_model(train_eps, method, ablation=None):
    features = [feature_vector(ep, method, ablation=ablation) for ep in train_eps]
    labels = [ep.intent for ep in train_eps]
    arrays = np.asarray(features)
    mean = np.mean(arrays, axis=0)
    std = np.std(arrays, axis=0)
    scaled = safe_standardize(arrays, mean, std)
    prototypes = {}
    param_means = {}
    for intent in INTENTS:
        idx = [i for i, y in enumerate(labels) if y == intent]
        prototypes[intent] = np.mean(scaled[idx], axis=0)
        param_means[intent] = np.mean([train_eps[i].true_param for i in idx], axis=0)
    return {"features": scaled, "labels": labels, "mean": mean, "std": std, "prototypes": prototypes, "param_means": param_means}


def predict_prototype(model, feature):
    xs = safe_standardize(np.asarray(feature, dtype=float), model["mean"], model["std"])
    dists = {intent: float(np.linalg.norm(xs - proto)) for intent, proto in model["prototypes"].items()}
    ordered = sorted(dists.items(), key=lambda kv: kv[1])
    pred = ordered[0][0]
    confidence = float(1.0 / (1.0 + math.exp(ordered[0][1] - ordered[1][1])))
    confidence = 0.50 + 0.50 * confidence
    return pred, confidence, dists


def train_knn_model(train_eps, method, ablation=None):
    features = [feature_vector(ep, method, ablation=ablation) for ep in train_eps]
    labels = [ep.intent for ep in train_eps]
    arrays = np.asarray(features)
    mean = np.mean(arrays, axis=0)
    std = np.std(arrays, axis=0)
    scaled = safe_standardize(arrays, mean, std)
    params = [ep.true_param for ep in train_eps]
    return {"features": scaled, "labels": labels, "params": params, "mean": mean, "std": std}


def predict_knn(model, feature, k=5):
    xs = safe_standardize(np.asarray(feature, dtype=float), model["mean"], model["std"])
    d = np.linalg.norm(model["features"] - xs[None, :], axis=1)
    order = np.argsort(d)[:k]
    votes = {intent: 0.0 for intent in INTENTS}
    for idx in order:
        votes[model["labels"][int(idx)]] += 1.0 / (float(d[int(idx)]) + 1e-6)
    pred = max(votes.items(), key=lambda kv: kv[1])[0]
    total = sum(votes.values())
    confidence = votes[pred] / max(total, 1e-6)
    nearest_same = [idx for idx in order if model["labels"][int(idx)] == pred]
    param = model["params"][int(nearest_same[0] if nearest_same else order[0])]
    return pred, float(confidence), np.asarray(param, dtype=float)


def add_vote(scores, label, weight):
    scores[label] = scores.get(label, 0.0) + float(weight)


def physical_rule_scores(ep, ablation=None):
    scores = {intent: 0.05 for intent in INTENTS}
    arm = estimate_arm_length(ep) if ablation != "minus_body_scale_normalization" else 1.0
    shoulder = ep.shoulder[0]
    obj = (ep.object_pos - shoulder) / max(arm, 1e-6)
    delta = ep.object_delta / max(arm, 1e-6)
    target = (ep.target - shoulder) / max(arm, 1e-6)
    if ablation == "minus_handedness_mirroring":
        handed = 1
    else:
        handed = estimate_handedness(ep)
    obj[0] *= handed
    delta[0] *= handed
    target[0] *= handed

    affordance = ep.affordance if ablation != "minus_object_affordance" else np.zeros_like(ep.affordance)
    xy_delta = float(np.linalg.norm(delta[:2]))
    z_delta = float(delta[2])
    rot = abs(float(ep.rotation_delta))
    body_axis = unit(np.r_[obj[:2], 0.0])[:2]
    delta_axis = unit(np.r_[delta[:2], 0.0])[:2]
    signed_motion = float(np.dot(body_axis, delta_axis))
    target_height = float(target[2] - obj[2])
    target_forward = float(target[1] - obj[1])

    if rot > 0.18:
        scores["rotate"] += 1.4 + 1.0 * affordance[2] + min(rot, 1.4)
    if z_delta > 0.14:
        scores["lift"] += 1.0 + 0.7 * affordance[3] + min(z_delta, 0.8)
    if xy_delta > 0.10 and z_delta < 0.18 and rot < 0.28:
        scores["push"] += 1.05 * max(0.0, signed_motion) + 0.35 * affordance[0]
        scores["pull"] += 1.05 * max(0.0, -signed_motion) + 0.35 * affordance[1]
    if target_forward > 0.24 and target_height < 0.22 and xy_delta > 0.18:
        scores["place"] += 1.15 + 0.9 * affordance[4]
    if target_height > 0.22 or (target_forward > 0.34 and target[2] > 0.20):
        scores["handoff"] += 1.25 + 0.9 * affordance[5]
    if ablation == "minus_contact_timing":
        scores["rotate"] *= 0.90
        scores["lift"] *= 0.92
    else:
        if 0.25 <= ep.contact_time <= 0.75:
            for label in scores:
                scores[label] *= 1.03
        else:
            scores["push"] *= 0.94
            scores["pull"] *= 0.94
    if ablation == "minus_style_speed_normalization" and (ep.style_speed < 0.70 or ep.style_speed > 1.45):
        scores["push"] *= 0.88
        scores["pull"] *= 0.88
        scores["place"] *= 0.92
        scores["handoff"] *= 0.92
    return scores


def fused_physical_prediction(ep, models, ablation=None):
    scores = {intent: 0.0 for intent in INTENTS}
    if ablation != "minus_object_affordance":
        obj_feat = feature_vector(ep, "object_affordance_classifier")
        obj_pred, obj_conf, _ = predict_prototype(models["object_affordance_classifier"], obj_feat)
        add_vote(scores, obj_pred, 1.00 * obj_conf)

    if ablation in {"minus_body_scale_normalization", "minus_handedness_mirroring"}:
        raw_feat = feature_vector(ep, "raw_trajectory_knn")
        raw_pred, raw_conf, _ = predict_knn(models["raw_trajectory_knn"], raw_feat)
        add_vote(scores, raw_pred, 0.30 * raw_conf)
    else:
        body_feat = feature_vector(ep, "body_normalized_knn")
        body_pred, body_conf, _ = predict_knn(models["body_normalized_knn"], body_feat)
        add_vote(scores, body_pred, 0.45 * body_conf)

    if ablation != "minus_style_speed_normalization":
        style_feat = feature_vector(ep, "style_invariant_logistic")
        style_pred, style_conf, _ = predict_softmax(models["style_invariant_logistic"], style_feat)
        add_vote(scores, style_pred, 0.55 * style_conf)

    if ablation != "minus_contact_timing":
        vc_feat = feature_vector(ep, "velocity_contact_classifier")
        vc_pred, vc_conf, _ = predict_prototype(models["velocity_contact_classifier"], vc_feat)
        add_vote(scores, vc_pred, 0.16 * vc_conf)

    proto_feat = feature_vector(ep, "physical_intent_disambiguation", ablation=ablation)
    proto_pred, proto_conf, _ = predict_prototype(models["physical_intent_disambiguation"], proto_feat)
    add_vote(scores, proto_pred, 0.35 * proto_conf)

    rule_scores = physical_rule_scores(ep, ablation=ablation)
    rule_weight = 0.95 if ablation != "minus_object_affordance" else 0.45
    for label, value in rule_scores.items():
        add_vote(scores, label, rule_weight * value)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    pred = ordered[0][0]
    top = ordered[0][1]
    second = ordered[1][1]
    confidence = 0.50 + 0.49 * max(0.0, (top - second) / max(top, 1e-6))
    return pred, float(min(0.99, confidence)), physical_param_from_observation(ep, pred)


def make_train_eps(seed):
    eps = []
    idx = 0
    for intent in INTENTS:
        for j in range(TRAIN_EPISODES_PER_INTENT):
            eps.append(make_episode("train", seed, idx, intent=intent))
            idx += 1
    return eps


def train_models(seed, ablation=None):
    train_eps = make_train_eps(seed)
    models = {}
    for method in METHODS:
        if method == "oracle_intent_upper_bound":
            continue
        if method in {"raw_trajectory_knn", "body_normalized_knn"}:
            models[method] = train_knn_model(train_eps, method, ablation=ablation)
        elif method == "style_invariant_logistic":
            features = [feature_vector(ep, method, ablation=ablation) for ep in train_eps]
            labels = [ep.intent for ep in train_eps]
            models[method] = train_softmax_classifier(features, labels)
        else:
            models[method] = train_prototype_model(train_eps, method, ablation=ablation)
    return train_eps, models


def physical_param_from_observation(ep, intent):
    delta = ep.object_delta.copy()
    if np.linalg.norm(delta) < 1e-6:
        delta = ep.target - ep.object_pos
    if intent == "rotate":
        return np.array([1.0 if ep.rotation_delta >= 0 else -1.0, abs(ep.rotation_delta), 0.0])
    if intent == "lift":
        return np.array([0.0, 0.0, 1.0])
    return unit(delta)


def action_success(pred, ep, param, confidence, method, ablation=None):
    correct = int(pred == ep.intent)
    if pred == "rotate":
        sign_ok = np.sign(param[0]) == np.sign(ep.true_param[0])
        mag_error = abs(abs(param[1]) - abs(ep.true_param[1]))
        param_error = 0.0 if sign_ok else 1.0
        param_error += min(1.0, mag_error)
    else:
        param_error = float(np.linalg.norm(unit(param) - unit(ep.true_param)))
    abstain = 0
    if method == "physical_intent_disambiguation" and ablation != "minus_uncertainty_gate":
        if confidence < 0.52:
            abstain = 1
    threshold = 0.52 if ep.intent in {"place", "handoff"} else 0.45
    success = int(correct and param_error <= threshold and not abstain)
    if not correct:
        failure = "wrong_intent"
    elif abstain:
        failure = "abstain_low_confidence"
    elif param_error > threshold:
        failure = "bad_action_parameter"
    else:
        failure = "success"
    return success, correct, float(param_error), abstain, failure


def predict_method(ep, method, models, ablation=None):
    if method == "oracle_intent_upper_bound":
        pred = ep.intent
        confidence = 0.985
        param = ep.true_param + stable_rng("oracle_param", ep.split, ep.seed, ep.episode_id).normal(0.0, 0.018, size=3)
        return pred, confidence, param
    if method == "physical_intent_disambiguation":
        return fused_physical_prediction(ep, models, ablation=ablation)
    feat = feature_vector(ep, method, ablation=ablation)
    if method in {"raw_trajectory_knn", "body_normalized_knn"}:
        return predict_knn(models[method], feat)
    if method == "style_invariant_logistic":
        pred, confidence, _ = predict_softmax(models[method], feat)
        param = physical_param_from_observation(ep, pred)
        return pred, confidence, param
    pred, confidence, _ = predict_prototype(models[method], feat)
    param = models[method]["param_means"][pred]
    if method == "object_affordance_classifier":
        param = 0.65 * param + 0.35 * physical_param_from_observation(ep, pred)
    return pred, confidence, np.asarray(param, dtype=float)


def run_prediction(ep, method, models, ablation=None):
    pred, confidence, param = predict_method(ep, method, models, ablation=ablation)
    success, correct, param_error, abstain, failure = action_success(pred, ep, param, confidence, method, ablation=ablation)
    return {
        "split": ep.split,
        "seed": ep.seed,
        "episode_id": ep.episode_id,
        "method": method if ablation is None else ablation,
        "true_intent": ep.intent,
        "pred_intent": pred,
        "intent_correct": correct,
        "action_success": success,
        "param_error": f"{param_error:.5f}",
        "confidence": f"{confidence:.5f}",
        "calibration_error": f"{abs(confidence - correct):.5f}",
        "abstain": abstain,
        "failure_label": failure,
        "morphology": ep.morphology,
        "arm_length": f"{ep.arm_length:.5f}",
        "camera_scale": f"{ep.camera_scale:.5f}",
        "style_speed": f"{ep.style_speed:.5f}",
        "curvature": f"{ep.curvature:.5f}",
        "occlusion": f"{ep.occlusion:.5f}",
        "noise": f"{ep.noise:.5f}",
        "object_class": ep.object_class,
        "contact_time": f"{ep.contact_time:.5f}",
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def leakage_score(train_eps, test_eps, method, ablation=None):
    labels = []
    train_features = []
    for ep in train_eps:
        train_features.append(feature_vector(ep, method, ablation=ablation))
        labels.append(0 if ep.arm_length < 1.0 else 1)
    train_features = np.asarray(train_features)
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    train_features = safe_standardize(train_features, mean, std)
    centroids = {}
    global_mean = np.mean(train_features, axis=0)
    for group in [0, 1]:
        grouped = [train_features[i] for i, lab in enumerate(labels) if lab == group]
        centroids[group] = np.mean(grouped, axis=0) if grouped else global_mean
    correct = []
    for ep in test_eps:
        feat = safe_standardize(feature_vector(ep, method, ablation=ablation), mean, std)
        pred = min(centroids.items(), key=lambda kv: float(np.linalg.norm(feat - kv[1])))[0]
        true = 0 if ep.arm_length < 1.0 else 1
        correct.append(int(pred == true))
    return float(np.mean(correct))


def aggregate_seed_metrics(rows, leakage_rows=None):
    out = []
    for split in sorted({r["split"] for r in rows}):
        for method in METHODS:
            for seed in SEEDS:
                vals = [r for r in rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                if not vals:
                    continue
                leakage = ""
                if leakage_rows:
                    matches = [r for r in leakage_rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                    leakage = f"{float(matches[0]['morphology_leakage']):.5f}" if matches else ""
                out.append(
                    {
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "episodes": len(vals),
                        "intent_accuracy": f"{np.mean([int(v['intent_correct']) for v in vals]):.5f}",
                        "action_success": f"{np.mean([int(v['action_success']) for v in vals]):.5f}",
                        "param_error": f"{np.mean([float(v['param_error']) for v in vals]):.5f}",
                        "calibration_error": f"{np.mean([float(v['calibration_error']) for v in vals]):.5f}",
                        "abstain_rate": f"{np.mean([int(v['abstain']) for v in vals]):.5f}",
                        "morphology_leakage": leakage,
                    }
                )
    return out


def aggregate_metrics(seed_rows):
    out = []
    metrics = ["intent_accuracy", "action_success", "param_error", "calibration_error", "abstain_rate", "morphology_leakage"]
    for split in sorted({r["split"] for r in seed_rows}):
        for method in METHODS:
            vals = [r for r in seed_rows if r["split"] == split and r["method"] == method]
            if not vals:
                continue
            for metric in metrics:
                nums = [float(v[metric]) for v in vals if v[metric] != ""]
                if not nums:
                    continue
                out.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "mean": f"{np.mean(nums):.5f}",
                        "ci95": f"{ci95(nums):.5f}",
                        "seeds": len(nums),
                        "episodes_per_seed": vals[0]["episodes"],
                    }
                )
    return out


def pairwise_stats(seed_rows):
    rows = []
    refs = ["body_normalized_knn", "object_affordance_classifier", "style_invariant_logistic"]
    metrics = ["intent_accuracy", "action_success", "param_error", "calibration_error", "morphology_leakage"]
    for split in sorted({r["split"] for r in seed_rows}):
        for ref in refs:
            for metric in metrics:
                diffs = []
                for seed in SEEDS:
                    target = [r for r in seed_rows if r["split"] == split and r["method"] == "physical_intent_disambiguation" and int(r["seed"]) == seed]
                    base = [r for r in seed_rows if r["split"] == split and r["method"] == ref and int(r["seed"]) == seed]
                    if target and base and target[0][metric] != "" and base[0][metric] != "":
                        diffs.append(float(target[0][metric]) - float(base[0][metric]))
                higher_better = metric in {"intent_accuracy", "action_success"}
                rows.append(
                    {
                        "split": split,
                        "target": "physical_intent_disambiguation",
                        "reference": ref,
                        "metric": metric,
                        "mean_diff": f"{np.mean(diffs):.5f}",
                        "ci95": f"{ci95(diffs):.5f}",
                        "target_better_seeds": sum(1 for d in diffs if (d > 0 if higher_better else d < 0)),
                        "seeds": len(diffs),
                    }
                )
    return rows


def metric_value(metric_rows, split, method, metric):
    rows = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not rows:
        return 0.0, 0.0
    return float(rows[0]["mean"]), float(rows[0]["ci95"])


def dataset_row(ep):
    return {
        "split": ep.split,
        "seed": ep.seed,
        "episode_id": ep.episode_id,
        "intent": ep.intent,
        "morphology": ep.morphology,
        "arm_length": f"{ep.arm_length:.5f}",
        "shoulder_width": f"{ep.shoulder_width:.5f}",
        "handedness": ep.handedness,
        "camera_scale": f"{ep.camera_scale:.5f}",
        "style_speed": f"{ep.style_speed:.5f}",
        "curvature": f"{ep.curvature:.5f}",
        "hesitation": f"{ep.hesitation:.5f}",
        "occlusion": f"{ep.occlusion:.5f}",
        "noise": f"{ep.noise:.5f}",
        "object_class": ep.object_class,
        "contact_time": f"{ep.contact_time:.5f}",
    }


def run_main():
    splits = ["seen_morphology_clean", "style_shift_same_intent", "morphology_shift", "object_affordance_ambiguity", "combined_hard_shift"]
    rows = []
    dataset = []
    leakage_rows = []
    for seed in SEEDS:
        train_eps, models = train_models(seed)
        for split in splits:
            test_eps = [make_episode(split, seed, episode_id) for episode_id in range(TEST_EPISODES_PER_SPLIT_SEED)]
            dataset += [dataset_row(ep) for ep in test_eps]
            for method in METHODS:
                if method != "oracle_intent_upper_bound":
                    leakage = leakage_score(train_eps, test_eps, method)
                    leakage_rows.append({"split": split, "method": method, "seed": seed, "morphology_leakage": f"{leakage:.5f}"})
                for ep in test_eps:
                    rows.append(run_prediction(ep, method, models))
            print(f"main seed={seed} split={split} rows={len(rows)}", flush=True)
    seed_rows = aggregate_seed_metrics(rows, leakage_rows)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "dataset_summary.csv", dataset)
    write_csv(RESULTS / "morphology_leakage.csv", leakage_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


ABLATIONS = [
    "full_physical_intent_disambiguation",
    "minus_body_scale_normalization",
    "minus_handedness_mirroring",
    "minus_object_affordance",
    "minus_contact_timing",
    "minus_style_speed_normalization",
    "minus_uncertainty_gate",
]


def run_ablation():
    rows = []
    summary = []
    for seed in SEEDS:
        train_eps, models = train_models(seed)
        ablation_models = {"full_physical_intent_disambiguation": models}
        for ablation in ABLATIONS:
            if ablation in {"full_physical_intent_disambiguation", "minus_uncertainty_gate"}:
                continue
            local_models = dict(models)
            local_models["physical_intent_disambiguation"] = train_prototype_model(
                train_eps,
                "physical_intent_disambiguation",
                ablation=ablation,
            )
            ablation_models[ablation] = local_models
        ablation_models["minus_uncertainty_gate"] = models
        for ep_id in range(TEST_EPISODES_PER_SPLIT_SEED):
            ep = make_episode("combined_hard_shift", seed, ep_id)
            for ablation in ABLATIONS:
                local = None if ablation == "full_physical_intent_disambiguation" else ablation
                models_local = ablation_models[ablation]
                rows.append(run_prediction(ep, "physical_intent_disambiguation", models_local, ablation=local) | {"ablation": ablation})
        print(f"ablation seed={seed} rows={len(rows)}", flush=True)
    for ablation in ABLATIONS:
        vals = [r for r in rows if r["ablation"] == ablation]
        seed_acc, seed_success, seed_error, seed_abstain = [], [], [], []
        for seed in SEEDS:
            seed_vals = [r for r in vals if int(r["seed"]) == seed]
            seed_acc.append(np.mean([int(v["intent_correct"]) for v in seed_vals]))
            seed_success.append(np.mean([int(v["action_success"]) for v in seed_vals]))
            seed_error.append(np.mean([float(v["param_error"]) for v in seed_vals]))
            seed_abstain.append(np.mean([int(v["abstain"]) for v in seed_vals]))
        summary.append(
            {
                "split": "combined_hard_shift",
                "ablation": ablation,
                "intent_accuracy": f"{np.mean(seed_acc):.5f}",
                "action_success": f"{np.mean(seed_success):.5f}",
                "ci95_success": f"{ci95(seed_success):.5f}",
                "param_error": f"{np.mean(seed_error):.5f}",
                "abstain_rate": f"{np.mean(seed_abstain):.5f}",
                "rows": len(vals),
            }
        )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def stress_split_for_axis(axis):
    if axis == "morphology":
        return "stress_morphology"
    if axis == "style":
        return "stress_style"
    if axis == "affordance":
        return "stress_affordance"
    return "combined_hard_shift"


def run_stress():
    axes = ["morphology", "style", "affordance", "combined"]
    methods = ["body_normalized_knn", "object_affordance_classifier", "style_invariant_logistic", "physical_intent_disambiguation", "oracle_intent_upper_bound"]
    raw = []
    summary = []
    for seed in SEEDS:
        _, models = train_models(seed)
        for axis in axes:
            split = stress_split_for_axis(axis)
            for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                for ep_id in range(STRESS_EPISODES_PER_SEED):
                    ep = make_episode(split, seed, ep_id, stress=level)
                    for method in methods:
                        row = run_prediction(ep, method, models)
                        row["stress_axis"] = axis
                        row["stress_level"] = f"{level:.1f}"
                        raw.append(row)
                print(f"stress seed={seed} axis={axis} level={level:.1f} rows={len(raw)}", flush=True)
    for axis in axes:
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for method in methods:
                vals = [r for r in raw if r["stress_axis"] == axis and r["stress_level"] == f"{level:.1f}" and r["method"] == method]
                seed_success, seed_acc, seed_error = [], [], []
                for seed in SEEDS:
                    seed_vals = [r for r in vals if int(r["seed"]) == seed]
                    seed_success.append(np.mean([int(v["action_success"]) for v in seed_vals]))
                    seed_acc.append(np.mean([int(v["intent_correct"]) for v in seed_vals]))
                    seed_error.append(np.mean([float(v["param_error"]) for v in seed_vals]))
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": f"{level:.1f}",
                        "method": method,
                        "intent_accuracy": f"{np.mean(seed_acc):.5f}",
                        "action_success": f"{np.mean(seed_success):.5f}",
                        "ci95_success": f"{ci95(seed_success):.5f}",
                        "param_error": f"{np.mean(seed_error):.5f}",
                        "rows": len(vals),
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def write_negative_cases(rows):
    failures = [r for r in rows if int(r["action_success"]) == 0]
    lessons = {
        "wrong_intent": "human motion was compatible with the wrong physical intent after style or morphology shift",
        "bad_action_parameter": "intent was classified but robot action geometry was not recoverable",
        "abstain_low_confidence": "uncertainty gate avoided committing to an unsafe robot primitive",
    }
    out = []
    seen = set()
    for r in failures:
        key = (r["split"], r["method"], r["failure_label"], r["true_intent"], r["pred_intent"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "split": r["split"],
                "seed": r["seed"],
                "episode_id": r["episode_id"],
                "method": r["method"],
                "true_intent": r["true_intent"],
                "pred_intent": r["pred_intent"],
                "failure_label": r["failure_label"],
                "confidence": r["confidence"],
                "param_error": r["param_error"],
                "lesson": lessons.get(r["failure_label"], "negative case retained for audit"),
            }
        )
        if len(out) >= 16:
            break
    write_csv(RESULTS / "negative_cases.csv", out)


def terminal_decision(metric_rows, pair_rows, ablation_summary):
    combined = metric_value(metric_rows, "combined_hard_shift", "physical_intent_disambiguation", "action_success")
    obj = metric_value(metric_rows, "combined_hard_shift", "object_affordance_classifier", "action_success")
    logistic = metric_value(metric_rows, "combined_hard_shift", "style_invariant_logistic", "action_success")
    body = metric_value(metric_rows, "combined_hard_shift", "body_normalized_knn", "action_success")
    raw_leak = metric_value(metric_rows, "combined_hard_shift", "raw_trajectory_knn", "morphology_leakage")
    prop_leak = metric_value(metric_rows, "combined_hard_shift", "physical_intent_disambiguation", "morphology_leakage")
    best_base = max(obj[0], logistic[0], body[0])
    diff_obj = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "object_affordance_classifier" and r["metric"] == "action_success"][0]
    diff_log = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "style_invariant_logistic" and r["metric"] == "action_success"][0]
    full = [r for r in ablation_summary if r["ablation"] == "full_physical_intent_disambiguation"][0]
    no_aff = [r for r in ablation_summary if r["ablation"] == "minus_object_affordance"][0]
    object_drop = float(full["action_success"]) - float(no_aff["action_success"])
    leakage_drop = raw_leak[0] - prop_leak[0]
    if (
        combined[0] >= best_base + 0.08
        and float(diff_obj["mean_diff"]) > 0.05
        and float(diff_log["mean_diff"]) > 0.05
        and object_drop >= 0.08
        and leakage_drop >= 0.10
    ):
        return "STRONG_REVISE"
    return "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, rollout_rows, ablation_rows, stress_raw):
    decision = terminal_decision(metric_rows, pair_rows, ablation_summary)
    combined_prop = metric_value(metric_rows, "combined_hard_shift", "physical_intent_disambiguation", "action_success")
    combined_obj = metric_value(metric_rows, "combined_hard_shift", "object_affordance_classifier", "action_success")
    combined_log = metric_value(metric_rows, "combined_hard_shift", "style_invariant_logistic", "action_success")
    combined_body = metric_value(metric_rows, "combined_hard_shift", "body_normalized_knn", "action_success")
    combined_oracle = metric_value(metric_rows, "combined_hard_shift", "oracle_intent_upper_bound", "action_success")
    acc_prop = metric_value(metric_rows, "combined_hard_shift", "physical_intent_disambiguation", "intent_accuracy")
    leak_prop = metric_value(metric_rows, "combined_hard_shift", "physical_intent_disambiguation", "morphology_leakage")
    leak_raw = metric_value(metric_rows, "combined_hard_shift", "raw_trajectory_knn", "morphology_leakage")
    diff_obj = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "object_affordance_classifier" and r["metric"] == "action_success"][0]
    diff_log = [r for r in pair_rows if r["split"] == "combined_hard_shift" and r["reference"] == "style_invariant_logistic" and r["metric"] == "action_success"][0]
    stress_max = [r for r in stress_summary if r["stress_axis"] == "combined" and r["stress_level"] == "1.0"]

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 80 human_video_physical_intent_disambiguation v4 rebuild\n")
        f.write(f"Terminal recommendation: {decision}\n")
        f.write(
            "Reason: local keypoint-video benchmark exists, but no real human-video/hardware benchmark is available; "
            "STRONG_REVISE is the maximum honest state even when local evidence is positive.\n"
        )
        f.write(f"Main rollout rows: {len(rollout_rows)}\n")
        f.write(f"Ablation rollout rows: {len(ablation_rows)}\n")
        f.write(f"Stress rollout rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("\nCombined hard-shift action success:\n")
        f.write(f"physical_intent_disambiguation={combined_prop[0]:.5f} ci95={combined_prop[1]:.5f}\n")
        f.write(f"object_affordance_classifier={combined_obj[0]:.5f} ci95={combined_obj[1]:.5f}\n")
        f.write(f"style_invariant_logistic={combined_log[0]:.5f} ci95={combined_log[1]:.5f}\n")
        f.write(f"body_normalized_knn={combined_body[0]:.5f} ci95={combined_body[1]:.5f}\n")
        f.write(f"oracle_intent_upper_bound={combined_oracle[0]:.5f} ci95={combined_oracle[1]:.5f}\n")
        f.write(f"physical intent accuracy={acc_prop[0]:.5f} ci95={acc_prop[1]:.5f}\n")
        f.write(f"morphology leakage raw={leak_raw[0]:.5f}, proposed={leak_prop[0]:.5f}\n")
        f.write(f"paired action-success diff vs object_affordance={diff_obj['mean_diff']} ci95={diff_obj['ci95']}\n")
        f.write(f"paired action-success diff vs style_invariant_logistic={diff_log['mean_diff']} ci95={diff_log['ci95']}\n")
        f.write("\nAblation combined_hard_shift:\n")
        for row in ablation_summary:
            f.write(
                f"{row['ablation']} action_success={row['action_success']} ci95={row['ci95_success']} "
                f"intent_accuracy={row['intent_accuracy']} param_error={row['param_error']} abstain={row['abstain_rate']}\n"
            )
        f.write("\nCombined stress level 1.0:\n")
        for row in stress_max:
            f.write(f"{row['method']} action_success={row['action_success']} ci95={row['ci95_success']} intent_accuracy={row['intent_accuracy']}\n")
    write_negative_cases(rollout_rows)
    return decision


def plot_outputs(metric_rows, ablation_summary, stress_summary):
    methods = METHODS
    vals = [metric_value(metric_rows, "combined_hard_shift", m, "action_success")[0] for m in methods]
    errs = [metric_value(metric_rows, "combined_hard_shift", m, "action_success")[1] for m in methods]
    colors = ["#868e96", "#adb5bd", "#74c0fc", "#4dabf7", "#f08c00", "#2f9e44", "#087f5b"]
    plt.figure(figsize=(11.5, 4.8))
    plt.bar(range(len(methods)), vals, yerr=errs, color=colors, capsize=3)
    plt.xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=7)
    plt.ylim(0, 1.05)
    plt.ylabel("robot action success")
    plt.title("Combined hard-shift human-video intent disambiguation")
    plt.tight_layout()
    plt.savefig(FIGURES / "intent_success_combined.png", dpi=220)
    plt.close()

    leak_methods = [m for m in methods if m != "oracle_intent_upper_bound"]
    vals = [metric_value(metric_rows, "combined_hard_shift", m, "morphology_leakage")[0] for m in leak_methods]
    plt.figure(figsize=(10.5, 4.6))
    plt.bar(range(len(leak_methods)), vals, color="#e8590c")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(len(leak_methods)), [m.replace("_", "\n") for m in leak_methods], fontsize=7)
    plt.ylabel("morphology leakage probe accuracy")
    plt.title("Lower leakage indicates stronger morphology suppression")
    plt.tight_layout()
    plt.savefig(FIGURES / "morphology_leakage.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.2, 4.8))
    plt.bar(range(len(ablation_summary)), [float(r["action_success"]) for r in ablation_summary], yerr=[float(r["ci95_success"]) for r in ablation_summary], color="#f08c00", capsize=3)
    plt.xticks(range(len(ablation_summary)), [r["ablation"].replace("_", "\n") for r in ablation_summary], fontsize=7)
    plt.ylim(0, 1.05)
    plt.ylabel("robot action success")
    plt.title("Physical-intent ablations on combined hard shift")
    plt.tight_layout()
    plt.savefig(FIGURES / "intent_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.2, 5.0))
    for method in ["body_normalized_knn", "object_affordance_classifier", "style_invariant_logistic", "physical_intent_disambiguation", "oracle_intent_upper_bound"]:
        rows = sorted([r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(r["stress_level"]) for r in rows]
        y = [float(r["action_success"]) for r in rows]
        e = [float(r["ci95_success"]) for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3, label=method)
    plt.xlabel("combined stress")
    plt.ylabel("robot action success")
    plt.ylim(0, 1.05)
    plt.title("Morphology/style/affordance stress sweep")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "intent_stress_sweep.png", dpi=220)
    plt.close()


def main():
    rollout_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    decision = write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, rollout_rows, ablation_rows, stress_raw)
    plot_outputs(metric_rows, ablation_summary, stress_summary)
    print(f"terminal={decision}")
    print(f"main_rollouts={len(rollout_rows)} ablation_rollouts={len(ablation_rows)} stress_rollouts={len(stress_raw)}")
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
