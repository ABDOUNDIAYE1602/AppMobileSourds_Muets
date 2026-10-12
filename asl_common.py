"""
asl_common.py
Code PARTAGÉ entre Colab (entraînement) et ton PC (temps réel).
Ne le modifie pas d'un côté sans le modifier de l'autre : c'est ce qui garantit
que le prétraitement est strictement identique à l'entraînement et à l'utilisation.
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
T_FRAMES = 32                       # nombre d'images par signe après rééchantillonnage
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nez, épaules, coudes, poignets (MediaPipe pose)

# Disposition d'un vecteur "image" (184 valeurs)
S_POSE = slice(0, 14)       # 7 points du haut du corps (x, y)
S_LH = slice(14, 56)        # main gauche, coordonnées relatives au corps (21 x 2)
S_RH = slice(56, 98)        # main droite, coordonnées relatives au corps (21 x 2)
S_LHL = slice(98, 140)      # main gauche, forme de la main seule (21 x 2)
S_RHL = slice(140, 182)     # main droite, forme de la main seule (21 x 2)
I_LFLAG = 182               # 1 si la main gauche est détectée
I_RFLAG = 183               # 1 si la main droite est détectée
N_BASE = 184
N_BODY = 98                 # les 98 premières valeurs servent aussi à calculer la vitesse
N_FEAT = N_BASE + N_BODY    # 282 valeurs par image au final


# ----------------------------------------------------------------------------
# MediaPipe (API "Tasks") : réglages identiques partout
# ----------------------------------------------------------------------------
MODEL_URLS = {
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def ensure_model(name, model_dir=None):
    """Télécharge le modèle MediaPipe une seule fois (dossier mp_models à côté de ce fichier)."""
    import urllib.request
    model_dir = model_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    print(f"Téléchargement de {name} ...")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
    except Exception as e:
        raise RuntimeError(
            f"Impossible de télécharger {name} ({e}).\n"
            f"Télécharge-le à la main depuis {MODEL_URLS[name]} et place-le dans {model_dir}"
        )
    os.replace(tmp, path)
    return path


class LandmarkDetector:
    """Détecte le corps (pose) et les mains sur chaque image d'une vidéo / de la webcam."""

    def __init__(self, model_dir=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self._mp = mp
        run = vision.RunningMode.VIDEO
        pose_path = ensure_model("pose_landmarker_full.task", model_dir)
        hand_path = ensure_model("hand_landmarker.task", model_dir)
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=pose_path), running_mode=run, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=hand_path), running_mode=run, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        self._ts = -1

    def process(self, rgb, ts_ms=None):
        """rgb : image RGB uint8 (H, W, 3). Retourne (pose_landmarks ou None, liste de mains)."""
        ts = self._ts + 33 if ts_ms is None else int(ts_ms)
        ts = max(ts, self._ts + 1)             # MediaPipe exige des temps strictement croissants
        self._ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pr = self.pose.detect_for_video(img, ts)
        hr = self.hand.detect_for_video(img, ts)
        pose = pr.pose_landmarks[0] if pr.pose_landmarks else None
        hands = list(hr.hand_landmarks) if hr.hand_landmarks else []
        return pose, hands

    def close(self):
        self.pose.close()
        self.hand.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _lm_array(lms):
    return np.array([[p.x, p.y] for p in lms], dtype=np.float32)


def frame_features(pose_lms, hands_lms, aspect):
    """
    Transforme la détection d'UNE image en vecteur de 184 valeurs.
    pose_lms : 33 points du corps (ou None) | hands_lms : liste de mains (21 points chacune)
    aspect = largeur / hauteur de l'image (pour ne pas déformer les distances).
    Chaque main est attribuée à gauche/droite selon le poignet du corps le plus proche
    (sans dépendre de l'étiquette gauche/droite de MediaPipe).
    Les coordonnées sont centrées sur le milieu des épaules et divisées par la largeur
    des épaules : le résultat ne dépend ni de la distance à la caméra ni de la résolution.
    """
    out = np.zeros(N_BASE, dtype=np.float32)
    if not pose_lms:
        return out
    asp = np.array([aspect, 1.0], dtype=np.float32)
    pose = _lm_array(pose_lms) * asp
    ls, rs = pose[11], pose[12]
    scale = float(np.linalg.norm(ls - rs))
    if scale < 1e-3:
        return out
    origin = (ls + rs) / 2.0
    out[S_POSE] = ((pose[POSE_IDX] - origin) / scale).reshape(-1)

    hands = [_lm_array(h) * asp for h in (hands_lms or [])][:2]
    wl, wr = pose[15], pose[16]
    d = lambda a, b: float(np.linalg.norm(a - b))
    left = right = None
    if len(hands) == 1:
        if d(hands[0][0], wl) <= d(hands[0][0], wr):
            left = hands[0]
        else:
            right = hands[0]
    elif len(hands) == 2:
        straight = d(hands[0][0], wl) + d(hands[1][0], wr)
        swapped = d(hands[0][0], wr) + d(hands[1][0], wl)
        left, right = (hands[0], hands[1]) if straight <= swapped else (hands[1], hands[0])

    for h, s_body, s_loc, flag in ((left, S_LH, S_LHL, I_LFLAG), (right, S_RH, S_RHL, I_RFLAG)):
        if h is None:
            continue
        rel = h - h[0]
        size = float(np.linalg.norm(rel, axis=1).max())
        if size < 1e-6:
            continue
        out[s_body] = ((h - origin) / scale).reshape(-1)
        out[s_loc] = (rel / size).reshape(-1)
        out[flag] = 1.0
    return out


def sequence_from_frames(frames, T=T_FRAMES):
    """(N, 184) -> (T, 282) : rééchantillonne à T images + ajoute la vitesse."""
    frames = np.asarray(frames, dtype=np.float32)
    n = len(frames)
    if n == 0:
        raise ValueError("segment vide")
    idx = np.linspace(0, n - 1, T).round().astype(int)
    seq = frames[idx]
    vel = np.zeros((T, N_BODY), dtype=np.float32)
    vel[1:] = seq[1:, :N_BODY] - seq[:-1, :N_BODY]
    return np.concatenate([seq, vel], axis=1)


# ----------------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------------
class SignNet(nn.Module):
    def __init__(self, n_feat=N_FEAT, n_classes=100, hidden=128, dropout=0.3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_feat))
        self.register_buffer("std", torch.ones(n_feat))
        self.inp = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.GELU()
        )
        self.gru = nn.GRU(
            hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):                      # x : (B, T, n_feat)
        x = (x - self.mean) / self.std
        x = self.inp(x)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(x)
        feat = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(feat)


# ----------------------------------------------------------------------------
# Sauvegarde / chargement / prédiction (ensemble de plusieurs modèles)
# ----------------------------------------------------------------------------
def save_ensemble(path, models, classes, hidden=128):
    torch.save(
        {
            "classes": list(classes),
            "T": T_FRAMES,
            "n_feat": N_FEAT,
            "hidden": hidden,
            "states": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models],
        },
        path,
    )


def load_ensemble(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    models = []
    for st in ck["states"]:
        m = SignNet(ck["n_feat"], len(ck["classes"]), ck["hidden"])
        m.load_state_dict(st)
        m.to(device).eval()
        models.append(m)
    return models, ck["classes"]


@torch.no_grad()
def ensemble_probs(models, x):
    """x : tensor (B, T, n_feat) -> probabilités (B, n_classes), moyenne sur les modèles."""
    device = next(models[0].parameters()).device
    x = x.to(device)
    return torch.stack([torch.softmax(m(x), dim=1) for m in models]).mean(0).cpu()
