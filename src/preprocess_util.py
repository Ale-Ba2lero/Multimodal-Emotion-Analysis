# Taken from https://www.researchgate.net/profile/James-Russell-30/
# publication/222741832_Evidence_for_a_Three-Factor_Theory_of_Emotions/links/
# 5b4c7ccf0f7e9b4637ddf5a0/Evidence-for-a-Three-Factor-Theory-of-Emotions.pdf
#   "Calm" was replaced with "relaxed".
EMOTION_CORE_AFFECT_MAP_LOC = {
    "neutral": (0.0, 0.0),
    "calm": (0.68, -0.46),
    "happy": (0.81, 0.61),
    "sad": (-0.63, -0.27),
    "angry": (-0.51, 0.59),
    "fearful": (-0.64, 0.6),
    "disgust": (-0.6, 0.35),
    "surprise": (0.4, 0.67)
}

EMOTION_CORE_AFFECT_MAP_STD_DEV = {
    "neutral": (0.2, 0.2),
    "calm": (0.3, 0.38),
    "happy": (0.21, 0.26),
    "sad": (0.23, 0.34),
    "angry": (0.2, 0.33),
    "fearful": (0.2, 0.32),
    "disgust": (0.2, 0.41),
    "surprise": (0.3, 0.27)
}


# Taken and adapted (normalized) from openFACS:
#   https://github.com/phuselab/openFACS
PROTOTYPICAL_EXPRESSIONS_NORMALIZED: dict = {
    "angry": [0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "contempt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "disgust": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fearful": [0.6, 0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0],
    "happy": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "shock": [0.6, 0.6, 0.4, 0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0],
    "surprise": [0.6, 0.6, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0]
}
