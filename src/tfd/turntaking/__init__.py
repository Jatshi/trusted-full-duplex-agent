"""turn-taking 子包：把"何时开口"从基座黑盒里拎出来，做成可训练、可评测组件。

三态决策（hold / take / backchannel）：
  hold        用户还在说（语音中或句中停顿），继续听
  take        用户说完了，bot 该接话
  backchannel 用户在犹豫/思考（填充停顿），bot 发短反馈（"嗯"）鼓励继续
"""
from .synth import Utterance, synth_dataset, frame_labels
from .features import FrameFeaturizer, featurize_utterance
from .predictor import (
    HOLD, TAKE, BACKCHANNEL,
    RulePauseStrategy, BaseAggressiveStrategy, LearnedStrategy,
    TurnEvent, evaluate_strategy, TrainableMLP,
)

__all__ = [
    "Utterance", "synth_dataset", "frame_labels",
    "FrameFeaturizer", "featurize_utterance",
    "HOLD", "TAKE", "BACKCHANNEL",
    "RulePauseStrategy", "BaseAggressiveStrategy", "LearnedStrategy",
    "TurnEvent", "evaluate_strategy", "TrainableMLP",
]
