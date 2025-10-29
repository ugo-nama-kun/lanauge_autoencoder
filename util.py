
import math
from torch import nn


def init_model(model: nn.Module):
    """
    - Conv2d        : Kaiming Normal (fan_out, relu/GELU想定), bias=0
    - Linear        : Xavier Uniform (gain≈sqrt(2) for GELU/ReLU), bias=0
    - BatchNorm2d   : weight=1, bias=0
    - Residual blockの最後のBN(bn2): weight=0 で恒等開始
    - 出力head(分類/メッセージ/再構成の最終Linear): 小さく初期化（±1e-3）, bias=0
    """
    # 1) 汎用レイヤの初期化
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # GELUでもOK
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))  # GELU/RELU向け
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # 2) 残差ブロックの最終BNをゼロ初期化（ResNet v2の定番）
    #    あなたの ResBlock 実装が m.bn2 を持つ想定
    for m in model.modules():
        if hasattr(m, "bn2") and isinstance(m.bn2, nn.BatchNorm2d):
            nn.init.zeros_(m.bn2.weight)

    # 3) 末端ヘッドの最終Linearを小さく（学習安定化のためロジットの初期スケールを抑える）
    def _small_init_last_linear(layer: nn.Linear):
        nn.init.uniform_(layer.weight, -1e-3, 1e-3)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    # msg_encoder / msg_decoder / policy の最後の Linear を小さく
    if isinstance(getattr(model, "mlp", None), nn.Sequential):
        last = model.mlp[-1]
        if isinstance(last, nn.Linear):
            _small_init_last_linear(last)

    return model