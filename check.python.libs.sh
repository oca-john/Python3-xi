# 快速检查常用三方库

#!/usr/bin/sh
python -c "import numpy;print('numpy:', numpy.__version__, '\n');
import scipy;print('scipy:', scipy.__version__, '\n');
import pandas;print('pandas:', pandas.__version__, '\n');
import matplotlib;print('matplotlib:', matplotlib.__version__, '\n');
import seaborn;print('seaborn:', seaborn.__version__, '\n');
import sklearn;print('sklearn:', sklearn.__version__, '\n');
import mne;print('mne:', mne.__version__, '\n');
import neo;print('neo:', neo.__version__, '\n');
import torch;print('torch:', torch.__version__, '\n');
import tensorflow;print('tensorflow:', tensorflow.__version__, '\n');
import mindspore;print('mindspore:', mindspore.__version__, '\n')"
