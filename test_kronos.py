import sys; sys.path.insert(0,'C:/kronos')
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-2k')
m = Kronos.from_pretrained('NeoQuasar/Kronos-mini')
pred = KronosPredictor(m, tok, max_context=400)
print('predictor ok')
idx = pd.date_range('2026-01-01', periods=400, freq='1min', tz='UTC')
df = pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.5,'volume':1000.}, index=idx)
y_ts = pd.date_range(idx[-1] + pd.Timedelta('1min'), periods=120, freq='1min', tz='UTC')
r = pred.predict(df=df, x_timestamp=idx, y_timestamp=y_ts, pred_len=120, T=0.8, top_p=0.9, sample_count=1)
print('prediction ok, close:', r['close'].iloc[-1])
