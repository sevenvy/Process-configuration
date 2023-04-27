import pandas as pd
import os
a = [1,2,3,4]
pd.DataFrame(a).to_csv(os.path.join('../result', 'time', 'AMOSA.csv'), index=False, header=False)