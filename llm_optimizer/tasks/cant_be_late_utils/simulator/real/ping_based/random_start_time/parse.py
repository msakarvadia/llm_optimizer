# %%
import datetime
import os
import json
import pandas as pd
import pathlib
TRACE = '../origin_traces/2022-10-26T22-05'

trace_path = pathlib.Path(TRACE).expanduser().absolute()
traces = {}
if trace_path.is_file():
    traces = {trace_path.stem: pd.read_csv(trace_path)}
else:
    for path in trace_path.glob('*.txt'):
        trace = pd.read_csv(path, names=['index', 'time', 'preempted'])
        traces[path.stem] = trace

# %%
starting_time_last = pd.to_datetime(trace['time']).iloc[-1] - datetime.timedelta(seconds=5 * 24 * 3600)
for trace_name, trace in traces.items():
    time_range = trace['time'][pd.to_datetime(trace['time']) < starting_time_last]
    for i in range(1000):
        start_time = time_range.sample(1).iloc[0]
        data = {
            'metadata': {'gap_seconds': 600, 'start_time': start_time},
            'data': trace['preempted'][pd.to_datetime(trace['time']) >= start_time].tolist(),
        }
        os.makedirs(f'{trace_name}', exist_ok=True)
        with pathlib.Path(__file__).parent.joinpath(f'{trace_name}/{i}.json').open('w') as f:
            json.dump(data, f)

# %%
