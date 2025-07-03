import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines'))

fig.show()