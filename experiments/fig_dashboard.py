"""Static snapshot of the interactive dashboard (Sec. VIII): figures/fig_dashboard.pdf."""
from piar.style import set_style, savefig
from piar.dashboard import dashboard_figure

set_style()
fig = dashboard_figure(mu=-0.8, gamma=0.25, A=0.8, Omega=2 / 3, th0=0.1, om0=0.0, n_periods=4000)
savefig(fig, "fig_dashboard")
print("done")
