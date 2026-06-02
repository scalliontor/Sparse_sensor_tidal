"""
OVCNO Architecture Diagram — Professional neural-net style figure.
Uses matplotlib with FancyBboxPatch, FancyArrowPatch, and gradient fills
to create a publication-quality architecture schematic.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

# ─── Color palette ───
C_INPUT   = '#C8E6C9'  # green pastel
C_INPUT_E = '#388E3C'
C_TOKEN   = '#B2DFDB'  # teal pastel
C_TOKEN_E = '#00796B'
C_ENC     = '#BBDEFB'  # blue pastel
C_ENC_E   = '#1565C0'
C_LATENT  = '#D1C4E9'  # purple pastel
C_LATENT_E= '#5E35B1'
C_QUERY   = '#FFE0B2'  # amber pastel
C_QUERY_E = '#EF6C00'
C_DEC     = '#FFCCBC'  # deep orange pastel
C_DEC_E   = '#D84315'
C_OUT     = '#F8BBD0'  # pink pastel
C_OUT_E   = '#C62828'

FONT_TITLE = {'fontsize': 11, 'fontweight': 'bold', 'fontfamily': 'sans-serif', 'color': '#212121'}
FONT_SUB   = {'fontsize': 9,  'fontfamily': 'sans-serif', 'color': '#424242', 'style': 'italic'}
FONT_MATH  = {'fontsize': 9.5, 'fontfamily': 'serif', 'color': '#37474F'}

def draw_box(ax, x, y, w, h, fc, ec, title, subtitle=None, math=None, shadow=True, round_size=0.15):
    """Draw a rounded box with optional shadow, title, subtitle, and math annotation."""
    if shadow:
        shadow_patch = FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h,
            boxstyle=f"round,pad={round_size}",
            facecolor='#E0E0E0', edgecolor='none', alpha=0.5, zorder=1
        )
        ax.add_patch(shadow_patch)
    
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={round_size}",
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2
    )
    ax.add_patch(box)
    
    cx, cy = x + w/2, y + h/2
    if subtitle and math:
        ax.text(cx, cy + 0.22, title, ha='center', va='center', zorder=3, **FONT_TITLE)
        ax.text(cx, cy - 0.05, subtitle, ha='center', va='center', zorder=3, **FONT_SUB)
        ax.text(cx, cy - 0.32, math, ha='center', va='center', zorder=3, **FONT_MATH)
    elif subtitle:
        ax.text(cx, cy + 0.15, title, ha='center', va='center', zorder=3, **FONT_TITLE)
        ax.text(cx, cy - 0.12, subtitle, ha='center', va='center', zorder=3, **FONT_SUB)
    elif math:
        ax.text(cx, cy + 0.15, title, ha='center', va='center', zorder=3, **FONT_TITLE)
        ax.text(cx, cy - 0.12, math, ha='center', va='center', zorder=3, **FONT_MATH)
    else:
        ax.text(cx, cy, title, ha='center', va='center', zorder=3, **FONT_TITLE)
    
    return (cx, y, cx, y + h)  # bottom_x, bottom_y, top_x, top_y

def draw_arrow(ax, x1, y1, x2, y2, color='#555555', style='->', lw=1.8, dashed=False):
    """Draw a curved arrow between two points."""
    ls = '--' if dashed else '-'
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f'{style}',
        mutation_scale=15,
        lw=lw, color=color,
        linestyle=ls,
        connectionstyle='arc3,rad=0',
        zorder=5
    )
    ax.add_patch(arrow)

# ─── Layout coordinates ───
CX = 5.0       # center x
BW = 3.2       # box width
BH = 0.85      # box height

# Positions (x_center, y_bottom)
y_sensors = 12.5
y_tokens  = 11.0
y_setenc  = 9.5
y_lstm    = 8.0
y_branch  = 6.2   # latent & obs side by side
y_decoder = 4.3
y_output  = 2.5

# ─── Draw boxes ───

# 1. Boundary Sensors (input)
bx = CX - BW/2
draw_box(ax, bx, y_sensors, BW, BH, C_INPUT, C_INPUT_E,
         'Boundary Sensors', math=r'$\mathcal{S}(t\!-\!k : t)$')

# 2. Coordinate Tokens
draw_box(ax, bx, y_tokens, BW, BH, C_TOKEN, C_TOKEN_E,
         'Coordinate Tokens', math=r'$(x_i,\, y_i,\, s_i(\tau))$')

# 3. Set Encoder
draw_box(ax, bx, y_setenc, BW, BH, C_ENC, C_ENC_E,
         'Set Encoder', subtitle='Permutation-invariant aggregation')

# 4. Causal LSTM
draw_box(ax, bx, y_lstm, BW, BH, C_ENC, C_ENC_E,
         'Causal LSTM', math=r'History $\rightarrow h_T$')

# 5. Latent z (left branch)
lw_branch = 2.6
lx_latent = CX - lw_branch/2 - 0.8
draw_box(ax, lx_latent, y_branch, lw_branch, BH, C_LATENT, C_LATENT_E,
         'Latent State z', subtitle='Reparameterized sampling')

# 6. Observability (right branch)
lx_obs = CX + 0.8
draw_box(ax, lx_obs, y_branch, lw_branch, BH, C_LATENT, C_LATENT_E,
         'Observability', math=r'$o_\psi(x,y,t) \in [0,1]$')

# 7. Query Targets (side input)
qx, qw, qh = 8.0, 1.8, 0.7
draw_box(ax, qx, y_branch + 1.5, qw, qh, C_QUERY, C_QUERY_E,
         'Query', math=r'$(x, y, t)$', shadow=True, round_size=0.12)

# 8. Decoder
draw_box(ax, bx, y_decoder, BW, BH, C_DEC, C_DEC_E,
         'Conditioned Decoder', math=r'$f_\theta(\mathbf{z},\, \Phi,\, o_\psi)$')

# 9. Forecast Mean (left output)
ow = 2.4
ox_mean = CX - ow - 0.3
draw_box(ax, ox_mean, y_output, ow, 0.75, C_OUT, C_OUT_E,
         'Forecast Mean', math=r'$\mu_\theta(x,y,t)$')

# 10. Uncertainty Field (right output)
ox_var = CX + 0.3
draw_box(ax, ox_var, y_output, ow, 0.75, C_OUT, C_OUT_E,
         'Uncertainty Field', math=r'$\sigma^2_\eta(x,y,t)$')

# ─── Arrows ───

# Main vertical flow
draw_arrow(ax, CX, y_sensors, CX, y_tokens + BH)
draw_arrow(ax, CX, y_tokens, CX, y_setenc + BH)
draw_arrow(ax, CX, y_setenc, CX, y_lstm + BH)

# LSTM → branches
draw_arrow(ax, CX - 0.5, y_lstm, lx_latent + lw_branch/2, y_branch + BH)
draw_arrow(ax, CX + 0.5, y_lstm, lx_obs + lw_branch/2, y_branch + BH)

# Branches → Decoder
draw_arrow(ax, lx_latent + lw_branch/2, y_branch, CX - 0.5, y_decoder + BH)
draw_arrow(ax, lx_obs + lw_branch/2, y_branch, CX + 0.5, y_decoder + BH)

# Query → Observability (dashed)
draw_arrow(ax, qx, y_branch + 1.5 + qh/2, lx_obs + lw_branch, y_branch + BH,
           color=C_QUERY_E, dashed=True)

# Query → Decoder (dashed)
draw_arrow(ax, qx, y_branch + 1.5, CX + BW/2, y_decoder + BH,
           color=C_QUERY_E, dashed=True)

# Decoder → Outputs
draw_arrow(ax, CX - 0.5, y_decoder, ox_mean + ow/2, y_output + 0.75)
draw_arrow(ax, CX + 0.5, y_decoder, ox_var + ow/2, y_output + 0.75)

# ─── Stage labels on the left ───
stage_font = {'fontsize': 8, 'fontfamily': 'sans-serif', 'color': '#9E9E9E',
              'fontweight': 'bold', 'rotation': 90, 'va': 'center', 'ha': 'center'}

ax.text(0.6, (y_sensors + y_tokens + BH) / 2, 'INPUT', **stage_font)
ax.text(0.6, (y_setenc + y_lstm + BH) / 2, 'ENCODER', **stage_font)
ax.text(0.6, y_branch + BH/2, 'LATENT', **stage_font)
ax.text(0.6, y_decoder + BH/2, 'DECODER', **stage_font)
ax.text(0.6, y_output + 0.75/2, 'OUTPUT', **stage_font)

# Thin vertical line on the left for stage grouping
for yb, yt in [(y_sensors, y_tokens + BH), (y_setenc, y_lstm + BH)]:
    ax.plot([0.95, 0.95], [yb + 0.1, yt - 0.1], color='#BDBDBD', lw=1.2, solid_capstyle='round')

plt.tight_layout()
plt.savefig('figures/ovcno_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/ovcno_architecture.pdf', bbox_inches='tight', facecolor='white')
print("Saved figures/ovcno_architecture.png and .pdf")
