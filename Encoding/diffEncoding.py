import matplotlib.pyplot as plt
import numpy as np

rate_spikes = np.load('m (2)-rate.npy')
temporal_spikes = np.load('m (2)-temp.npy')

# Select pixel
pixel_i, pixel_j = 64, 64
rate_pattern = rate_spikes[pixel_i, pixel_j, :]
temporal_pattern = temporal_spikes[pixel_i, pixel_j, :]

# Spike times
rate_spike_times = np.where(rate_pattern == 1)[0]
temporal_spike_times = np.where(temporal_pattern == 1)[0]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
plt.style.use('default')

# 1. Combined Raster Plot
ax1 = axes[0]
ax1.scatter(rate_spike_times, np.full_like(rate_spike_times, 1), color='red', marker='|', s=200, label='Rate Encoding')
ax1.scatter(temporal_spike_times, np.full_like(temporal_spike_times, 0), color='blue', marker='|', s=200, label='Temporal Encoding')
ax1.set_title('Raster Plot - Spike Timing')
ax1.set_xlabel('Time Steps')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Temporal', 'Rate'])
ax1.set_ylim(-0.5, 1.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Cumulative Spike Distribution
ax2 = axes[1]
rate_cumsum = np.cumsum(rate_pattern)
temporal_cumsum = np.cumsum(temporal_pattern)
ax2.plot(rate_cumsum, 'r-', label='Rate Encoding', linewidth=2)
ax2.plot(temporal_cumsum, 'b-', label='Temporal Encoding', linewidth=2)
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Cumulative Spikes')
ax2.set_title('Cumulative Spike Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Spike Time Histogram
ax3 = axes[2]
bins = np.arange(0, 101, 5)
ax3.hist(rate_spike_times, bins=bins, alpha=0.7, label='Rate Encoding', color='red', density=True)
ax3.hist(temporal_spike_times, bins=bins, alpha=0.7, label='Temporal Encoding', color='blue', density=True)
ax3.set_xlabel('Spike Time')
ax3.set_ylabel('Density')
ax3.set_title('Spike Time Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Finalize layout
plt.tight_layout()
plt.show()
