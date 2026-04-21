import numpy as np

def generate_explanation(heatmap, label, confidence):
    """
    Generates a clinical explanation based on heatmap focal points.
    Uses a 5x5 grid resolution.
    """
    rows, cols = 5, 5
    h, w = heatmap.shape
    
    # Divide heatmap into 5x5 grid and calculate mean intensity in each sector
    grid = np.zeros((rows, cols))
    row_step = h // rows
    col_step = w // cols
    
    for r in range(rows):
        for c in range(cols):
            sector = heatmap[r*row_step:(r+1)*row_step, c*col_step:(c+1)*col_step]
            grid[r, c] = np.mean(sector)
            
    # Identify Principal Sector of Interest (PSI)
    max_idx = np.unravel_index(np.argmax(grid), grid.shape)
    psi_intensity = grid[max_idx]
    global_mean = np.mean(grid)
    
    # Analyze Activation Uniformity
    # Ratio of peak intensity to mean intensity
    uniformity_ratio = psi_intensity / (global_mean + 1e-7)
    distribution = "Localized" if uniformity_ratio > 1.5 else "Distributed"
    
    # Determine Clinical Significance
    # If the PSI is very low intensity, it might be background noise
    significance = "High" if psi_intensity > 0.4 else "Moderate"
    
    # Generate Clinical Report
    max_row, max_col = int(max_idx[0]), int(max_idx[1])
    report = (
        f"Clinical Observation: Primary feature activation identified in sector ({max_row},{max_col}). "
        f"Diagnostic Prediction: {label} (Confidence: {confidence:.1%}). "
        f"Activation Topology: {distribution}. "
        f"Feature Intensity: {significance}."
    )
    
    return report
