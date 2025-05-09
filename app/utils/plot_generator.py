import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional,Union

class PlotGenerator:
    def __init__(self):
        # Set the style for all plots
        plt.style.use('seaborn-v0_8')
        self.color_palette = sns.color_palette("husl", 8)

    def save_figure_to_image(self,
        fig: plt.Figure,
        filename: str,
        output_dir: Union[str, Path] = "output",
        format: str = "png",
        dpi: int = 100,
        bbox_inches: str = "tight",
        **savefig_kwargs
    ) -> Path:
        """
        Save matplotlib figure to an image file.
        
        Args:
            fig: Matplotlib figure object
            filename: Name of the output file (without extension)
            output_dir: Directory to save the image (default: "output")
            format: Image format ('png', 'jpg', 'svg', etc.) (default: "png")
            dpi: Dots per inch for raster formats (default: 100)
            bbox_inches: Bounding box option (default: "tight")
            **savefig_kwargs: Additional arguments to pass to fig.savefig()
            
        Returns:
            Path to the saved image file
            
        Raises:
            ValueError: If invalid format is specified
            OSError: If directory cannot be created or file cannot be written
        """
        print("debug: inside save_figure_to_image")
        # Validate format
        valid_formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf', 'tif', 'tiff']
        if format.lower() not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of {valid_formats}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Add extension if not present
        if not filename.lower().endswith(f".{format.lower()}"):
            filename = f"{filename}.{format.lower()}"
        
        # Full file path
        file_path = output_path / filename
        
        # Save the figure
        fig.savefig(
            file_path,
            format=format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **savefig_kwargs
        )
        
        return filename

    def generate_keystroke_timing_plot(self, dwell_times: List[float], flight_times: List[float]) -> str:
        """
        Generate a plot showing keystroke timing patterns
        
        Args:
            dwell_times: List of dwell times (key press durations)
            flight_times: List of flight times (between key releases)
            
        Returns:
            Base64 encoded PNG image of the plot
        """
        print("debug: inside generate_keystroke_timing_plot")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Dwell time plot
        sns.histplot(dwell_times, kde=True, ax=ax1, color=self.color_palette[0])
        ax1.set_title('Dwell Time Distribution (ms)')
        ax1.set_xlabel('Duration (ms)')
        ax1.set_ylabel('Frequency')
        
        # Flight time plot
        sns.histplot(flight_times, kde=True, ax=ax2, color=self.color_palette[1])
        ax2.set_title('Flight Time Distribution (ms)')
        ax2.set_xlabel('Duration (ms)')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        return self.save_figure_to_image(fig,"keystroke_timing_plot",output_dir='static/plot',format='png',dpi=300,facecolor='white',transparent=False)

    
    def generate_keystroke_heatmap(self, processed_data: List[dict]) -> Optional[str]:
        """
        Generate a heatmap of common key transitions from processed keystroke data.
        
        Args:
            processed_data: List of dicts with processed keystroke pairs:
                {'key1': str, 'key2': str, 'dwell_time': float, 'flight_time': float, ...}
        
        Returns:
            Base64 encoded PNG image of the heatmap, or None if insufficient data.
        """
        print("debug: inside generate_keystroke_heatmap (processed)")
        
        # Filter out valid key transitions
        valid_pairs = [
            f"{item['key1']}-{item['key2']}" 
            for item in processed_data 
            if item.get('key1') and item.get('key2')
        ]
        
        if not valid_pairs:
            return None

        # Count key transitions
        transition_counts = defaultdict(int)
        for pair in valid_pairs:
            transition_counts[pair] += 1

        return self._create_heatmap(transition_counts)
    
    def _create_heatmap(self, transition_counts: Dict[str, int]) -> Optional[str]:
        """
        Create heatmap visualization from transition counts
        """
        print("debug: inside _create_heatmap")
        if not transition_counts:
            return None
            
        # Normalize frequencies (convert counts to percentages)
        total_transitions = sum(transition_counts.values())
        key_transitions = {
            pair: (count / total_transitions * 1000)  # as percentage #i changed to 1000 for better visibility
            for pair, count in transition_counts.items()
        }
        
        # Extract unique keys from transitions
        keys = sorted({key for pair in key_transitions.keys() for key in pair.split('-')})
        
        # Create transition matrix
        matrix = np.zeros((len(keys), len(keys)))
        for pair, freq in key_transitions.items():
            from_key, to_key = pair.split('-')
            i = keys.index(from_key)
            j = keys.index(to_key)
            matrix[i, j] = freq
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=keys,
            yticklabels=keys,
            ax=ax,
            vmin=0,
            vmax=100
        )
        ax.set_title('Key Transition Frequency Heatmap (%)')
        ax.set_xlabel('To Key')
        ax.set_ylabel('From Key')
        plt.tight_layout()
        
        # Save to image
        return self.save_figure_to_image(fig,"keystroke_heatmap_plot",output_dir='static/plot',format='png',dpi=300,facecolor='white',transparent=False)

# Default instance for easy import
plot_generator = PlotGenerator()