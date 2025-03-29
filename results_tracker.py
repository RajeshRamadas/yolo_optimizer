"""
Performance tracking and reporting for YOLO Neural Architecture Search.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class NASResultsTracker:
    """Tracker for NAS experiment results with visualization capabilities."""
    
    def __init__(self, output_dir="results"):
        """
        Initialize the results tracker.
        
        Args:
            output_dir: Directory to save results and visualizations
        """
        self.output_dir = output_dir
        self.results_csv = os.path.join(output_dir, "nas_results.csv")
        self.history = []
        self.best_result = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize empty dataframe
        if not os.path.exists(self.results_csv):
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'iteration', 'resolution', 'depth_mult', 'width_mult', 
                'kernel_size', 'num_channels', 'lr0', 'momentum', 'batch_size', 
                'iou_thresh', 'weight_decay', 'map50', 'training_time'
            ])
            empty_df.to_csv(self.results_csv, index=False)
            
    def add_result(self, iteration, params, map50, training_time):
        """
        Add a new experiment result.
        
        Args:
            iteration: Iteration number
            params: Dictionary of architecture parameters
            map50: mAP50 performance metric
            training_time: Training time in seconds
        """
        # Create result entry
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iteration': iteration,
            'map50': map50,
            'training_time': training_time
        }
        
        # Add all parameters
        result.update(params)
        
        # Update best result if applicable
        if self.best_result is None or map50 > self.best_result['map50']:
            self.best_result = result
            
        # Add to history
        self.history.append(result)
        
        # Update CSV
        try:
            df = pd.read_csv(self.results_csv)
            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
            df.to_csv(self.results_csv, index=False)
        except Exception as e:
            logging.error(f"Error updating results CSV: {e}")
            
    def load_from_optimizer_log(self, log_file):
        """
        Load results from a Bayesian Optimization JSON log file.
        
        Args:
            log_file: Path to the JSON log file
        """
        if not os.path.exists(log_file):
            logging.error(f"Log file not found: {log_file}")
            return
            
        try:
            with open(log_file, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            params = data.get('params', {})
                            target = data.get('target', 0.0)
                            
                            # Add to history - training time unknown from logs
                            self.add_result(i, params, target, 0.0)
                        except json.JSONDecodeError:
                            logging.warning(f"Could not parse line {i} in log file")
        except Exception as e:
            logging.error(f"Error loading optimizer log file: {e}")
            
    def get_results_table(self, top_n=10, format="markdown"):
        """
        Generate a formatted table of the top N results.
        
        Args:
            top_n: Number of top results to include
            format: Table format ('markdown', 'grid', 'html', etc.)
            
        Returns:
            Formatted table as a string
        """
        if not self.history:
            return "No results available"
            
        # Convert history to DataFrame and sort by mAP50
        df = pd.DataFrame(self.history)
        df = df.sort_values('map50', ascending=False).head(top_n)
        
        # Format numbers for display
        display_df = df.copy()
        for col in df.columns:
            if col in ['map50', 'depth_mult', 'width_mult', 'lr0', 'momentum', 'iou_thresh', 'weight_decay']:
                display_df[col] = display_df[col].map(lambda x: f"{float(x):.4f}")
            elif col in ['resolution', 'kernel_size', 'num_channels', 'batch_size']:
                display_df[col] = display_df[col].map(lambda x: f"{int(float(x))}")
            elif col == 'training_time':
                display_df[col] = display_df[col].map(lambda x: f"{float(x)/60:.2f} min" if x else "N/A")
        
        # Select key columns for display
        display_cols = ['iteration', 'map50', 'resolution', 'depth_mult', 'width_mult', 
                        'kernel_size', 'num_channels', 'lr0', 'batch_size', 'training_time']
        
        return tabulate(display_df[display_cols], headers='keys', tablefmt=format, showindex=False)
    
    def plot_optimization_progress(self, save=True):
        """
        Plot the optimization progress over iterations.
        
        Args:
            save: Whether to save the plot to a file
            
        Returns:
            Path to the saved plot or None
        """
        if not self.history:
            return None
            
        # Create a DataFrame from history
        df = pd.DataFrame(self.history)
        df = df.sort_values('iteration')
        
        # Set up the figure
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        # Plot mAP50 scores over iterations
        ax.plot(df['iteration'], df['map50'], 'o-', color='#1f77b4', label='mAP50')
        
        # Plot a moving average if we have enough points
        if len(df) > 3:
            df['smoothed_map50'] = df['map50'].rolling(window=3, min_periods=1).mean()
            ax.plot(df['iteration'], df['smoothed_map50'], '--', color='#ff7f0e', label='3-point Moving Avg')
        
        # Add best score line
        best_score = df['map50'].max()
        ax.axhline(y=best_score, color='r', linestyle='--', alpha=0.5, label=f'Best Score: {best_score:.4f}')
        
        # Add annotations for best model
        best_iteration = df.loc[df['map50'].idxmax(), 'iteration']
        ax.annotate(f'Best: {best_score:.4f}',
                   xy=(best_iteration, best_score),
                   xytext=(best_iteration, best_score * 0.95),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10,
                   ha='center')
        
        # Add grid, legend, and labels
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('mAP50 Score', fontsize=12)
        ax.set_title('Neural Architecture Search Progress', fontsize=14)
        ax.legend(loc='best')
        
        # Set y-axis to start at 0 or slightly below minimum score
        ax.set_ylim(bottom=max(0, df['map50'].min() * 0.9))
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show the plot
        if save:
            output_path = os.path.join(self.output_dir, 'nas_progress.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def plot_parameter_effects(self, save=True):
        """
        Plot the effects of different parameters on model performance.
        
        Args:
            save: Whether to save the plot to a file
            
        Returns:
            Path to the saved plot or None
        """
        if len(self.history) < 5:  # Need some data points for meaningful analysis
            return None
            
        # Create DataFrame
        df = pd.DataFrame(self.history)
        
        # Parameters to analyze
        params = ['resolution', 'depth_mult', 'width_mult', 'kernel_size', 
                 'num_channels', 'lr0', 'batch_size']
        
        # Set up the figure
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
        axes = axes.flatten()
        
        # For each parameter, plot its relationship with mAP50
        for i, param in enumerate(params):
            if i < len(axes):
                # Convert to float to ensure proper plotting
                df[param] = df[param].astype(float)
                
                # Create the scatter plot
                axes[i].scatter(df[param], df['map50'], alpha=0.7)
                
                # Try to fit a polynomial trend line
                try:
                    z = np.polyfit(df[param], df['map50'], 2)
                    p = np.poly1d(z)
                    x_range = np.linspace(df[param].min(), df[param].max(), 100)
                    axes[i].plot(x_range, p(x_range), 'r--', alpha=0.7)
                except:
                    # Skip trend line if fitting fails
                    pass
                
                axes[i].set_title(f'Effect of {param}')
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('mAP50')
                axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Remove any unused subplots
        for i in range(len(params), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save or show the plot
        if save:
            output_path = os.path.join(self.output_dir, 'parameter_effects.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def generate_report(self):
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with report information and paths to generated files
        """
        report = {
            'total_trials': len(self.history),
            'best_model': self.best_result,
            'results_csv': self.results_csv,
            'top_models_table': self.get_results_table(top_n=10),
            'progress_plot': self.plot_optimization_progress(),
            'parameter_effects_plot': self.plot_parameter_effects() if len(self.history) >= 5 else None
        }
        
        # Save report as JSON
        report_path = os.path.join(self.output_dir, 'nas_report.json')
        with open(report_path, 'w') as f:
            # Clean up non-serializable items before saving
            report_json = {k: v for k, v in report.items() if k not in ['top_models_table']}
            if report_json.get('best_model'):
                report_json['best_model'] = {str(k): float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v 
                                            for k, v in report_json['best_model'].items()}
            json.dump(report_json, f, indent=4)
        
        report['report_json'] = report_path
        
        # Create a text report
        text_report = f"""
# YOLO Neural Architecture Search Report

## Summary
- Total Trials: {report['total_trials']}
- Best mAP50: {self.best_result['map50'] if self.best_result else 'N/A'}
- Results CSV: {self.results_csv}

## Top Models
{self.get_results_table(top_n=10)}

## Best Model Parameters
{json.dumps(self.best_result, indent=2) if self.best_result else 'N/A'}
        """
        
        report_txt_path = os.path.join(self.output_dir, 'nas_report.md')
        with open(report_txt_path, 'w') as f:
            f.write(text_report)
        
        report['report_text'] = report_txt_path
        
        return report


# Function to integrate the tracker with existing optimization code
def track_nas_result(tracker, params, map50, training_time, iteration):
    """
    Track a single NAS result.
    
    Args:
        tracker: NASResultsTracker instance
        params: Model parameters
        map50: Performance metric
        training_time: Training time in seconds
        iteration: Iteration number
    """
    # Convert numpy types to float/int
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (np.float32, np.float64, np.int64, np.int32)):
            clean_params[k] = float(v)
        else:
            clean_params[k] = v
            
    tracker.add_result(iteration, clean_params, map50, training_time)
    return tracker