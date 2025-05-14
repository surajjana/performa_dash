import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import timedelta
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Performa Data Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

class SoccerGPSAnalyzer:
    def __init__(self, data_path=None, data_df=None):
        """
        Initialize the analyzer with raw GPS data (timestamp, lat, long only)
        
        Args:
            data_path: Path to the CSV file containing GPS data
            data_df: Pandas DataFrame containing GPS data
        """
        # Read the data - assuming it only has timestamp, lat, lon
        if data_df is not None:
            self.data = data_df
        elif data_path:
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or data_df must be provided")
            
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Set some constants
        self.sprint_threshold = 5.5  # m/s (approximately 20 km/h)
        self.high_speed_threshold = 4.2  # m/s (approximately 15 km/h)
        self.speed_zones = [
            (0, 2, 'Walking'),      # 0-2 m/s (0-7.2 km/h)
            (2, 4, 'Jogging'),      # 2-4 m/s (7.2-14.4 km/h)
            (4, 5.5, 'Running'),    # 4-5.5 m/s (14.4-19.8 km/h)
            (5.5, 7, 'Sprinting'),  # 5.5-7 m/s (19.8-25.2 km/h)
            (7, 10, 'Max Sprint')   # 7+ m/s (25.2+ km/h)
        ]
        
        # Preprocess the data - derive speed and acceleration from lat/lon and timestamps
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        Preprocess the data to calculate:
        - time differences between samples
        - distances between consecutive points
        - speeds
        - accelerations
        All calculated from raw timestamp, lat, lon data
        """
        # Sort by player_id and timestamp to ensure proper sequence
        self.data = self.data.sort_values(['player_id', 'timestamp'])
        
        # Calculate time difference between samples
        self.data['time_diff'] = self.data.groupby('player_id')['timestamp'].diff().dt.total_seconds()
        
        # First sample for each player has NaN, fill with the median time difference
        median_time_diff = self.data['time_diff'].median()
        self.data['time_diff'] = self.data['time_diff'].fillna(median_time_diff)
        
        # Calculate distance traveled (in meters) using Haversine formula
        earth_radius = 6371000  # Earth radius in meters
        
        # Store previous lat/lon values to calculate distances between consecutive points
        self.data['lat_prev'] = self.data.groupby('player_id')['lat'].shift(1)
        self.data['lon_prev'] = self.data.groupby('player_id')['lon'].shift(1)
        
        # First point for each player will have NaN for prev values, fill with current values (zero distance)
        self.data['lat_prev'] = self.data['lat_prev'].fillna(self.data['lat'])
        self.data['lon_prev'] = self.data['lon_prev'].fillna(self.data['lon'])
        
        # Convert to radians for Haversine formula
        lat_rad = np.radians(self.data['lat'])
        lon_rad = np.radians(self.data['lon'])
        lat_prev_rad = np.radians(self.data['lat_prev'])
        lon_prev_rad = np.radians(self.data['lon_prev'])
        
        # Haversine formula components
        dlat = lat_rad - lat_prev_rad
        dlon = lon_rad - lon_prev_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat_prev_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = earth_radius * c
        
        self.data['distance'] = distance
        
        # Calculate instantaneous speed (m/s) from distance and time difference
        # v = d/t
        self.data['speed'] = self.data['distance'] / self.data['time_diff']
        
        # Handle potential division by zero or very small time differences
        self.data['speed'] = self.data['speed'].replace([np.inf, -np.inf], np.nan)
        
        # Apply median filter to smooth out noise in speed calculations
        # This is important when working with real GPS data which can have small errors
        self.data['speed'] = self.data.groupby('player_id')['speed'].transform(
            lambda x: x.rolling(window=5, center=True, min_periods=1).median()
        )
        
        # Calculate acceleration (m/s²) as the rate of change of speed
        self.data['acceleration'] = self.data.groupby('player_id')['speed'].diff() / self.data['time_diff']
        
        # Apply the same smoothing to acceleration
        self.data['acceleration'] = self.data.groupby('player_id')['acceleration'].transform(
            lambda x: x.rolling(window=5, center=True, min_periods=1).median()
        )
        
        # Replace remaining NaN values with 0
        self.data['acceleration'] = self.data['acceleration'].fillna(0)
        
        # Handle any remaining NaN values in speed
        self.data['speed'] = self.data['speed'].fillna(0)
    
    def analyze_player(self, player_id):
        """
        Analyze a single player's GPS data and return a dictionary of metrics
        
        Args:
            player_id: ID of the player to analyze
            
        Returns:
            Dictionary of computed metrics
        """
        player_data = self.data[self.data['player_id'] == player_id].copy()
        
        if player_data.empty:
            st.warning(f"No data found for player {player_id}")
            return {}
        
        # Get team ID
        team_id = player_data['team_id'].iloc[0]
        
        # Total game time
        start_time = player_data['timestamp'].min()
        end_time = player_data['timestamp'].max()
        game_time_seconds = (end_time - start_time).total_seconds()
        game_time_minutes = game_time_seconds / 60
        
        # Total distance
        total_distance = player_data['distance'].sum()
        
        # Average speed
        avg_speed = player_data['speed'].mean()
        
        # Top speed
        top_speed = player_data['speed'].max()
        
        # Distance per minute
        distance_per_minute = total_distance / game_time_minutes
        
        # Sprints
        sprints = player_data['speed'] >= self.sprint_threshold
        sprint_starts = ((sprints) & (~sprints.shift(1).fillna(False)))
        total_sprints = sprint_starts.sum()
        
        # Sprint distance
        sprint_distance = player_data.loc[sprints, 'distance'].sum()
        
        # Calculate time spent in each speed zone
        speed_zone_times = []
        speed_zone_distances = []
        for low, high, name in self.speed_zones:
            in_zone = (player_data['speed'] >= low) & (player_data['speed'] < high)
            time_in_zone = player_data.loc[in_zone, 'time_diff'].sum()
            distance_in_zone = player_data.loc[in_zone, 'distance'].sum()
            speed_zone_times.append({
                'zone': name,
                'time_seconds': time_in_zone,
                'percentage': time_in_zone / game_time_seconds * 100
            })
            speed_zone_distances.append({
                'zone': name,
                'distance': distance_in_zone,
                'percentage': distance_in_zone / total_distance * 100
            })
        
        # Calculate work rate (distance/time)
        work_rate = total_distance / game_time_seconds
        
        # Acceleration/deceleration metrics
        positive_acc = player_data[player_data['acceleration'] > 0]['acceleration']
        negative_acc = player_data[player_data['acceleration'] < 0]['acceleration']
        
        avg_acceleration = positive_acc.mean() if not positive_acc.empty else 0
        avg_deceleration = abs(negative_acc.mean()) if not negative_acc.empty else 0
        max_acceleration = positive_acc.max() if not positive_acc.empty else 0
        max_deceleration = abs(negative_acc.min()) if not negative_acc.empty else 0
        
        # Return all metrics as a dictionary
        return {
            'player_id': player_id,
            'team_id': team_id,
            'game_time_minutes': game_time_minutes,
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'top_speed': top_speed,
            'distance_per_minute': distance_per_minute,
            'total_sprints': total_sprints,
            'sprint_distance': sprint_distance,
            'work_rate': work_rate,
            'avg_acceleration': avg_acceleration,
            'avg_deceleration': avg_deceleration,
            'max_acceleration': max_acceleration,
            'max_deceleration': max_deceleration,
            'speed_zone_times': speed_zone_times,
            'speed_zone_distances': speed_zone_distances,
            'position_data': player_data[['lat', 'lon']].values
        }
    
    def analyze_all_players(self):
        """Analyze all players and return list of player metrics"""
        player_ids = self.data['player_id'].unique()
        
        all_player_metrics = []
        for player_id in player_ids:
            metrics = self.analyze_player(player_id)
            if metrics:
                all_player_metrics.append(metrics)
        
        return all_player_metrics
    
    def generate_player_summary_plots(self, player_id):
        """
        Generate a comprehensive summary with visualizations for a player
        
        Args:
            player_id: ID of the player to analyze
            
        Returns:
            Dictionary containing the plots as figures
        """
        player_metrics = self.analyze_player(player_id)
        if not player_metrics:
            return {}
        
        # Create a comprehensive report with multiple plots
        fig_summary = plt.figure(figsize=(20, 15))
        fig_summary.suptitle(f"Player {player_id} (Team {player_metrics['team_id']}) Performance Summary", 
                     fontsize=18, fontweight='bold')
        
        # 1. Heat map of player positions
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self._plot_heatmap(ax1, player_metrics['position_data'])
        
        # 2. Speed zone distribution (pie chart)
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        self._plot_speed_zones_pie(ax2, player_metrics['speed_zone_times'])
        
        # 3. Distance covered in each speed zone (bar chart)
        ax3 = plt.subplot2grid((3, 3), (1, 2))
        self._plot_speed_zone_distances(ax3, player_metrics['speed_zone_distances'])
        
        # 4. Key metrics table
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self._plot_metrics_table(ax4, player_metrics)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Create a separate speed profile plot
        fig_speed = self._get_speed_profile_fig(player_id)
        
        return {
            'summary': fig_summary,
            'speed_profile': fig_speed
        }
    
    def _plot_heatmap(self, ax, position_data):
        """Plot a heatmap of player positions"""
        # Extract lat/lon data
        lat_data = position_data[:, 0]
        lon_data = position_data[:, 1]
        
        # Create a custom colormap
        colors = [(0, 0, 0.5), (0, 0.5, 1), (0.1, 0.8, 0.8), (0.5, 1, 0.5), 
                 (1, 1, 0), (1, 0.6, 0), (1, 0, 0)]
        cmap_name = 'custom_heat'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        
        # Create hexbin plot
        hb = ax.hexbin(lon_data, lat_data, gridsize=30, cmap=cm, mincnt=1)
        
        # Get field boundaries
        min_lat = self.data['lat'].min()
        max_lat = self.data['lat'].max()
        min_lon = self.data['lon'].min()
        max_lon = self.data['lon'].max()
        
        # Draw field outline
        field_outline = plt.Rectangle((min_lon, min_lat), 
                                     max_lon - min_lon, 
                                     max_lat - min_lat,
                                     fill=False, color='black', linewidth=2)
        ax.add_patch(field_outline)
        
        # Draw centerline
        center_lat = (min_lat + max_lat) / 2
        ax.axhline(y=center_lat, color='black', linestyle='-', linewidth=1)
        
        # Set consistent axis limits slightly beyond the field
        padding = 0.0001  # Add a small padding around the field
        ax.set_xlim(min_lon - padding, max_lon + padding)
        ax.set_ylim(min_lat - padding, max_lat + padding)
        
        # Add a colorbar
        plt.colorbar(hb, ax=ax, label='Position density')
        
        ax.set_title('Player Movement Heatmap')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    def _plot_speed_zones_pie(self, ax, speed_zone_times):
        """Plot a pie chart of time spent in each speed zone"""
        zones = [zone['zone'] for zone in speed_zone_times]
        percentages = [zone['percentage'] for zone in speed_zone_times]
        
        # Define colors for each speed zone
        colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59']
        
        ax.pie(percentages, labels=zones, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax.set_title('Time in Speed Zones')
    
    def _plot_speed_zone_distances(self, ax, speed_zone_distances):
        """Plot a bar chart of distance covered in each speed zone"""
        zones = [zone['zone'] for zone in speed_zone_distances]
        distances = [zone['distance'] for zone in speed_zone_distances]
        
        # Define colors for each speed zone
        colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59']
        
        bars = ax.bar(zones, distances, color=colors)
        ax.set_title('Distance by Speed Zone (m)')
        ax.set_ylabel('Distance (m)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
    
    def _plot_metrics_table(self, ax, metrics):
        """Plot a table of key performance metrics"""
        # Hide axis
        ax.axis('off')
        
        # Prepare data for the table
        data = [
            ['Total Distance', f"{metrics['total_distance']:.1f} m"],
            ['Game Time', f"{metrics['game_time_minutes']:.1f} min"],
            ['Avg Speed', f"{metrics['avg_speed']:.2f} m/s"],
            ['Top Speed', f"{metrics['top_speed']:.2f} m/s"],
            ['Distance/Min', f"{metrics['distance_per_minute']:.1f} m/min"],
            ['Total Sprints', f"{metrics['total_sprints']}"],
            ['Sprint Distance', f"{metrics['sprint_distance']:.1f} m"],
            ['Work Rate', f"{metrics['work_rate']:.2f} m/s"],
            ['Avg Acceleration', f"{metrics['avg_acceleration']:.2f} m/s²"],
            ['Avg Deceleration', f"{metrics['avg_deceleration']:.2f} m/s²"],
            ['Max Acceleration', f"{metrics['max_acceleration']:.2f} m/s²"],
            ['Max Deceleration', f"{metrics['max_deceleration']:.2f} m/s²"],
        ]
        
        # Create table
        table = ax.table(
            cellText=data,
            cellLoc='left',
            loc='center',
            colWidths=[0.3, 0.3]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Add color to header row and alternating rows
        for i, key in enumerate(data):
            cell = table[i, 0]
            cell.set_text_props(fontweight='bold')
            
            # Add alternating row colors
            for j in range(2):
                cell = table[i, j]
                if i % 2 == 0:
                    cell.set_facecolor('#f2f2f2')
    
    def _get_speed_profile_fig(self, player_id):
        """Generate a speed profile graph over time"""
        player_data = self.data[self.data['player_id'] == player_id].copy()
        
        # Calculate elapsed time in minutes from the start
        start_time = player_data['timestamp'].min()
        player_data['elapsed_minutes'] = (player_data['timestamp'] - start_time).dt.total_seconds() / 60
        
        fig = plt.figure(figsize=(15, 7))
        
        # Plot speed over time
        plt.plot(player_data['elapsed_minutes'], player_data['speed'], linewidth=1)
        
        # Add sprint threshold line
        plt.axhline(y=self.sprint_threshold, color='r', linestyle='--', 
                   label=f'Sprint Threshold ({self.sprint_threshold} m/s)')
        
        # Add high speed threshold line
        plt.axhline(y=self.high_speed_threshold, color='orange', linestyle='--', 
                   label=f'High Speed Threshold ({self.high_speed_threshold} m/s)')
        
        plt.title(f'Player {player_id} Speed Profile')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Speed (m/s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return fig
    
    def get_team_comparison_figs(self):
        """Generate comparison plots of all player metrics by team"""
        player_metrics = self.analyze_all_players()
        
        # Prepare data for plotting
        team1_players = [m for m in player_metrics if m['team_id'] == 1]
        team2_players = [m for m in player_metrics if m['team_id'] == 2]
        
        figs = {}
        
        # 1. Compare total distance
        figs['distance'] = self._get_team_comparison_fig(
            team1_players, team2_players, 
            'total_distance', 'Total Distance (m)'
        )
        
        # 2. Compare average speed
        figs['speed'] = self._get_team_comparison_fig(
            team1_players, team2_players, 
            'avg_speed', 'Average Speed (m/s)'
        )
        
        # 3. Compare sprints
        figs['sprints'] = self._get_team_comparison_fig(
            team1_players, team2_players, 
            'total_sprints', 'Total Sprints'
        )
        
        # 4. Compare work rate
        figs['workrate'] = self._get_team_comparison_fig(
            team1_players, team2_players, 
            'work_rate', 'Work Rate (m/s)'
        )
        
        return figs
    
    def _get_team_comparison_fig(self, team1_players, team2_players, metric, title):
        """Create a comparison plot of a metric between teams"""
        fig = plt.figure(figsize=(12, 8))
        
        # Extract player IDs and metric values
        team1_ids = [p['player_id'] for p in team1_players]
        team1_values = [p[metric] for p in team1_players]
        
        team2_ids = [p['player_id'] for p in team2_players]
        team2_values = [p[metric] for p in team2_players]
        
        # Create grouped bar chart
        bar_width = 0.35
        x = np.arange(max(len(team1_ids), len(team2_ids)))
        
        plt.bar(x - bar_width/2, team1_values, bar_width, label='Team 1')
        plt.bar(x + bar_width/2, team2_values, bar_width, label='Team 2')
        
        # Add labels and title
        plt.xlabel('Player ID')
        plt.ylabel(title)
        plt.title(f'Team Comparison: {title}')
        plt.xticks(x, list(range(1, max(len(team1_ids), len(team2_ids)) + 1)))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(team1_values):
            plt.text(i - bar_width/2, v + 0.05, f'{v:.1f}', ha='center')
        
        for i, v in enumerate(team2_values):
            plt.text(i + bar_width/2, v + 0.05, f'{v:.1f}', ha='center')
        
        plt.tight_layout()
        return fig
    
    def get_player_summary_table(self):
        """Generate a DataFrame summary table of all player metrics"""
        player_metrics = self.analyze_all_players()
        
        # Prepare data for the table
        summary_data = []
        
        for player in player_metrics:
            row = {
                'Player ID': player['player_id'],
                'Team': player['team_id'],
                'Game Time (min)': round(player['game_time_minutes'], 1),
                'Total Distance (m)': round(player['total_distance'], 1),
                'Avg Speed (m/s)': round(player['avg_speed'], 2),
                'Top Speed (m/s)': round(player['top_speed'], 2),
                'Distance/Min (m/min)': round(player['distance_per_minute'], 1),
                'Total Sprints': player['total_sprints'],
                'Sprint Distance (m)': round(player['sprint_distance'], 1),
                'Work Rate (m/s)': round(player['work_rate'], 2),
                'Avg Acceleration (m/s²)': round(player['avg_acceleration'], 2),
                'Avg Deceleration (m/s²)': round(player['avg_deceleration'], 2),
            }
            
            # Add speed zone percentages
            for zone in player['speed_zone_times']:
                row[f"% Time in {zone['zone']}"] = round(zone['percentage'], 1)
            
            summary_data.append(row)
        
        # Create DataFrame
        return pd.DataFrame(summary_data)

def get_download_link(df, filename, text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Add a function to convert speed between units
def convert_speed(speed_value, from_unit='m/s', to_unit='km/h'):
    """Convert speed between different units"""
    if from_unit == 'm/s' and to_unit == 'km/h':
        return speed_value * 3.6
    elif from_unit == 'km/h' and to_unit == 'm/s':
        return speed_value / 3.6
    else:
        return speed_value  # No conversion needed or unsupported units

def main():
    st.title("⚽ Performa Data Analyzer")
    
    # Add app description
    st.markdown("""
    This application analyzes soccer player GPS data to provide performance insights.
    Upload your tracking data or use our sample data to get started.
    """)
    
    # Add a sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Page", ["Upload Data", "Player Analysis", "Team Comparison"])
    
    # Add sidebar info about thresholds
    st.sidebar.title("Analysis Parameters")
    st.sidebar.info("""
    **Default Thresholds:**
    - Sprint: 5.5 m/s (19.8 km/h)
    - High Speed: 4.2 m/s (15.1 km/h)
    """)
    
    # Add a sidebar footer
    st.sidebar.title("About")
    st.sidebar.info("""
    This app uses GPS tracking data to analyze soccer player performance.
    
    Data should include:
    - timestamp
    - player_id
    - team_id
    - lat (latitude)
    - lon (longitude)
    """)
    
    # Initialize session state to store data across pages
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    if 'player_ids' not in st.session_state:
        st.session_state.player_ids = []
    
    if app_mode == "Upload Data":
        st.header("Upload Performa Data")
        
        # Option to use sample data
        use_sample = st.checkbox("Use sample data", value=False)
        
        if use_sample:
            st.info("Using sample data. The sample contains GPS data for 10 players (5 from each team) in a soccer match.")
            # You would need to have a sample CSV file
            # In a real app, you might want to include sample data
            
            # Create a very small sample dataset for demonstration
            st.warning("Note: This is a simplified sample. In a real application, you would provide real sample data.")
            
            # Generate a simple sample dataset
            sample_data = []
            teams = [1, 2]
            for team_id in teams:
                for player_id in range(1, 6):  # 5 players per team
                    player_id_global = player_id if team_id == 1 else player_id + 5
                    
                    # Generate some timestamps
                    base_time = pd.Timestamp('2023-01-01 10:00:00')
                    
                    for i in range(100):  # 100 data points per player
                        time_offset = pd.Timedelta(seconds=i*10)  # Every 10 seconds
                        
                        # Create some movement patterns
                        # Center of field with some random movement
                        lat_base = 51.5 + (team_id - 1.5) * 0.001  # Team 1 on one side, Team 2 on the other
                        lon_base = -0.1
                        
                        # Add some random movement
                        lat = lat_base + np.sin(i/10) * 0.0005 + np.random.normal(0, 0.0001)
                        lon = lon_base + np.cos(i/10) * 0.0005 + np.random.normal(0, 0.0001)
                        
                        sample_data.append({
                            'timestamp': base_time + time_offset,
                            'player_id': player_id_global,
                            'team_id': team_id,
                            'lat': lat,
                            'lon': lon
                        })
            
            sample_df = pd.DataFrame(sample_data)
            
            # Display the first few rows of the sample data
            st.subheader("Sample Data Preview")
            st.dataframe(sample_df.head())
            
            # Initialize the analyzer with the sample data
            st.session_state.analyzer = SoccerGPSAnalyzer(data_df=sample_df)
            st.session_state.data_loaded = True
            st.session_state.player_ids = sorted(sample_df['player_id'].unique())
            
            st.success("Sample data loaded successfully! Navigate to Player Analysis or Team Comparison to explore the data.")
        
        else:
            # File uploader widget
            uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    # Read the CSV file
                    data = pd.read_csv(uploaded_file)
                    
                    # Check if the CSV has the required columns
                    required_columns = ['timestamp', 'player_id', 'team_id', 'lat', 'lon']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    
                    if missing_columns:
                        st.error(f"CSV file is missing required columns: {', '.join(missing_columns)}")
                        st.info("Required columns: timestamp, player_id, team_id, lat, lon")
                    else:
                        # Display the first few rows of the uploaded data
                        st.subheader("Data Preview")
                        st.dataframe(data.head())
                        
                        # Initialize the analyzer with the uploaded data
                        st.session_state.analyzer = SoccerGPSAnalyzer(data_df=data)
                        st.session_state.data_loaded = True
                        st.session_state.player_ids = sorted(data['player_id'].unique())
                        
                        st.success("Data loaded successfully! Navigate to Player Analysis or Team Comparison to explore the data.")
                except Exception as e:
                    st.error(f"Error loading the data: {e}")
    
    elif app_mode == "Player Analysis":
        st.header("Player Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload data first or use sample data on the Upload Data page.")
            return
            
        # Add an expander with explanation of the analysis
        with st.expander("About Player Analysis"):
            st.markdown("""
            This page provides detailed analysis of individual player performance based on GPS tracking data.
            
            **Key Metrics Explained:**
            - **Total Distance**: Total distance covered by the player during the match
            - **Game Time**: Total time the player was tracked during the match
            - **Average Speed**: Average speed maintained throughout the game
            - **Top Speed**: Maximum speed reached during the match
            - **Total Sprints**: Number of times the player accelerated above 5.5 m/s (19.8 km/h)
            - **Sprint Distance**: Total distance covered while sprinting
            - **Work Rate**: Average distance covered per second (intensity metric)
            - **Distance per Minute**: Average distance covered each minute
            
            **Speed Zones:**
            - Walking: 0-2 m/s (0-7.2 km/h)
            - Jogging: 2-4 m/s (7.2-14.4 km/h)
            - Running: 4-5.5 m/s (14.4-19.8 km/h)
            - Sprinting: 5.5-7 m/s (19.8-25.2 km/h)
            - Max Sprint: 7+ m/s (25.2+ km/h)
            """)
            
        
        # Player selection
        player_id = st.selectbox("Select Player", st.session_state.player_ids)
        
        if st.button("Analyze Player"):
            # Show a spinner while analyzing
            with st.spinner("Analyzing player data..."):
                # Get player metrics
                player_metrics = st.session_state.analyzer.analyze_player(player_id)
                
                if not player_metrics:
                    st.error(f"No data available for Player {player_id}")
                    return
                
                # Display player information
                st.subheader(f"Player {player_id} (Team {player_metrics['team_id']})")
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                # Display key metrics in the first column
                with col1:
                    st.metric("Total Distance", f"{player_metrics['total_distance']:.1f} m")
                    st.metric("Game Time", f"{player_metrics['game_time_minutes']:.1f} min")
                    st.metric("Average Speed", f"{player_metrics['avg_speed']:.2f} m/s")
                    st.metric("Top Speed", f"{player_metrics['top_speed']:.2f} m/s")
                
                # Display more metrics in the second column
                with col2:
                    st.metric("Total Sprints", f"{player_metrics['total_sprints']}")
                    st.metric("Sprint Distance", f"{player_metrics['sprint_distance']:.1f} m")
                    st.metric("Work Rate", f"{player_metrics['work_rate']:.2f} m/s")
                    st.metric("Distance per Minute", f"{player_metrics['distance_per_minute']:.1f} m/min")
                
                # Display acceleration metrics
                st.subheader("Acceleration & Deceleration")
                acc_col1, acc_col2 = st.columns(2)
                
                with acc_col1:
                    st.metric("Average Acceleration", f"{player_metrics['avg_acceleration']:.2f} m/s²")
                    st.metric("Maximum Acceleration", f"{player_metrics['max_acceleration']:.2f} m/s²")
                
                with acc_col2:
                    st.metric("Average Deceleration", f"{player_metrics['avg_deceleration']:.2f} m/s²")
                    st.metric("Maximum Deceleration", f"{player_metrics['max_deceleration']:.2f} m/s²")
                
                # Display speed zone information
                st.subheader("Speed Zones Analysis")
                
                # Create a table for speed zone times
                speed_zone_df = pd.DataFrame(player_metrics['speed_zone_times'])
                speed_zone_df['time_minutes'] = speed_zone_df['time_seconds'] / 60
                speed_zone_df = speed_zone_df[['zone', 'time_minutes', 'percentage']]
                speed_zone_df.columns = ['Speed Zone', 'Time (minutes)', 'Percentage (%)']
                speed_zone_df['Time (minutes)'] = speed_zone_df['Time (minutes)'].round(2)
                speed_zone_df['Percentage (%)'] = speed_zone_df['Percentage (%)'].round(2)
                
                st.dataframe(speed_zone_df)
                
                # Generate plots
                st.subheader("Visualizations")
                plots = st.session_state.analyzer.generate_player_summary_plots(player_id)
                
                # Display the speed profile
                if 'speed_profile' in plots:
                    st.subheader("Speed Profile")
                    st.pyplot(plots['speed_profile'])
                
                # Display the summary plot
                if 'summary' in plots:
                    st.subheader("Performance Summary")
                    st.pyplot(plots['summary'])
    
    elif app_mode == "Team Comparison":
        st.header("Team Comparison")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload data first or use sample data on the Upload Data page.")
            return
        
        # Add an expander with explanation of the team comparison
        with st.expander("About Team Comparison"):
            st.markdown("""
            This page compares the performance of two teams based on aggregated GPS data from their players.
            
            **Features:**
            - **Aggregate Team Metrics**: Summary table of total and average metrics for each team.
            - **Player Metric Distributions**: Box plots showing the distribution of key metrics across players.
            - **Team Position Heatmaps**: Visualizations of each team's overall positioning on the field.
            
            **Metrics Compared:**
            - Total Distance (m)
            - Average Distance per Player (m)
            - Total Sprints
            - Average Sprints per Player
            - Average Speed (m/s)
            - Top Speed (m/s)
            """)
        
        # Get all player metrics
        player_metrics = st.session_state.analyzer.analyze_all_players()
        
        # Separate into teams
        team1_players = [p for p in player_metrics if p['team_id'] == 1]
        team2_players = [p for p in player_metrics if p['team_id'] == 2]
        
        # Check if both teams have players
        if not team1_players or not team2_players:
            st.error("Data must include players from both teams.")
            return
        
        # Compute aggregate metrics
        total_distance_team1 = sum(p['total_distance'] for p in team1_players)
        total_distance_team2 = sum(p['total_distance'] for p in team2_players)
        avg_distance_team1 = np.mean([p['total_distance'] for p in team1_players])
        avg_distance_team2 = np.mean([p['total_distance'] for p in team2_players])
        total_sprints_team1 = sum(p['total_sprints'] for p in team1_players)
        total_sprints_team2 = sum(p['total_sprints'] for p in team2_players)
        avg_sprints_team1 = np.mean([p['total_sprints'] for p in team1_players])
        avg_sprints_team2 = np.mean([p['total_sprints'] for p in team2_players])
        avg_speed_team1 = np.mean([p['avg_speed'] for p in team1_players])
        avg_speed_team2 = np.mean([p['avg_speed'] for p in team2_players])
        top_speed_team1 = max(p['top_speed'] for p in team1_players)
        top_speed_team2 = max(p['top_speed'] for p in team2_players)
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': ['Total Distance (m)', 'Avg Distance per Player (m)', 'Total Sprints', 
                       'Avg Sprints per Player', 'Avg Speed (m/s)', 'Top Speed (m/s)'],
            'Team 1': [total_distance_team1, avg_distance_team1, total_sprints_team1, 
                       avg_sprints_team1, avg_speed_team1, top_speed_team1],
            'Team 2': [total_distance_team2, avg_distance_team2, total_sprints_team2, 
                       avg_sprints_team2, avg_speed_team2, top_speed_team2]
        })
        
        # Round the values
        metrics_df['Team 1'] = metrics_df['Team 1'].round(2)
        metrics_df['Team 2'] = metrics_df['Team 2'].round(2)
        
        # Display team info
        st.markdown("### Team Overview")
        st.write(f"Team 1: {len(team1_players)} players")
        st.write(f"Team 2: {len(team2_players)} players")
        
        # Display aggregate metrics
        st.markdown("### Aggregate Team Metrics")
        st.write("This table shows the overall performance metrics for each team.")
        st.dataframe(metrics_df)
        st.markdown(get_download_link(metrics_df, "team_metrics.csv", "Download Team Metrics CSV"), unsafe_allow_html=True)
        
        # Create box plots
        st.markdown("### Player Metric Distributions")
        st.write("These box plots compare the distribution of individual player metrics between the two teams.")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics_to_plot = ['total_distance', 'avg_speed', 'total_sprints']
        titles = ['Total Distance (m)', 'Average Speed (m/s)', 'Total Sprints']
        for ax, metric, title in zip(axes, metrics_to_plot, titles):
            team1_data = [p[metric] for p in team1_players]
            team2_data = [p[metric] for p in team2_players]
            ax.boxplot([team1_data, team2_data], labels=['Team 1', 'Team 2'])
            ax.set_title(title)
            ax.set_ylabel(title)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Create heatmaps
        st.markdown("### Team Position Heatmaps")
        st.write("These heatmaps show the overall positioning of each team on the field based on all players' GPS data.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Team 1 Position Heatmap")
            team1_positions = st.session_state.analyzer.data[
                st.session_state.analyzer.data['team_id'] == 1][['lat', 'lon']].values
            fig_team1 = plt.figure(figsize=(8, 6))
            ax = fig_team1.add_subplot(111)
            st.session_state.analyzer._plot_heatmap(ax, team1_positions)
            st.pyplot(fig_team1)
        with col2:
            st.subheader("Team 2 Position Heatmap")
            team2_positions = st.session_state.analyzer.data[
                st.session_state.analyzer.data['team_id'] == 2][['lat', 'lon']].values
            fig_team2 = plt.figure(figsize=(8, 6))
            ax = fig_team2.add_subplot(111)
            st.session_state.analyzer._plot_heatmap(ax, team2_positions)
            st.pyplot(fig_team2)

if __name__ == '__main__':
    main()