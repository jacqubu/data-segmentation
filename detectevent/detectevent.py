import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional, Callable
from collections import defaultdict

class DetectEvent:
    """
    A library for detecting significant events in water data time series,
    with capabilities to combine multiple detection methods.
    """

    def __init__(self):
        self.data = None
        self.time_column = None
        self.value_column = None
        # Store detection methods for combined detection
        self.detection_methods = {}

    def load_data(self, filepath: str, time_column: str = 'datetime',
                  value_column: str = 'gage_height',
                  date_format: str = None) -> pd.DataFrame:
        """
        Load data from a CSV file

        Args:
            filepath: Path to the CSV file
            time_column: Column name containing timestamps
            value_column: Column name containing water level measurements
            date_format: Format string for parsing dates if needed

        Returns:
            Loaded DataFrame
        """
        self.data = pd.read_csv(filepath)
        self.time_column = time_column
        self.value_column = value_column

        # Convert time column to datetime if it's not already
        if date_format:
            self.data[time_column] = pd.to_datetime(self.data[time_column], format=date_format)
        else:
            self.data[time_column] = pd.to_datetime(self.data[time_column])

        # Sort by time
        self.data = self.data.sort_values(by=time_column)

        return self.data

    def detect_events_threshold(self,
                              threshold: float,
                              min_duration: int = 5,
                              rise_rate: float = None) -> List[Dict[str, Union[datetime, float]]]:
        """
        Detect water events based on a simple threshold approach.

        Args:
            threshold: The water level value above which an event is triggered
            min_duration: Minimum number of consecutive readings to be considered an event
            rise_rate: Optional rate of change threshold to detect rapid rises

        Returns:
            List of dictionaries containing event details (start time, end time, peak, etc.)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        events = []
        in_event = False
        event_start = None
        event_peak = -float('inf')
        event_peak_time = None

        # Calculate rate of change if needed
        if rise_rate is not None:
            self.data['rate_of_change'] = self.data[self.value_column].diff() / \
                                         (self.data[self.time_column].diff().dt.total_seconds() / 3600)  # per hour

        # Iterate through the data
        for i, row in self.data.iterrows():
            value = row[self.value_column]
            current_time = row[self.time_column]

            # Check if this is an event point
            is_event_point = value > threshold
            if rise_rate is not None:
                if not np.isnan(row.get('rate_of_change', np.nan)):
                    is_event_point = is_event_point or (row['rate_of_change'] > rise_rate)

            # Start of a potential event
            if is_event_point and not in_event:
                in_event = True
                event_start = current_time
                event_peak = value
                event_peak_time = current_time

            # During an event, update peak
            elif is_event_point and in_event:
                if value > event_peak:
                    event_peak = value
                    event_peak_time = current_time

            # End of an event
            elif not is_event_point and in_event:
                in_event = False
                # Calculate duration
                event_duration = (current_time - event_start).total_seconds() / 60  # in minutes

                # Only add events that meet minimum duration
                if event_duration >= min_duration:
                    events.append({
                        'start_time': event_start,
                        'end_time': current_time,
                        'duration_minutes': event_duration,
                        'peak_value': event_peak,
                        'peak_time': event_peak_time,
                        'method': 'threshold'
                    })

        # Handle case where data ends during an event
        if in_event:
            event_duration = (self.data[self.time_column].iloc[-1] - event_start).total_seconds() / 60
            if event_duration >= min_duration:
                events.append({
                    'start_time': event_start,
                    'end_time': self.data[self.time_column].iloc[-1],
                    'duration_minutes': event_duration,
                    'peak_value': event_peak,
                    'peak_time': event_peak_time,
                    'method': 'threshold'
                })

        # Register this method in our detection methods dictionary
        method_name = f"threshold_{threshold}"
        self.detection_methods[method_name] = events

        return events

    def detect_events_moving_average(self,
                                    window_size: int = 24,
                                    std_factor: float = 2.0,
                                    min_duration: int = 5) -> List[Dict[str, Union[datetime, float]]]:
        """
        Detect water events using a moving average and standard deviation approach.
        Useful for detecting anomalies relative to recent behavior.

        Args:
            window_size: Number of readings to include in moving window
            std_factor: Number of standard deviations above moving average to trigger an event
            min_duration: Minimum number of consecutive readings to be considered an event

        Returns:
            List of dictionaries containing event details
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Calculate rolling mean and standard deviation
        self.data['rolling_mean'] = self.data[self.value_column].rolling(window=window_size, center=False).mean()
        self.data['rolling_std'] = self.data[self.value_column].rolling(window=window_size, center=False).std()

        # Calculate dynamic threshold
        self.data['threshold'] = self.data['rolling_mean'] + (std_factor * self.data['rolling_std'])

        # Start detection after we have enough data for the rolling window
        detect_data = self.data.iloc[window_size:].copy()

        # Use the threshold method with our dynamic threshold
        events = []
        in_event = False
        event_start = None
        event_peak = -float('inf')
        event_peak_time = None

        for i, row in detect_data.iterrows():
            value = row[self.value_column]
            threshold = row['threshold']
            current_time = row[self.time_column]

            # Check if this is an event point
            is_event_point = value > threshold

            # Start of a potential event
            if is_event_point and not in_event:
                in_event = True
                event_start = current_time
                event_peak = value
                event_peak_time = current_time

            # During an event, update peak
            elif is_event_point and in_event:
                if value > event_peak:
                    event_peak = value
                    event_peak_time = current_time

            # End of an event
            elif not is_event_point and in_event:
                in_event = False
                # Calculate duration
                event_duration = (current_time - event_start).total_seconds() / 60  # in minutes

                # Only add events that meet minimum duration
                if event_duration >= min_duration:
                    events.append({
                        'start_time': event_start,
                        'end_time': current_time,
                        'duration_minutes': event_duration,
                        'peak_value': event_peak,
                        'peak_time': event_peak_time,
                        'method': 'moving_average'
                    })

        # Handle case where data ends during an event
        if in_event:
            event_duration = (detect_data[self.time_column].iloc[-1] - event_start).total_seconds() / 60
            if event_duration >= min_duration:
                events.append({
                    'start_time': event_start,
                    'end_time': detect_data[self.time_column].iloc[-1],
                    'duration_minutes': event_duration,
                    'peak_value': event_peak,
                    'peak_time': event_peak_time,
                    'method': 'moving_average'
                })

        # Register this method in our detection methods dictionary
        method_name = f"moving_avg_{window_size}_{std_factor}"
        self.detection_methods[method_name] = events

        return events

    def detect_events_derivative(self,
                              rise_threshold: float = 0.5,
                              window_size: int = 5,
                              min_duration: int = 5) -> List[Dict[str, Union[datetime, float]]]:
        """
        Detect water events based on rapid changes in water level (derivative method).

        Args:
            rise_threshold: Minimum rate of rise to trigger an event (units per hour)
            window_size: Window size for smoothing derivatives
            min_duration: Minimum number of consecutive readings to be considered an event

        Returns:
            List of dictionaries containing event details
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Calculate the rate of change
        self.data['rate_of_change'] = self.data[self.value_column].diff() / \
                                    (self.data[self.time_column].diff().dt.total_seconds() / 3600)  # per hour

        # Apply smoothing to reduce noise
        self.data['smooth_rate'] = self.data['rate_of_change'].rolling(window=window_size, center=True).mean()

        # Start detection after we have enough data
        detect_data = self.data.dropna(subset=['smooth_rate']).copy()

        events = []
        in_event = False
        event_start = None
        event_peak = -float('inf')
        event_peak_time = None
        max_rate = -float('inf')

        for i, row in detect_data.iterrows():
            value = row[self.value_column]
            rate = row['smooth_rate']
            current_time = row[self.time_column]

            # Check if this is an event point
            is_event_point = rate > rise_threshold

            # Start of a potential event
            if is_event_point and not in_event:
                in_event = True
                event_start = current_time
                event_peak = value
                event_peak_time = current_time
                max_rate = rate

            # During an event, update peak
            elif is_event_point and in_event:
                if value > event_peak:
                    event_peak = value
                    event_peak_time = current_time
                if rate > max_rate:
                    max_rate = rate

            # End of an event
            elif not is_event_point and in_event:
                in_event = False
                # Calculate duration
                event_duration = (current_time - event_start).total_seconds() / 60  # in minutes

                # Only add events that meet minimum duration
                if event_duration >= min_duration:
                    events.append({
                        'start_time': event_start,
                        'end_time': current_time,
                        'duration_minutes': event_duration,
                        'peak_value': event_peak,
                        'peak_time': event_peak_time,
                        'max_rise_rate': max_rate,
                        'method': 'derivative'
                    })

        # Handle case where data ends during an event
        if in_event:
            event_duration = (detect_data[self.time_column].iloc[-1] - event_start).total_seconds() / 60
            if event_duration >= min_duration:
                events.append({
                    'start_time': event_start,
                    'end_time': detect_data[self.time_column].iloc[-1],
                    'duration_minutes': event_duration,
                    'peak_value': event_peak,
                    'peak_time': event_peak_time,
                    'max_rise_rate': max_rate,
                    'method': 'derivative'
                })

        # Register this method in our detection methods dictionary
        method_name = f"derivative_{rise_threshold}"
        self.detection_methods[method_name] = events

        return events

    def register_detection_method(self, method_name: str, events: List[Dict[str, Union[datetime, float]]]):
        """
        Register a custom detection method for use in combined detection.

        Args:
            method_name: Name to identify this detection method
            events: List of event dictionaries from this method
        """
        self.detection_methods[method_name] = events

    def _are_events_similar(self, event1: Dict, event2: Dict, time_tolerance_percent: float = 2.0) -> bool:
        """
        Check if two events are similar based on their timing.

        Args:
            event1: First event dictionary
            event2: Second event dictionary
            time_tolerance_percent: Percentage of the first event's duration to consider as tolerance

        Returns:
            Boolean indicating if events are similar
        """
        # Calculate the time tolerance in minutes
        event1_duration = event1['duration_minutes']
        tolerance_minutes = (time_tolerance_percent / 100) * event1_duration

        # Convert tolerance to timedelta
        tolerance = timedelta(minutes=tolerance_minutes)

        # Check if event start and end times are within tolerance
        start_diff = abs((event1['start_time'] - event2['start_time']).total_seconds() / 60)
        end_diff = abs((event1['end_time'] - event2['end_time']).total_seconds() / 60)

        return start_diff <= tolerance_minutes and end_diff <= tolerance_minutes

    def _cluster_similar_events(self, all_events: List[Dict], time_tolerance_percent: float = 2.0) -> List[List[Dict]]:
        """
        Group events from different methods that are likely detecting the same event.

        Args:
            all_events: List of all events from different detection methods
            time_tolerance_percent: Percentage tolerance for event similarity

        Returns:
            List of event clusters, where each cluster is a list of similar events
        """
        # Sort all events by start time
        sorted_events = sorted(all_events, key=lambda x: x['start_time'])

        # Initialize clusters
        clusters = []

        # Process each event
        for event in sorted_events:
            # Check if event belongs to any existing cluster
            found_cluster = False
            for cluster in clusters:
                # Compare with the first event in the cluster as reference
                if self._are_events_similar(cluster[0], event, time_tolerance_percent):
                    cluster.append(event)
                    found_cluster = True
                    break

            # If no matching cluster, start a new one
            if not found_cluster:
                clusters.append([event])

        return clusters

    def combine_detection_methods(self,
                                agreement_threshold: float = 0.5,
                                time_tolerance_percent: float = 2.0) -> List[Dict]:
        """
        Combine multiple detection methods using a voting system.

        Args:
            agreement_threshold: Fraction of methods that must agree for an event to be valid (0-1)
            time_tolerance_percent: Percentage of event duration to use as time tolerance

        Returns:
            List of combined events that meet the agreement threshold
        """
        if not self.detection_methods:
            raise ValueError("No detection methods registered. Run detection methods first.")

        # Collect all events from all methods
        all_events = []
        for method_name, events in self.detection_methods.items():
            for event in events:
                event_copy = event.copy()
                event_copy['detection_method'] = method_name
                all_events.append(event_copy)

        # Cluster similar events
        clusters = self._cluster_similar_events(all_events, time_tolerance_percent)

        # Count unique methods in each cluster
        combined_events = []
        total_methods = len(self.detection_methods)

        for cluster in clusters:
            # Count unique methods in this cluster
            methods_in_cluster = set(event['detection_method'] for event in cluster)
            agreement_ratio = len(methods_in_cluster) / total_methods

            # If enough methods agree, create a combined event
            if agreement_ratio >= agreement_threshold:
                # Average the start, end, and peak times
                start_times = [event['start_time'] for event in cluster]
                end_times = [event['end_time'] for event in cluster]
                peak_times = [event['peak_time'] for event in cluster]
                peak_values = [event['peak_value'] for event in cluster]

                # Create a combined event using median values for robustness
                combined_event = {
                    'start_time': pd.Series(start_times).median(),
                    'end_time': pd.Series(end_times).median(),
                    'peak_time': pd.Series(peak_times).median(),
                    'peak_value': np.median(peak_values),
                    'duration_minutes': (pd.Series(end_times).median() - pd.Series(start_times).median()).total_seconds() / 60,
                    'agreement_ratio': agreement_ratio,
                    'method': 'combined',
                    'supporting_methods': list(methods_in_cluster),
                    'detection_count': len(cluster)
                }

                combined_events.append(combined_event)

        return combined_events

    def visualize_events(self, events: List[Dict],
                         title: str = "Water Level Events",
                         show_thresholds: bool = False,
                         highlight_combined: bool = False):
        """
        Visualize detected events on the time series data

        Args:
            events: List of event dictionaries
            title: Plot title
            show_thresholds: Whether to show dynamic thresholds if available
            highlight_combined: Whether to highlight combined events differently
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        plt.figure(figsize=(15, 8))

        # Plot water level data
        plt.plot(self.data[self.time_column], self.data[self.value_column],
                 label='Water Level', color='blue', alpha=0.7)

        # Plot thresholds if available
        if show_thresholds and 'threshold' in self.data.columns:
            plt.plot(self.data[self.time_column], self.data['threshold'],
                     label='Detection Threshold', color='red', linestyle='--', alpha=0.5)

        # Highlight events
        combined_events = [e for e in events if e.get('method') == 'combined']
        other_events = [e for e in events if e.get('method') != 'combined']

        # Plot non-combined events
        for event in other_events:
            start = event['start_time']
            end = event['end_time']
            peak_time = event['peak_time']
            peak_value = event['peak_value']
            method = event.get('method', 'unknown')

            # Get data points during the event
            event_mask = (self.data[self.time_column] >= start) & (self.data[self.time_column] <= end)

            # Highlight event region
            plt.axvspan(start, end, alpha=0.2, color='orange')

            # Mark peak
            plt.plot(peak_time, peak_value, 'rD', markersize=8)
            # plt.annotate(f"{method}: {peak_value:.2f}",
            #              (peak_time, peak_value),
            #              xytext=(10, 10),
            #              textcoords='offset points',
            #              fontsize=8,
            #              arrowprops=dict(arrowstyle='->', alpha=0.5))

        # Plot combined events with special highlighting if requested
        if highlight_combined:
            for event in combined_events:
                start = event['start_time']
                end = event['end_time']
                peak_time = event['peak_time']
                peak_value = event['peak_value']
                agreement = event.get('agreement_ratio', 0) * 100

                # Highlight combined events more prominently
                plt.axvspan(start, end, alpha=0.3, color='green')

                # Mark peak with larger marker
                plt.plot(peak_time, peak_value, 'g*', markersize=12)
                plt.annotate(f"Combined ({agreement:.0f}%): {peak_value:.2f}",
                            (peak_time, peak_value),
                            xytext=(10, 20),
                            textcoords='offset points',
                            fontsize=10,
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='green'))

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(self.value_column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print event summary
        if combined_events and highlight_combined:
            print(f"Detected {len(combined_events)} combined events:")
            for i, event in enumerate(combined_events):
                supporting = ", ".join(event.get('supporting_methods', []))
                print(f"Combined Event {i+1}: Start={event['start_time']}, End={event['end_time']}, "
                    f"Peak={event['peak_value']:.2f} at {event['peak_time']}, "
                    f"Agreement={event['agreement_ratio']*100:.1f}%, Supporting: {supporting}")
        else:
            print(f"Detected {len(events)} events:")
            for i, event in enumerate(events):
                method = event.get('method', 'unknown')
                print(f"Event {i+1} ({method}): Start={event['start_time']}, End={event['end_time']}, "
                    f"Peak={event['peak_value']:.2f} at {event['peak_time']}")


if __name__ == "__main__":
    detector = DetectEvent()

    # Load data
    data = detector.load_data("usgs-data/01302020_modified.csv",
                             time_column="datetime",
                             value_column="00065",
                             date_format="%Y-%m-%d %H:%M:%S%z")

    # Apply multiple detection methods
    # detector.detect_events_threshold(threshold=10.0, min_duration=30)
    events_ma = detector.detect_events_moving_average(window_size=24, std_factor=2.0, min_duration=30)
    events_der = detector.detect_events_derivative(rise_threshold=0.5, min_duration=30)

    # Get combined events where at least 50% of methods agree
    # combined_events = detector.combine_detection_methods(agreement_threshold=0.5, time_tolerance_percent=90.0)

    # Visualize the combined results
    # detector.visualize_events(combined_events, title="Combined Water Level Events", highlight_combined=True)
    detector.visualize_events(events_ma, title="Moving Average")
    detector.visualize_events(events_der, title="Derivatives")






    # Detect events using a simple threshold approach
    # events = detector.detect_events_threshold(threshold=10.0,  # adjust threshold based on your data
    #                                        min_duration=30,   # minimum 30 minutes
    #                                        rise_rate=0.5)     # 0.5 feet per hour rise rate

    # Alternatively, detect events using moving average approach
    # events = detector.detect_events_moving_average(window_size=48,  # 48 data points
    #                                              std_factor=2.5,   # 2.5 standard deviations
    #                                              min_duration=30)  # minimum 30 minutes

    # # Visualize the results
    # detector.visualize_events(events, title="USGS Gage Height Events")



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from typing import List, Tuple, Dict, Optional, Union

# class DetectEvent:
#     """
#     A library for detecting significant events in water data time series.
#     Primarily designed for USGS gage height data.
#     """

#     def __init__(self):
#         self.data = None
#         self.time_column = None
#         self.value_column = None

#     def load_data(self, filepath: str, time_column: str = 'datetime',
#                   value_column: str = 'gage_height',
#                   date_format: str = None) -> pd.DataFrame:
#         """
#         Load data from a CSV file

#         Args:
#             filepath: Path to the CSV file
#             time_column: Column name containing timestamps
#             value_column: Column name containing water level measurements
#             date_format: Format string for parsing dates if needed

#         Returns:
#             Loaded DataFrame
#         """
#         self.data = pd.read_csv(filepath)
#         self.time_column = time_column
#         self.value_column = value_column

#         # Convert time column to datetime if it's not already
#         if date_format:
#             self.data[time_column] = pd.to_datetime(self.data[time_column], format=date_format)
#         else:
#             self.data[time_column] = pd.to_datetime(self.data[time_column])

#         # Sort by time
#         self.data = self.data.sort_values(by=time_column)

#         return self.data

#     def detect_events_threshold(self,
#                               threshold: float,
#                               min_duration: int = 5,
#                               rise_rate: float = None) -> List[Dict[str, Union[datetime, float]]]:
#         """
#         Detect water events based on a simple threshold approach.

#         Args:
#             threshold: The water level value above which an event is triggered
#             min_duration: Minimum number of consecutive readings to be considered an event
#             rise_rate: Optional rate of change threshold to detect rapid rises

#         Returns:
#             List of dictionaries containing event details (start time, end time, peak, etc.)
#         """
#         if self.data is None:
#             raise ValueError("No data loaded. Call load_data() first.")

#         events = []
#         in_event = False
#         event_start = None
#         event_peak = -float('inf')
#         event_peak_time = None

#         # Calculate rate of change if needed
#         if rise_rate is not None:
#             self.data['rate_of_change'] = self.data[self.value_column].diff() / \
#                                          (self.data[self.time_column].diff().dt.total_seconds() / 3600)  # per hour

#         # Iterate through the data
#         for i, row in self.data.iterrows():
#             value = row[self.value_column]
#             current_time = row[self.time_column]

#             # Check if this is an event point
#             is_event_point = value > threshold
#             if rise_rate is not None:
#                 if not np.isnan(row['rate_of_change']):
#                     is_event_point = is_event_point or (row['rate_of_change'] > rise_rate)

#             # Start of a potential event
#             if is_event_point and not in_event:
#                 in_event = True
#                 event_start = current_time
#                 event_peak = value
#                 event_peak_time = current_time

#             # During an event, update peak
#             elif is_event_point and in_event:
#                 if value > event_peak:
#                     event_peak = value
#                     event_peak_time = current_time

#             # End of an event
#             elif not is_event_point and in_event:
#                 in_event = False
#                 # Calculate duration
#                 event_duration = (current_time - event_start).total_seconds() / 60  # in minutes

#                 # Only add events that meet minimum duration
#                 if event_duration >= min_duration:
#                     events.append({
#                         'start_time': event_start,
#                         'end_time': current_time,
#                         'duration_minutes': event_duration,
#                         'peak_value': event_peak,
#                         'peak_time': event_peak_time
#                     })

#         # Handle case where data ends during an event
#         if in_event:
#             event_duration = (self.data[self.time_column].iloc[-1] - event_start).total_seconds() / 60
#             if event_duration >= min_duration:
#                 events.append({
#                     'start_time': event_start,
#                     'end_time': self.data[self.time_column].iloc[-1],
#                     'duration_minutes': event_duration,
#                     'peak_value': event_peak,
#                     'peak_time': event_peak_time
#                 })

#         return events

#     def detect_events_moving_average(self,
#                                     window_size: int = 24,
#                                     std_factor: float = 2.0,
#                                     min_duration: int = 5) -> List[Dict[str, Union[datetime, float]]]:
#         """
#         Detect water events using a moving average and standard deviation approach.
#         Useful for detecting anomalies relative to recent behavior.

#         Args:
#             window_size: Number of readings to include in moving window
#             std_factor: Number of standard deviations above moving average to trigger an event
#             min_duration: Minimum number of consecutive readings to be considered an event

#         Returns:
#             List of dictionaries containing event details
#         """
#         if self.data is None:
#             raise ValueError("No data loaded. Call load_data() first.")

#         # Calculate rolling mean and standard deviation
#         self.data['rolling_mean'] = self.data[self.value_column].rolling(window=window_size, center=False).mean()
#         self.data['rolling_std'] = self.data[self.value_column].rolling(window=window_size, center=False).std()

#         # Calculate dynamic threshold
#         self.data['threshold'] = self.data['rolling_mean'] + (std_factor * self.data['rolling_std'])

#         # Start detection after we have enough data for the rolling window
#         detect_data = self.data.iloc[window_size:].copy()

#         # Use the threshold method with our dynamic threshold
#         events = []
#         in_event = False
#         event_start = None
#         event_peak = -float('inf')
#         event_peak_time = None

#         for i, row in detect_data.iterrows():
#             value = row[self.value_column]
#             threshold = row['threshold']
#             current_time = row[self.time_column]

#             # Check if this is an event point
#             is_event_point = value > threshold

#             # Start of a potential event
#             if is_event_point and not in_event:
#                 in_event = True
#                 event_start = current_time
#                 event_peak = value
#                 event_peak_time = current_time

#             # During an event, update peak
#             elif is_event_point and in_event:
#                 if value > event_peak:
#                     event_peak = value
#                     event_peak_time = current_time

#             # End of an event
#             elif not is_event_point and in_event:
#                 in_event = False
#                 # Calculate duration
#                 event_duration = (current_time - event_start).total_seconds() / 60  # in minutes

#                 # Only add events that meet minimum duration
#                 if event_duration >= min_duration:
#                     events.append({
#                         'start_time': event_start,
#                         'end_time': current_time,
#                         'duration_minutes': event_duration,
#                         'peak_value': event_peak,
#                         'peak_time': event_peak_time
#                     })

#         # Handle case where data ends during an event
#         if in_event:
#             event_duration = (detect_data[self.time_column].iloc[-1] - event_start).total_seconds() / 60
#             if event_duration >= min_duration:
#                 events.append({
#                     'start_time': event_start,
#                     'end_time': detect_data[self.time_column].iloc[-1],
#                     'duration_minutes': event_duration,
#                     'peak_value': event_peak,
#                     'peak_time': event_peak_time
#                 })

#         return events

#     def visualize_events(self, events: List[Dict[str, Union[datetime, float]]],
#                          title: str = "Water Level Events",
#                          show_thresholds: bool = True):
#         """
#         Visualize detected events on the time series data

#         Args:
#             events: List of event dictionaries returned by detect_events methods
#             title: Plot title
#             show_thresholds: Whether to show dynamic thresholds if available
#         """
#         if self.data is None:
#             raise ValueError("No data loaded. Call load_data() first.")

#         plt.figure(figsize=(15, 8))

#         # Plot water level data
#         plt.plot(self.data[self.time_column], self.data[self.value_column],
#                  label='Water Level', color='blue', alpha=0.7)

#         # Plot thresholds if available
#         if show_thresholds and 'threshold' in self.data.columns:
#             plt.plot(self.data[self.time_column], self.data['threshold'],
#                      label='Detection Threshold', color='red', linestyle='--', alpha=0.5)

#         # Highlight events
#         for event in events:
#             start = event['start_time']
#             end = event['end_time']
#             peak_time = event['peak_time']
#             peak_value = event['peak_value']

#             # Get data points during the event
#             event_mask = (self.data[self.time_column] >= start) & (self.data[self.time_column] <= end)
#             event_data = self.data[event_mask]

#             # Highlight event region
#             plt.axvspan(start, end, alpha=0.2, color='orange')

#             # Mark peak
#             plt.plot(peak_time, peak_value, 'r*', markersize=10)
#             plt.annotate(f"Peak: {peak_value:.2f}",
#                          (peak_time, peak_value),
#                          xytext=(10, 20),
#                          textcoords='offset points',
#                          arrowprops=dict(arrowstyle='->'))

#         plt.title(title)
#         plt.xlabel('Time')
#         plt.ylabel(self.value_column)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()

#         # Print event summary
#         print(f"Detected {len(events)} events:")
#         for i, event in enumerate(events):
#             print(f"Event {i+1}: Start={event['start_time']}, End={event['end_time']}, "
#                   f"Peak={event['peak_value']:.2f} at {event['peak_time']}")


# # Example usage
# if __name__ == "__main__":
#     detector = DetectEvent()

#     # Load data - adjust the file path and column names to match your CSV
#     data = detector.load_data("usgs-data/01302020_modified.csv",
#                              time_column="datetime",
#                              value_column="00065",
#                              date_format="%Y-%m-%d %H:%M:%S%z")

#     # Detect events using a simple threshold approach
#     # events = detector.detect_events_threshold(threshold=10.0,  # adjust threshold based on your data
#     #                                        min_duration=30,   # minimum 30 minutes
#     #                                        rise_rate=0.5)     # 0.5 feet per hour rise rate

#     # Alternatively, detect events using moving average approach
#     events = detector.detect_events_moving_average(window_size=48,  # 48 data points
#                                                  std_factor=2.5,   # 2.5 standard deviations
#                                                  min_duration=30)  # minimum 30 minutes

#     # Visualize the results
#     detector.visualize_events(events, title="USGS Gage Height Events")