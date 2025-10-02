import json
import pandas as pd

class RaceReportGenerator:
    """
    Processes race timing data from a JSON file to generate a detailed
    performance report and rankings for each driver, including 3-lap and 5-lap averages.
    """
    def __init__(self, filepath):
        """
        Initializes the report generator with the path to the input JSON file.

        Args:
            filepath (str): The path to the race data JSON file.
        """
        self.filepath = filepath
        self.raw_data = None
        self.driver_stats = []
        self.final_report = {}

    def _load_data(self):
        """Loads the race data from the specified JSON file."""
        try:
            with open(self.filepath, 'r') as f:
                self.raw_data = json.load(f)
            print(f"Successfully loaded data from '{self.filepath}'")
        except FileNotFoundError:
            print(f"Error: The file '{self.filepath}' was not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{self.filepath}'.")
            raise

    @staticmethod
    def _time_to_seconds(time_str):
        """
        Converts a time string (e.g., '1:35.531' or '35.531') to seconds.
        Returns None if the format is invalid.
        """
        if not time_str:
            return None
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            return float(time_str)
        except (ValueError, TypeError):
            return None

    def _process_data(self):
        """
        Flattens the raw data into a pandas DataFrame of valid laps,
        making it suitable for analysis.
        """
        if not self.raw_data:
            return pd.DataFrame()

        flat_laps = []
        for participant in self.raw_data.get('participants', []):
            driver_map = {
                str(d.get('number')): {
                    'full_name': f"{d.get('firstname', '')} {d.get('surname', '')}".strip(),
                    'license': d.get('license')
                } for d in participant.get('drivers', [])
            }

            for lap in participant.get('laps', []):
                if lap.get('is_valid') and 'sector_times' in lap and len(lap['sector_times']) >= 3:
                    driver_details = driver_map.get(lap.get('driver_number'))
                    if driver_details:
                        lap_time = self._time_to_seconds(lap.get('time'))
                        s1_time = self._time_to_seconds(lap['sector_times'][0].get('time'))
                        s2_time = self._time_to_seconds(lap['sector_times'][1].get('time'))
                        s3_time = self._time_to_seconds(lap['sector_times'][2].get('time'))

                        if all(t is not None for t in [lap_time, s1_time, s2_time, s3_time]):
                            flat_laps.append({
                                'driver_name': driver_details['full_name'],
                                'license': driver_details['license'],
                                'vehicle': participant.get('vehicle'),
                                'class': participant.get('class'),
                                'lap_time': lap_time,
                                's1_time': s1_time,
                                's2_time': s2_time,
                                's3_time': s3_time
                            })
        
        print(f"Processed {len(flat_laps)} valid laps.")
        return pd.DataFrame(flat_laps)

    @staticmethod
    def _calculate_stats(series):
        """
        Calculates fastest time, averages of the best 3 and 5 times, and their
        respective deviations from the fastest time.
        """
        sorted_times = sorted(series.dropna())
        if not sorted_times:
            return {
                'fastest': None,
                'best_3_avg': None, 'best_5_avg': None,
                'deviation_3_lap': None, 'deviation_5_lap': None
            }

        fastest = sorted_times[0]

        # Calculate 3-lap average and deviation
        num_laps_for_3_avg = min(3, len(sorted_times))
        best_3_laps = sorted_times[:num_laps_for_3_avg]
        best_3_avg = sum(best_3_laps) / len(best_3_laps) if best_3_laps else None
        deviation_3_lap = best_3_avg - fastest if best_3_avg is not None else None

        # Calculate 5-lap average and deviation
        num_laps_for_5_avg = min(5, len(sorted_times))
        best_5_laps = sorted_times[:num_laps_for_5_avg]
        best_5_avg = sum(best_5_laps) / len(best_5_laps) if best_5_laps else None
        deviation_5_lap = best_5_avg - fastest if best_5_avg is not None else None

        return {
            'fastest': fastest,
            'best_3_avg': best_3_avg,
            'best_5_avg': best_5_avg,
            'deviation_3_lap': deviation_3_lap,
            'deviation_5_lap': deviation_5_lap
        }

    def _calculate_all_driver_stats(self, laps_df):
        """
        Groups the data by driver and calculates all required performance metrics
        for laps and sectors.
        """
        if laps_df.empty:
            print("No valid lap data found to calculate statistics.")
            return

        for group_keys, group_df in laps_df.groupby(['driver_name', 'license', 'vehicle', 'class']):
            driver_name, license, vehicle, p_class = group_keys

            self.driver_stats.append({
                'driver_name': driver_name, 'license': license,
                'vehicle': vehicle, 'class': p_class,
                'lap_times': self._calculate_stats(group_df['lap_time']),
                'sector_1_times': self._calculate_stats(group_df['s1_time']),
                'sector_2_times': self._calculate_stats(group_df['s2_time']),
                'sector_3_times': self._calculate_stats(group_df['s3_time'])
            })
        print(f"Calculated statistics for {len(self.driver_stats)} drivers.")

    def _generate_rankings(self):
        """
        Generates overall, by-license, and by-vehicle rankings based on the
        calculated driver statistics.
        """
        def create_ranking(stats, time_key, metric_key):
            """Helper to create a single ranked list."""
            ranked_list = [
                {'driver_name': d['driver_name'], 'license': d['license'],
                 'vehicle': d['vehicle'], 'value': d.get(time_key, {}).get(metric_key)}
                for d in stats if d.get(time_key, {}).get(metric_key) is not None
            ]
            return sorted(ranked_list, key=lambda x: x['value'])

        def generate_rankings_for_group(stats):
            """Helper to generate a full set of rankings for a list of drivers."""
            rankings = {}
            time_metrics = {
                'lap': 'lap_times', 's1': 'sector_1_times',
                's2': 'sector_2_times', 's3': 'sector_3_times'
            }
            # Expanded list of metrics to rank by
            metrics_to_rank = ['fastest', 'best_3_avg', 'best_5_avg', 'deviation_3_lap', 'deviation_5_lap']
            
            for metric in metrics_to_rank:
                for name, key in time_metrics.items():
                    rankings[f'by_{metric}_{name}'] = create_ranking(stats, key, metric)
            return rankings

        # Group stats by license and vehicle
        license_groups = pd.DataFrame(self.driver_stats).groupby('license')
        vehicle_groups = pd.DataFrame(self.driver_stats).groupby('vehicle')

        return {
            'overall': generate_rankings_for_group(self.driver_stats),
            'by_license': {
                name: generate_rankings_for_group(group.to_dict('records'))
                for name, group in license_groups
            },
            'by_vehicle': {
                name: generate_rankings_for_group(group.to_dict('records'))
                for name, group in vehicle_groups
            }
        }

    def generate_report(self):
        """
        Executes the full workflow: loads data, processes laps, calculates
        stats, generates rankings, and assembles the final report.

        Returns:
            dict: The complete, structured JSON report as a Python dictionary.
        """
        self._load_data()
        laps_df = self._process_data()
        self._calculate_all_driver_stats(laps_df)
        rankings = self._generate_rankings()
        
        self.final_report = {
            'driver_performance': self.driver_stats,
            'rankings': rankings
        }
        print("Final report has been generated.")
        return self.final_report

    def save_report(self, output_filepath):
        """Saves the generated report to a specified JSON file."""
        if not self.final_report:
            print("No report has been generated to save.")
            return

        with open(output_filepath, 'w') as f:
            json.dump(self.final_report, f, indent=2)
        print(f"Report successfully saved to '{output_filepath}'")


# --- Example Usage ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Replace 'race_data.json' with the actual name of your input file.
    INPUT_FILE = "test_2025_data.json"
    # The name of the output file that will be created.
    OUTPUT_FILE = "race_report_output_v2.json"

    # --- EXECUTION ---
    try:
        # 1. Initialize the generator with the input file path.
        report_generator = RaceReportGenerator(INPUT_FILE)
        
        # 2. Run the full report generation process.
        report_generator.generate_report()
        
        # 3. Save the final report to a JSON file.
        report_generator.save_report(OUTPUT_FILE)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")