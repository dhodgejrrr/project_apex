import json
import pandas as pd
import numpy as np

class RaceReportGenerator:
    """
    Processes race timing data to generate a report with dynamically calculated,
    data-driven license adjustment factors for performance handicap ranking.
    """
    STINT_LAP_PENALTY = 0.0001

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.driver_stats = []
        self.final_report = {}
        # This will hold our dynamically generated license factors
        self.dynamic_license_factors = {}

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
        """Converts a time string to seconds."""
        if not time_str: return None
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                return int(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except (ValueError, TypeError):
            return None

    def _process_data(self):
        """
        Flattens raw data, identifies stints, and adds a 'stint_lap_number'
        to each valid lap.
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

            all_laps_for_car = sorted(participant.get('laps', []), key=lambda x: x.get('number', 0))
            driver_stint_counters = {num: 1 for num in driver_map.keys()}

            for lap in all_laps_for_car:
                driver_num = lap.get('driver_number')
                if not driver_num or driver_num not in driver_map:
                    continue

                current_stint_lap = driver_stint_counters[driver_num]

                if lap.get('is_valid') and 'sector_times' in lap and len(lap['sector_times']) >= 3:
                    lap_time = self._time_to_seconds(lap.get('time'))
                    s1_time = self._time_to_seconds(lap['sector_times'][0].get('time'))
                    s2_time = self._time_to_seconds(lap['sector_times'][1].get('time'))
                    s3_time = self._time_to_seconds(lap['sector_times'][2].get('time'))

                    if all(t is not None for t in [lap_time, s1_time, s2_time, s3_time]):
                        flat_laps.append({
                            'driver_name': driver_map[driver_num]['full_name'],
                            'license': driver_map[driver_num]['license'],
                            'vehicle': participant.get('vehicle'),
                            'class': participant.get('class'),
                            'lap_time': lap_time, 's1_time': s1_time,
                            's2_time': s2_time, 's3_time': s3_time,
                            'stint_lap_number': current_stint_lap
                        })

                if lap.get('crossing_pit_finish_lane', False):
                    driver_stint_counters[driver_num] = 1
                else:
                    driver_stint_counters[driver_num] += 1

        print(f"Processed {len(flat_laps)} valid laps with stint analysis.")
        return pd.DataFrame(flat_laps)

    @staticmethod
    def _calculate_stats(df_slice, time_column):
        """Calculates performance and stint metrics for a given set of times."""
        df = df_slice[[time_column, 'stint_lap_number']].dropna().sort_values(by=time_column)
        if df.empty:
            return {}

        fastest_row = df.iloc[0]
        best_3_laps = df.head(3)
        best_5_laps = df.head(5)

        return {
            'fastest': fastest_row[time_column],
            'fastest_stint_lap': int(fastest_row['stint_lap_number']),
            'best_3_avg': best_3_laps[time_column].mean(),
            'avg_stint_lap_for_best_3': round(best_3_laps['stint_lap_number'].mean(), 2),
            'best_5_avg': best_5_laps[time_column].mean(),
            'avg_stint_lap_for_best_5': round(best_5_laps['stint_lap_number'].mean(), 2),
            'deviation_3_lap': best_3_laps[time_column].mean() - fastest_row[time_column],
            'deviation_5_lap': best_5_laps[time_column].mean() - fastest_row[time_column]
        }

    def _calculate_all_driver_stats(self, laps_df):
        """Calculates all performance metrics for each driver."""
        if laps_df.empty: return
        for group_keys, group_df in laps_df.groupby(['driver_name', 'license', 'vehicle', 'class']):
            self.driver_stats.append({
                'driver_name': group_keys[0], 'license': group_keys[1],
                'vehicle': group_keys[2], 'class': group_keys[3],
                'lap_times': self._calculate_stats(group_df, 'lap_time'),
                'sector_1_times': self._calculate_stats(group_df, 's1_time'),
                'sector_2_times': self._calculate_stats(group_df, 's2_time'),
                'sector_3_times': self._calculate_stats(group_df, 's3_time')
            })
        print(f"Calculated statistics for {len(self.driver_stats)} drivers.")

    def _calculate_dynamic_license_factors(self):
        """
        NEW: Calculates license adjustment factors based on the actual performance
        data from the session.
        """
        df = pd.DataFrame(self.driver_stats)
        # Extract the best 5 lap average for each driver to use as the pace metric
        df['pace'] = df['lap_times'].apply(lambda x: x.get('best_5_avg'))
        df = df.dropna(subset=['pace', 'vehicle', 'license'])

        if df.empty:
            print("Not enough data to calculate dynamic license factors.")
            return

        # --- Step 1: Create Global Fallback ---
        global_silver_pace = df[df['license'] == 'Silver']['pace'].mean()
        if pd.isna(global_silver_pace):
            print("Warning: No Silver drivers found in data. Cannot create dynamic weights.")
            return # Abort if no baseline is possible

        # --- Step 2: Calculate Vehicle-Specific Averages ---
        vehicle_license_pace = df.groupby(['vehicle', 'license'])['pace'].mean().unstack()

        # --- Step 3: Generate Factors for Each Vehicle ---
        for vehicle, row in vehicle_license_pace.iterrows():
            # Use vehicle-specific Silver pace, or the global fallback
            baseline = row.get('Silver', global_silver_pace)
            if pd.isna(baseline): baseline = global_silver_pace

            self.dynamic_license_factors[vehicle] = {}
            for license_class, avg_pace in row.items():
                if pd.notna(avg_pace):
                    deviation = (avg_pace - baseline) / baseline
                    # The factor is the inverse of the deviation
                    self.dynamic_license_factors[vehicle][license_class] = 1.0 - deviation
        
        print("Successfully generated dynamic license adjustment factors.")
        # print(json.dumps(self.dynamic_license_factors, indent=2)) # Uncomment for debugging

    def _add_stint_adjusted_scores(self):
        """Adds new stint-adjusted scores to each driver's stats."""
        for driver in self.driver_stats:
            for time_key in ['lap_times', 'sector_1_times', 'sector_2_times', 'sector_3_times']:
                stats = driver[time_key]
                if not stats or 'fastest' not in stats: continue
                stats['stint_adjusted_fastest'] = stats['fastest'] + (stats.get('fastest_stint_lap', 0) * self.STINT_LAP_PENALTY)
                stats['stint_adjusted_best_3_avg'] = stats['best_3_avg'] + (stats.get('avg_stint_lap_for_best_3', 0) * self.STINT_LAP_PENALTY)
                stats['stint_adjusted_best_5_avg'] = stats['best_5_avg'] + (stats.get('avg_stint_lap_for_best_5', 0) * self.STINT_LAP_PENALTY)
        print("Generated new stint-adjusted scores.")

    def _add_license_adjusted_scores(self):
        """Adds new license-adjusted scores using the dynamic factors."""
        if not self.dynamic_license_factors:
            print("Skipping license adjustment; no dynamic factors were calculated.")
            return

        for driver in self.driver_stats:
            vehicle = driver['vehicle']
            license = driver['license']
            # Get the dynamic factor for this driver's vehicle and license
            factor = self.dynamic_license_factors.get(vehicle, {}).get(license, 1.0)

            for time_key in ['lap_times', 'sector_1_times', 'sector_2_times', 'sector_3_times']:
                stats = driver[time_key]
                if not stats or 'fastest' not in stats: continue
                stats['license_adjusted_fastest'] = stats['fastest'] * factor
                stats['license_adjusted_best_3_avg'] = stats['best_3_avg'] * factor
                stats['license_adjusted_best_5_avg'] = stats['best_5_avg'] * factor
        print("Generated new license-adjusted scores using dynamic factors.")

    def _generate_rankings(self):
        """Generates a comprehensive set of rankings for all calculated metrics."""
        def create_ranking(stats, time_key, metric_key):
            ranked_list = [
                {'driver_name': d['driver_name'], 'license': d['license'], 'vehicle': d['vehicle'], 'value': d.get(time_key, {}).get(metric_key)}
                for d in stats if d.get(time_key, {}).get(metric_key) is not None
            ]
            return sorted(ranked_list, key=lambda x: x['value'])

        def generate_rankings_for_group(stats):
            rankings = {}
            time_metrics = {'lap': 'lap_times', 's1': 'sector_1_times', 's2': 'sector_2_times', 's3': 'sector_3_times'}
            
            metrics_to_rank = [
                'fastest', 'best_3_avg', 'best_5_avg', 'deviation_3_lap', 'deviation_5_lap',
                'stint_adjusted_fastest', 'stint_adjusted_best_3_avg', 'stint_adjusted_best_5_avg',
                'license_adjusted_fastest', 'license_adjusted_best_3_avg', 'license_adjusted_best_5_avg',
                'fastest_stint_lap', 'avg_stint_lap_for_best_3', 'avg_stint_lap_for_best_5'
            ]

            for metric in metrics_to_rank:
                for name, key in time_metrics.items():
                    rankings[f'by_{metric}_{name}'] = create_ranking(stats, key, metric)
            return rankings

        license_groups = pd.DataFrame(self.driver_stats).groupby('license')
        vehicle_groups = pd.DataFrame(self.driver_stats).groupby('vehicle')
        return {
            'overall': generate_rankings_for_group(self.driver_stats),
            'by_license': {name: generate_rankings_for_group(g.to_dict('records')) for name, g in license_groups},
            'by_vehicle': {name: generate_rankings_for_group(g.to_dict('records')) for name, g in vehicle_groups}
        }

    def generate_report(self):
        """Executes the full workflow to generate the final report."""
        self._load_data()
        laps_df = self._process_data()
        self._calculate_all_driver_stats(laps_df)
        self._calculate_dynamic_license_factors() # New step
        self._add_stint_adjusted_scores()
        self._add_license_adjusted_scores()
        self.final_report = {
            'driver_performance': self.driver_stats,
            'rankings': self._generate_rankings(),
            'dynamic_license_factors_used': self.dynamic_license_factors # Added for transparency
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
    INPUT_FILE = "test_2025_data.json"
    OUTPUT_FILE = "race_report_output_v11.json"
    try:
        report_generator = RaceReportGenerator(INPUT_FILE)
        report_generator.generate_report()
        report_generator.save_report(OUTPUT_FILE)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")