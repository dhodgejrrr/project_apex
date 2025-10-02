import json
import pandas as pd
import numpy as np

class RaceReportGenerator:
    """
    Processes race timing data to generate a report with a fully dynamic handicap
    system where the strength of the adjustment is derived from the performance
    overlap between license classes.
    """
    STINT_LAP_PENALTY = 0.0002
    COMPETITIVE_PACE_PERCENTILE = 0.60
    # --- NEW: Safe fallback value if the data is insufficient to derive a strength ---
    DEFAULT_HANDICAP_STRENGTH = 0.5

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.driver_stats = []
        self.final_report = {}
        self.dynamic_license_factors = {}
        self.handicap_strength_used = self.DEFAULT_HANDICAP_STRENGTH

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
    
    def _calculate_handicap_strength(self, df):
        """
        NEW: Calculates handicap strength based on the performance overlap
        between Bronze and Silver drivers.
        """
        silver_paces = df.loc[df['license'] == 'Silver', 'pace'].dropna()
        bronze_paces = df.loc[df['license'] == 'Bronze', 'pace'].dropna()

        # Require a minimum number of drivers in each class for a reliable calculation
        if len(silver_paces) < 5 or len(bronze_paces) < 5:
            print(f"Insufficient driver data to derive handicap strength. Using default: {self.DEFAULT_HANDICAP_STRENGTH*100}%")
            return self.DEFAULT_HANDICAP_STRENGTH

        # Get the pace range of the "competitive pack" for each class
        silver_25th, silver_75th = silver_paces.quantile(0.25), silver_paces.quantile(0.75)
        bronze_25th = bronze_paces.quantile(0.25)
        
        silver_iqr = silver_75th - silver_25th
        if silver_iqr == 0: # Avoid division by zero
            return self.DEFAULT_HANDICAP_STRENGTH

        # Measure the gap between the slowest competitive Silvers and fastest competitive Bronzes
        separation_gap = silver_75th - bronze_25th
        
        # Normalize this gap by the spread of the Silver class itself
        # This score indicates how "separate" the classes are.
        separation_score = separation_gap / silver_iqr
        
        # Clamp the score to a reasonable range (e.g., 20% to 90%) to create the strength
        # A higher separation score means a stronger, more confident handicap.
        strength = np.clip(separation_score, 0.2, 0.9)
        
        print(f"Class Separation Score: {separation_score:.2f}. Derived Handicap Strength: {strength:.2f}")
        return strength

    def _calculate_dynamic_license_factors(self):
        """
        Calculates license factors using a dynamically calculated handicap strength.
        """
        df = pd.DataFrame(self.driver_stats)
        df['pace'] = df['lap_times'].apply(lambda x: x.get('best_5_avg'))
        df = df.dropna(subset=['pace', 'license'])

        valid_licenses = ['Bronze', 'Silver', 'Gold', 'Platinum']
        df = df[df['license'].isin(valid_licenses)]

        if df.empty:
            print("Not enough data to calculate dynamic license factors.")
            return

        # --- DYNAMIC STRENGTH CALCULATION ---
        self.handicap_strength_used = self._calculate_handicap_strength(df)

        competitive_paces = {}
        for license_class in df['license'].unique():
            class_df = df[df['license'] == license_class]
            cutoff_pace = class_df['pace'].quantile(self.COMPETITIVE_PACE_PERCENTILE)
            competitive_group = class_df[class_df['pace'] <= cutoff_pace]
            
            if not competitive_group.empty:
                competitive_paces[license_class] = competitive_group['pace'].mean()

        if 'Silver' not in competitive_paces:
            print("Warning: No competitive Silver drivers found. Cannot create dynamic weights.")
            return
        global_silver_pace = competitive_paces['Silver']

        for license_class, avg_pace in competitive_paces.items():
            deviation = (avg_pace - global_silver_pace) / global_silver_pace
            adjusted_deviation = deviation * self.handicap_strength_used
            self.dynamic_license_factors[license_class] = 1.0 - adjusted_deviation
        
        print(f"\nSuccessfully generated {self.handicap_strength_used*100:.0f}% strength license adjustment factors.")
        print(json.dumps(self.dynamic_license_factors, indent=2))

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
        """Adds new license-adjusted scores using the global dynamic factors."""
        if not self.dynamic_license_factors:
            print("Skipping license adjustment; no dynamic factors were calculated.")
            return

        for driver in self.driver_stats:
            license = driver['license']
            factor = self.dynamic_license_factors.get(license, 1.0)

            for time_key in ['lap_times', 'sector_1_times', 'sector_2_times', 'sector_3_times']:
                stats = driver[time_key]
                if not stats or 'fastest' not in stats: continue
                stats['license_adjusted_fastest'] = stats['fastest'] * factor
                stats['license_adjusted_best_3_avg'] = stats['best_3_avg'] * factor
                stats['license_adjusted_best_5_avg'] = stats['best_5_avg'] * factor
        print("Generated new license-adjusted scores using global dynamic factors.")

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
        self._calculate_dynamic_license_factors()
        self._add_stint_adjusted_scores()
        self._add_license_adjusted_scores()
        self.final_report = {
            'driver_performance': self.driver_stats,
            'rankings': self._generate_rankings(),
            'dynamic_license_factors_used': self.dynamic_license_factors,
            'handicap_strength_used': self.handicap_strength_used
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
    OUTPUT_FILE = "race_report_output_v18.json"
    try:
        report_generator = RaceReportGenerator(INPUT_FILE)
        report_generator.generate_report()
        report_generator.save_report(OUTPUT_FILE)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")