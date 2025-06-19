import pandas as pd
import json
import numpy as np
import warnings

# Optional: Suppress the FutureWarning about downcasting if it's too noisy
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class IMSADataAnalyzer:
    """
    This class combines the original user-defined race analysis methods with
    new, enhanced strategic analysis capabilities, including traffic-based lap
    categorization and a predictive polynomial tire degradation model.
    """

    # <<< MODIFIED __init__ >>>
    def __init__(self, csv_filepath, pit_json_filepath=None, fuel_capacity_json_filepath=None, config=None):
        """
        Initializes the analyzer with the data and configuration.
        """
        print(f"Initializing IMSADataAnalyzer with CSV: {csv_filepath}")
        
        default_config = {'fuel_burn_rate_kg_per_lap': 2.2, 'fuel_weight_penalty_s_per_kg': 0.035, 'max_fuel_load_kg': 85.0, 'pit_lane_delta_s': 45.0, 'traffic_proximity_threshold_s': 2.0, 'traffic_compromise_threshold_s': 1.5, 'driver_potential_percentile': 0.05, 'min_laps_for_deg_model': 5, 'min_laps_for_metronome': 5, 'min_laps_for_metronome_longer': 10, 'min_laps_for_manu_showdown': 15}
        self.config = default_config
        if config: self.config.update(config)

        # <<< NEW: Load manufacturer-specific fuel capacities >>>
        self.fuel_capacities = None
        if fuel_capacity_json_filepath:
            try:
                with open(fuel_capacity_json_filepath, 'r') as f:
                    # Load and normalize keys to be lowercase for case-insensitive matching
                    loaded_caps = json.load(f)
                    self.fuel_capacities = {k.lower().strip(): v for k, v in loaded_caps.items()}
                print(f"Loaded manufacturer fuel capacities from {fuel_capacity_json_filepath}.")
            except Exception as e:
                print(f"WARNING: Failed to load fuel capacity JSON {fuel_capacity_json_filepath}: {e}. Will use default value.")
        # <<< END NEW >>>

        self.pit_data_df = None
        if pit_json_filepath:
            try:
                self.pit_data_df = self._load_pit_data(pit_json_filepath); print(f"Loaded pit JSON data from {pit_json_filepath}.")
            except Exception as e:
                print(f"WARNING: Failed to load pit JSON {pit_json_filepath}: {e}. Will use CSV-derived pit data if necessary.")
        
        try:
            self.df = pd.read_csv(csv_filepath, sep=';'); self.df.columns = self.df.columns.str.strip(); print(f"CSV loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError: raise FileNotFoundError(f"Error: The file {csv_filepath} was not found.")
        if self.df.empty: raise ValueError("CSV file is empty or not parsed correctly.")
        self._preprocess_data()

    def _parse_time_to_seconds(self, time_str):
        if pd.isna(time_str) or not isinstance(time_str, str) or time_str.strip() == "": return np.nan
        time_str = str(time_str).strip(); parts = time_str.split(':')
        try:
            if len(parts) == 1: return float(parts[0])
            elif len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except (ValueError, IndexError): return np.nan
        return np.nan

    def _format_seconds_to_ms_str(self, total_seconds):
        if pd.isna(total_seconds) or not isinstance(total_seconds, (int, float)): return None
        sign = "";
        if total_seconds < 0: sign = "-"; total_seconds = abs(total_seconds)
        minutes = int(total_seconds // 60); seconds_part = total_seconds % 60
        if minutes > 0: return f"{sign}{minutes}:{seconds_part:06.3f}"
        return f"{sign}{total_seconds:.3f}"
        
    # <<< MODIFIED _preprocess_data >>>
    def _preprocess_data(self):
        print("\n--- Starting Master Preprocessing ---"); df = self.df; time_cols = ['LAP_TIME', 'S1', 'S2', 'S3', 'PIT_TIME']
        for col in time_cols:
            if col in df.columns: df[col + '_SEC'] = df[col].apply(self._parse_time_to_seconds)
        if 'HOUR' in df.columns: df['FINISH_TIME_DT'] = pd.to_datetime(df['HOUR'], format='%H:%M:%S.%f', errors='coerce')
        else: df['FINISH_TIME_DT'] = pd.NaT
        num_cols = ['LAP_NUMBER', 'KPH']; str_cols = ['NUMBER', 'DRIVER_NUMBER', 'DRIVER_NAME', 'TEAM', 'MANUFACTURER', 'FLAG_AT_FL', 'CROSSING_FINISH_LINE_IN_PIT', 'CLASS', 'GROUP']
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in str_cols:
            if col in df.columns: df[col] = df[col].fillna('').astype(str).str.strip()
        df = df.sort_values(by=['NUMBER', 'LAP_NUMBER']).copy()
        df['is_pit_stop_lap'] = df['CROSSING_FINISH_LINE_IN_PIT'] == 'B'; df['is_stint_start'] = df.groupby('NUMBER')['is_pit_stop_lap'].shift(1).fillna(True)
        df['stint_id_num'] = df.groupby('NUMBER')['is_stint_start'].cumsum(); df['stint_id'] = df['NUMBER'] + "_S" + df['stint_id_num'].astype(str)
        df['lap_in_stint'] = df.groupby('stint_id').cumcount() + 1
        
        # <<< NEW: Use manufacturer-specific fuel loads if available >>>
        if self.fuel_capacities:
            print("Applying manufacturer-specific fuel capacities.")
            # Map manufacturer to its fuel capacity, use default if not found
            df['max_fuel_load'] = df['MANUFACTURER'].str.lower().str.strip().map(self.fuel_capacities)
            df['max_fuel_load'].fillna(self.config['max_fuel_load_kg'], inplace=True)
        else:
            print("Using default fuel capacity for all cars.")
            df['max_fuel_load'] = self.config['max_fuel_load_kg']

        df['fuel_load_kg'] = df['max_fuel_load'] - ((df['lap_in_stint'] - 1) * self.config['fuel_burn_rate_kg_per_lap'])
        # <<< END NEW >>>

        df['fuel_correction_s'] = df['fuel_load_kg'] * self.config['fuel_weight_penalty_s_per_kg']; df['LAP_TIME_FUEL_CORRECTED_SEC'] = df['LAP_TIME_SEC'] + df['fuel_correction_s']
        self.df = df; print("Categorizing laps for traffic..."); self._categorize_laps_for_traffic(); print("--- Master Preprocessing Finished ---")
        
    def _categorize_laps_for_traffic(self):
        if 'FINISH_TIME_DT' not in self.df.columns or self.df['FINISH_TIME_DT'].isna().all():
            print("WARNING: 'HOUR' column missing or invalid. Skipping traffic analysis."); self.df['LAP_CATEGORY'] = 'UNKNOWN'; return
        
        clean_laps = self.df[(self.df['FLAG_AT_FL'] == 'GF') & (~self.df['is_pit_stop_lap']) & (self.df['lap_in_stint'] > 1) & (self.df['LAP_TIME_SEC'].notna())].copy()
        potentials = clean_laps.groupby('DRIVER_NAME')['LAP_TIME_SEC'].quantile(self.config['driver_potential_percentile']).to_dict()
        self.df['DRIVER_POTENTIAL_LAP_SEC'] = self.df['DRIVER_NAME'].map(potentials)
        self.df['is_compromised'] = self.df['LAP_TIME_SEC'] > (self.df['DRIVER_POTENTIAL_LAP_SEC'] + self.config['traffic_compromise_threshold_s'])
        
        df_sorted = self.df.sort_values(by='FINISH_TIME_DT').copy()
        df_sorted['time_delta_to_prev_car_s'] = (df_sorted['FINISH_TIME_DT'] - df_sorted['FINISH_TIME_DT'].shift(1)).dt.total_seconds()
        df_sorted['prev_car_class'] = df_sorted['CLASS'].shift(1)
        df_sorted['prev_car_number'] = df_sorted['NUMBER'].shift(1)
        
        conditions = [
            (df_sorted['is_compromised']) & (df_sorted['time_delta_to_prev_car_s'] <= self.config['traffic_proximity_threshold_s']) & (df_sorted['CLASS'] == df_sorted['prev_car_class']) & (df_sorted['NUMBER'] != df_sorted['prev_car_number']),
            (df_sorted['is_compromised']) & (df_sorted['time_delta_to_prev_car_s'] <= self.config['traffic_proximity_threshold_s']) & (df_sorted['CLASS'] != df_sorted['prev_car_class'])
        ]
        choices = ['TRAFFIC_IN_CLASS', 'TRAFFIC_OUT_OF_CLASS']
        df_sorted['LAP_CATEGORY'] = np.select(conditions, choices, default='NORMAL')
        
        self.df = self.df.merge(df_sorted[['NUMBER', 'LAP_NUMBER', 'LAP_CATEGORY']], on=['NUMBER', 'LAP_NUMBER'], how='left')

    def _get_row_at_min_time(self, df_group, time_col_sec, original_time_col_str=None, required_fields=None):
        if required_fields is None: required_fields = ['DRIVER_NAME', 'LAP_NUMBER', 'TEAM', 'NUMBER']
        default_data = {time_col_sec: np.nan, **{f: None for f in required_fields}}
        if original_time_col_str: default_data[original_time_col_str] = None
        valid_df = df_group.dropna(subset=[time_col_sec])
        if valid_df.empty: return pd.Series({**{c: None for c in df_group.columns}, **default_data})
        return df_group.loc[valid_df[time_col_sec].idxmin()]

    def _load_pit_data(self, pit_json_filepath):
        with open(pit_json_filepath, 'r') as fp: pit_json = json.load(fp)
        pit_entries = []
        for car_entry in pit_json.get('pit_stop_analysis', []):
            car_no = str(car_entry.get('number', '')).strip()
            for stop in car_entry.get('pit_stops', []):
                in_dt = pd.to_datetime(stop.get('in_time'), format='%H:%M:%S.%f', errors='coerce'); out_dt = pd.to_datetime(stop.get('out_time'), format='%H:%M:%S.%f', errors='coerce')
                pit_entries.append({'NUMBER': car_no, 'pit_number': stop.get('pit_number'), 'pit_time_sec': self._parse_time_to_seconds(stop.get('pit_time')), 'in_time_str': stop.get('in_time'), 'out_time_str': stop.get('out_time'), 'in_dt': in_dt, 'out_dt': out_dt, 'driver_change': stop.get('in_driver_number') != stop.get('out_driver_number') if stop.get('in_driver_number') and stop.get('out_driver_number') else np.nan})
        if not pit_entries: raise ValueError("Pit JSON file contained no pit stop information.")
        return pd.DataFrame(pit_entries)
        
    def _get_json_pit_data_for_car(self, car_number):
        if self.pit_data_df is None or self.pit_data_df.empty: return pd.DataFrame()
        return self.pit_data_df[self.pit_data_df['NUMBER'] == str(car_number)].copy()

    def _analyze_pit_stops_json(self, car_df):
        car_no = car_df['NUMBER'].iloc[0]; pit_df = self._get_json_pit_data_for_car(car_no)
        if pit_df.empty: return self._analyze_pit_stops_original(car_df)
        total_time = pit_df['pit_time_sec'].sum(); num_stops = len(pit_df)
        total_minus_travel = pit_df['pit_time_sec'].apply(lambda x: max(0, x - self.config['pit_lane_delta_s'])).sum()
        return {'total_pit_stops': num_stops, 'total_pit_time_sec': total_time, 'total_pit_time_minus_travel_sec': total_minus_travel, 'total_pit_time_formatted': self._format_seconds_to_ms_str(total_time), 'total_pit_time_minus_travel_formatted': self._format_seconds_to_ms_str(total_minus_travel), 'average_pit_time_formatted': self._format_seconds_to_ms_str(total_time / num_stops if num_stops > 0 else np.nan)}

    def _get_pit_stop_details_json(self, car_df, driver_changes):
        car_no = car_df['NUMBER'].iloc[0]; pit_df = self._get_json_pit_data_for_car(car_no)
        if pit_df.empty: return self._get_pit_stop_details_original(car_df, driver_changes)
        change_laps = {item['lap_number'] for item in driver_changes.get('change_details', [])}; details = []
        for _, row in pit_df.sort_values('pit_number').iterrows():
            lap_entry = None
            if pd.notna(row['in_dt']) and car_df['FINISH_TIME_DT'].notna().any():
                prev_laps = car_df[car_df['FINISH_TIME_DT'] <= row['in_dt']]
                if not prev_laps.empty: lap_entry = int(prev_laps['LAP_NUMBER'].iloc[-1])
                else:
                    next_laps = car_df[car_df['FINISH_TIME_DT'] > row['in_dt']]
                    if not next_laps.empty: lap_entry = int(next_laps['LAP_NUMBER'].iloc[0])
            stationary = row['pit_time_sec'] - self.config['pit_lane_delta_s'] if pd.notna(row['pit_time_sec']) else np.nan
            details.append({'stop_number': int(row['pit_number']) if pd.notna(row['pit_number']) else None, 'lap_number_entry': lap_entry, 'total_pit_lane_time': self._format_seconds_to_ms_str(row['pit_time_sec']), 'stationary_time': self._format_seconds_to_ms_str(stationary), 'driver_change': bool(row['driver_change']) if 'driver_change' in row and pd.notna(row['driver_change']) else (lap_entry is not None and (lap_entry in change_laps or (lap_entry + 1) in change_laps))})
        return details

    def get_fastest_by_car_number(self):
        results = [];
        for car_no, group in self.df.groupby('NUMBER'):
            fastest_lap_row = self._get_row_at_min_time(group, 'LAP_TIME_SEC', 'LAP_TIME');
            if pd.isna(fastest_lap_row['LAP_TIME_SEC']): continue 
            best_s1_row = self._get_row_at_min_time(group, 'S1_SEC', 'S1'); best_s2_row = self._get_row_at_min_time(group, 'S2_SEC', 'S2'); best_s3_row = self._get_row_at_min_time(group, 'S3_SEC', 'S3')
            optimal_lap_time_sec = np.nan
            if pd.notna(best_s1_row.get('S1_SEC')) and pd.notna(best_s2_row.get('S2_SEC')) and pd.notna(best_s3_row.get('S3_SEC')):
                optimal_lap_time_sec = best_s1_row['S1_SEC'] + best_s2_row['S2_SEC'] + best_s3_row['S3_SEC']
            results.append({"car_number": car_no, "fastest_lap": {"time": fastest_lap_row.get('LAP_TIME'), "driver_name": fastest_lap_row.get('DRIVER_NAME'), "lap_number": fastest_lap_row.get('LAP_NUMBER')}, "best_s1": {"time": best_s1_row.get('S1'), "driver_name": best_s1_row.get('DRIVER_NAME'), "lap_number": best_s1_row.get('LAP_NUMBER')}, "best_s2": {"time": best_s2_row.get('S2'), "driver_name": best_s2_row.get('DRIVER_NAME'), "lap_number": best_s2_row.get('LAP_NUMBER')}, "best_s3": {"time": best_s3_row.get('S3'), "driver_name": best_s3_row.get('DRIVER_NAME'), "lap_number": best_s3_row.get('LAP_NUMBER')}, "optimal_lap_time": self._format_seconds_to_ms_str(optimal_lap_time_sec)})
        return results

    def get_fastest_by_manufacturer(self):
        results = [];
        for manufacturer, group in self.df.groupby('MANUFACTURER'):
            if not manufacturer: continue
            fastest_lap_row = self._get_row_at_min_time(group, 'LAP_TIME_SEC', 'LAP_TIME');
            if pd.isna(fastest_lap_row['LAP_TIME_SEC']): continue
            best_s1_row = self._get_row_at_min_time(group, 'S1_SEC', 'S1'); best_s2_row = self._get_row_at_min_time(group, 'S2_SEC', 'S2'); best_s3_row = self._get_row_at_min_time(group, 'S3_SEC', 'S3')
            optimal_lap_time_sec = np.nan
            if pd.notna(best_s1_row.get('S1_SEC')) and pd.notna(best_s2_row.get('S2_SEC')) and pd.notna(best_s3_row.get('S3_SEC')):
                optimal_lap_time_sec = best_s1_row['S1_SEC'] + best_s2_row['S2_SEC'] + best_s3_row['S3_SEC']
            results.append({"manufacturer": manufacturer, "fastest_lap": {"time": fastest_lap_row.get('LAP_TIME'), "driver_name": fastest_lap_row.get('DRIVER_NAME'), "team": fastest_lap_row.get('TEAM'), "car_number": fastest_lap_row.get('NUMBER'), "lap_number": fastest_lap_row.get('LAP_NUMBER')}, "best_s1": {"time": best_s1_row.get('S1'), "driver_name": best_s1_row.get('DRIVER_NAME'), "team": best_s1_row.get('TEAM'), "car_number": best_s1_row.get('NUMBER'), "lap_number": best_s1_row.get('LAP_NUMBER')}, "best_s2": {"time": best_s2_row.get('S2'), "driver_name": best_s2_row.get('DRIVER_NAME'), "team": best_s2_row.get('TEAM'), "car_number": best_s2_row.get('NUMBER'), "lap_number": best_s2_row.get('LAP_NUMBER')}, "best_s3": {"time": best_s3_row.get('S3'), "driver_name": best_s3_row.get('DRIVER_NAME'), "team": best_s3_row.get('TEAM'), "car_number": best_s3_row.get('NUMBER'), "lap_number": best_s3_row.get('LAP_NUMBER')}, "optimal_lap_time": self._format_seconds_to_ms_str(optimal_lap_time_sec)})
        return results

    def get_longest_stints_by_manufacturer(self):
        results = []; stint_df = self.df
        if 'stint_id' not in stint_df.columns or stint_df['stint_id'].isna().all(): return results
        stints_data = []
        for stint_id_val, laps_this_stint_df in stint_df.groupby('stint_id'):
            if laps_this_stint_df.empty: continue
            racing_laps_for_stint_df = laps_this_stint_df.copy()
            if laps_this_stint_df.iloc[-1]['is_pit_stop_lap']: racing_laps_for_stint_df = laps_this_stint_df.iloc[:-1] if len(laps_this_stint_df) > 1 else pd.DataFrame(columns=laps_this_stint_df.columns) 
            if racing_laps_for_stint_df.empty: continue
            valid_green_racing_laps_df = racing_laps_for_stint_df[(racing_laps_for_stint_df['FLAG_AT_FL'] == 'GF') & (racing_laps_for_stint_df['LAP_TIME_SEC'].notna())]
            if len(valid_green_racing_laps_df) == len(racing_laps_for_stint_df) and not valid_green_racing_laps_df.empty:
                stints_data.append({'stint_id': stint_id_val, 'manufacturer': valid_green_racing_laps_df['MANUFACTURER'].iloc[0], 'num_green_laps': len(valid_green_racing_laps_df), 'laps_data_for_metrics': valid_green_racing_laps_df})
        if not stints_data: return results
        all_valid_green_stints_df = pd.DataFrame(stints_data)
        for manufacturer_name, manu_stints_df in all_valid_green_stints_df.groupby('manufacturer'): 
            if manu_stints_df.empty or manufacturer_name == "": continue
            longest_stint_len = manu_stints_df['num_green_laps'].max()
            if longest_stint_len == 0 or pd.isna(longest_stint_len): continue
            candidate_stints_for_manu_df = manu_stints_df[manu_stints_df['num_green_laps'] == longest_stint_len]
            chosen_stint_laps_df, fastest_lap_in_chosen_stint_sec = None, float('inf')
            for _, candidate_row in candidate_stints_for_manu_df.iterrows():
                current_stint_green_laps = candidate_row['laps_data_for_metrics']
                if current_stint_green_laps.empty: continue
                min_lap_time_sec_this_stint = current_stint_green_laps['LAP_TIME_SEC'].min() 
                if pd.notna(min_lap_time_sec_this_stint) and min_lap_time_sec_this_stint < fastest_lap_in_chosen_stint_sec:
                    fastest_lap_in_chosen_stint_sec = min_lap_time_sec_this_stint; chosen_stint_laps_df = current_stint_green_laps.copy() 
            if chosen_stint_laps_df is None or chosen_stint_laps_df.empty: continue
            best_lap_row_stint = self._get_row_at_min_time(chosen_stint_laps_df, 'LAP_TIME_SEC', 'LAP_TIME')
            results.append({"manufacturer": manufacturer_name, "longest_green_stint_laps": len(chosen_stint_laps_df), "stint_details": {"car_number": best_lap_row_stint.get('NUMBER'), "driver_at_best_lap": best_lap_row_stint.get('DRIVER_NAME'), "stint_id_debug": chosen_stint_laps_df['stint_id'].iloc[0], "start_lap_number_race": int(chosen_stint_laps_df['LAP_NUMBER'].min()), "end_lap_number_race": int(chosen_stint_laps_df['LAP_NUMBER'].max()), "best_lap_time": best_lap_row_stint.get('LAP_TIME'), "best_lap_position_in_stint": int(best_lap_row_stint.get('lap_in_stint')), "best_s1_in_stint": self._get_row_at_min_time(chosen_stint_laps_df, 'S1_SEC')['S1'], "best_s2_in_stint": self._get_row_at_min_time(chosen_stint_laps_df, 'S2_SEC')['S2'], "best_s3_in_stint": self._get_row_at_min_time(chosen_stint_laps_df, 'S3_SEC')['S3'], "average_lap_time_in_stint": self._format_seconds_to_ms_str(chosen_stint_laps_df['LAP_TIME_SEC'].mean())}})
        return results

    def get_driver_deltas_by_car(self):
        results = [];
        for car_no, car_laps_df in self.df.groupby('NUMBER'):
            driver_performances = []
            for driver_num_val, driver_laps_df in car_laps_df.groupby('DRIVER_NUMBER'):
                if not driver_num_val or pd.isna(driver_num_val) or str(driver_num_val).lower() == 'nan' or driver_laps_df.empty: continue
                driver_name = driver_laps_df['DRIVER_NAME'].mode()[0] if not driver_laps_df['DRIVER_NAME'].mode().empty else f"Driver_{driver_num_val}"
                best_lap_row = self._get_row_at_min_time(driver_laps_df, 'LAP_TIME_SEC', 'LAP_TIME')
                if pd.isna(best_lap_row['LAP_TIME_SEC']): continue 
                driver_performances.append({"driver_name": driver_name, "driver_number": driver_num_val, "best_lap_time": best_lap_row.get('LAP_TIME'), "best_lap_time_sec": best_lap_row.get('LAP_TIME_SEC'), "best_s1": self._get_row_at_min_time(driver_laps_df, 'S1_SEC', 'S1').get('S1'), "best_s1_sec": self._get_row_at_min_time(driver_laps_df, 'S1_SEC', 'S1').get('S1_SEC'), "best_s2": self._get_row_at_min_time(driver_laps_df, 'S2_SEC', 'S2').get('S2'), "best_s2_sec": self._get_row_at_min_time(driver_laps_df, 'S2_SEC', 'S2').get('S2_SEC'), "best_s3": self._get_row_at_min_time(driver_laps_df, 'S3_SEC', 'S3').get('S3'), "best_s3_sec": self._get_row_at_min_time(driver_laps_df, 'S3_SEC', 'S3').get('S3_SEC')})
            if len(driver_performances) < 2: continue
            driver_performances.sort(key=lambda x: x.get('best_lap_time_sec', float('inf')))
            fastest_driver_perf = driver_performances[0]; deltas_to_fastest, lap_time_deltas_for_avg_sec = [], []
            for other_driver_perf in driver_performances[1:]:
                lap_delta = other_driver_perf['best_lap_time_sec'] - fastest_driver_perf['best_lap_time_sec']
                s1_delta = other_driver_perf.get('best_s1_sec', np.nan) - fastest_driver_perf.get('best_s1_sec', np.nan); s2_delta = other_driver_perf.get('best_s2_sec', np.nan) - fastest_driver_perf.get('best_s2_sec', np.nan); s3_delta = other_driver_perf.get('best_s3_sec', np.nan) - fastest_driver_perf.get('best_s3_sec', np.nan)
                deltas_to_fastest.append({"driver_name": other_driver_perf['driver_name'], "lap_time_delta": self._format_seconds_to_ms_str(lap_delta), "s1_delta": self._format_seconds_to_ms_str(s1_delta), "s2_delta": self._format_seconds_to_ms_str(s2_delta), "s3_delta": self._format_seconds_to_ms_str(s3_delta)})
                lap_time_deltas_for_avg_sec.append(lap_delta)
            avg_lap_time_delta_sec = np.mean(lap_time_deltas_for_avg_sec) if lap_time_deltas_for_avg_sec else np.nan
            results.append({"car_number": car_no, "drivers_performance": driver_performances, "fastest_driver_name": fastest_driver_perf['driver_name'], "deltas_to_fastest": deltas_to_fastest, "average_lap_time_delta_for_car": self._format_seconds_to_ms_str(avg_lap_time_delta_sec)})
        return results

    def get_manufacturer_driver_pace_gap(self):
        lap_gaps, s1_gaps, s2_gaps, s3_gaps = [], [], [], []
        for manu_name, manu_df in self.df.groupby('MANUFACTURER'):
            if not manu_name or manu_df.empty or manu_df['DRIVER_NAME'].nunique() < 2:
                continue
            driver_bests = []
            for driver_name, driver_df in manu_df.groupby('DRIVER_NAME'):
                if not driver_name or driver_df.empty: continue
                best_lap_row = self._get_row_at_min_time(driver_df, 'LAP_TIME_SEC', 'LAP_TIME')
                best_s1_row = self._get_row_at_min_time(driver_df, 'S1_SEC', 'S1')
                best_s2_row = self._get_row_at_min_time(driver_df, 'S2_SEC', 'S2')
                best_s3_row = self._get_row_at_min_time(driver_df, 'S3_SEC', 'S3')
                driver_bests.append({'driver_name': driver_name, 'lap_time_sec': best_lap_row['LAP_TIME_SEC'], 'LAP_TIME': best_lap_row['LAP_TIME'], 's1_sec': best_s1_row['S1_SEC'], 'S1': best_s1_row['S1'], 's2_sec': best_s2_row['S2_SEC'], 'S2': best_s2_row['S2'], 's3_sec': best_s3_row['S3_SEC'], 'S3': best_s3_row['S3']})
            driver_bests_df = pd.DataFrame(driver_bests)
            if (lap_result := self._process_metric_gap(manu_name, driver_bests_df, 'lap_time_sec', 'LAP_TIME')): lap_gaps.append(lap_result)
            if (s1_result := self._process_metric_gap(manu_name, driver_bests_df, 's1_sec', 'S1')): s1_gaps.append(s1_result)
            if (s2_result := self._process_metric_gap(manu_name, driver_bests_df, 's2_sec', 'S2')): s2_gaps.append(s2_result)
            if (s3_result := self._process_metric_gap(manu_name, driver_bests_df, 's3_sec', 'S3')): s3_gaps.append(s3_result)
        lap_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        s1_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        s2_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        s3_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        return {'lap_time_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(lap_gaps)], 's1_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(s1_gaps)], 's2_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(s2_gaps)], 's3_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(s3_gaps)]}
        
    def get_race_strategy_by_car(self):
        results = []
        for car_no, car_df in self.df.groupby('NUMBER'):
            if car_df.empty: continue
            driver_changes = self._analyze_driver_changes_original(car_df)
            if self.pit_data_df is not None and not self._get_json_pit_data_for_car(car_no).empty:
                pit_analysis = self._analyze_pit_stops_json(car_df)
                pit_stop_details = self._get_pit_stop_details_json(car_df, driver_changes)
            else:
                pit_analysis = self._analyze_pit_stops_original(car_df)
                pit_stop_details = self._get_pit_stop_details_original(car_df, driver_changes)
            stint_analysis = self._analyze_stints_original(car_df)
            results.append({"car_number": car_no, "total_pit_time": pit_analysis.get('total_pit_time_formatted'), "average_pit_time": pit_analysis.get('average_pit_time_formatted'), "total_pit_time_minus_travel": pit_analysis.get('total_pit_time_minus_travel_formatted'), "total_pit_stops": pit_analysis.get('total_pit_stops', 0), "total_driver_changes": driver_changes.get('total_driver_changes', 0), "driver_change_details": driver_changes.get('change_details', []), "stints": stint_analysis, "pit_stop_details": pit_stop_details})
        return results

    def _process_metric_gap(self, manu_name, perfs_df, metric_sec, metric_str):
        valid_perfs = perfs_df.dropna(subset=[metric_sec]).copy()
        if len(valid_perfs) < 2: return None
        fastest_idx, slowest_idx = valid_perfs[metric_sec].idxmin(), valid_perfs[metric_sec].idxmax()
        fastest_perf, slowest_perf = valid_perfs.loc[fastest_idx], valid_perfs.loc[slowest_idx]
        if fastest_perf['driver_name'] == slowest_perf['driver_name']: return None
        gap_sec = slowest_perf[metric_sec] - fastest_perf[metric_sec]
        if gap_sec <= 0: return None
        return {"manufacturer": manu_name, "gap_seconds": gap_sec, "gap_formatted": self._format_seconds_to_ms_str(gap_sec), "fastest_driver": {"name": fastest_perf['driver_name'], "time": fastest_perf[metric_str]}, "slowest_driver": {"name": slowest_perf['driver_name'], "time": slowest_perf[metric_str]}}
        
    def _get_pit_stop_details_original(self, car_df, driver_changes):
        pit_stop_list = []; driver_change_laps = {item['lap_number'] for item in driver_changes.get('change_details', [])}
        for stop_number, (_, stop_df) in enumerate(car_df[car_df['is_pit_stop_lap']].groupby('stint_id_num'), 1):
            entry_lap = int(stop_df['LAP_NUMBER'].min()); total_time_sec = stop_df['LAP_TIME_SEC'].sum()
            pit_stop_list.append({"stop_number": stop_number, "lap_number_entry": entry_lap, "total_pit_lane_time": self._format_seconds_to_ms_str(total_time_sec), "stationary_time": self._format_seconds_to_ms_str(total_time_sec - self.config['pit_lane_delta_s']), "driver_change": entry_lap in driver_change_laps})
        return pit_stop_list

    def _analyze_stints_original(self, car_df):
        stint_results = []
        for stint_num, stint_df in car_df.groupby('stint_id_num'):
            if stint_df.empty: continue
            racing_laps = stint_df[~stint_df['is_pit_stop_lap']];
            if racing_laps.empty: continue
            flag_stats = self._categorize_laps_by_flag(racing_laps); total_stint_time_sec = racing_laps['LAP_TIME_SEC'].sum() if racing_laps['LAP_TIME_SEC'].notna().any() else 0
            green_laps = racing_laps[racing_laps['FLAG_AT_FL'] == 'GF'].dropna(subset=['LAP_TIME_SEC']); best_5_lap_avg_sec = np.nan
            if len(green_laps) >= 5: best_5_lap_avg_sec = green_laps['LAP_TIME_SEC'].nsmallest(5).mean()
            traffic_counts = racing_laps['LAP_CATEGORY'].value_counts().to_dict()
            stint_results.append({'stint_number': int(stint_num), 'total_laps': len(racing_laps), 'total_time': self._format_seconds_to_ms_str(total_stint_time_sec), **flag_stats, 'lap_range': f"{int(racing_laps['LAP_NUMBER'].min())}-{int(racing_laps['LAP_NUMBER'].max())}", 'best_5_lap_avg': self._format_seconds_to_ms_str(best_5_lap_avg_sec), 'traffic_in_class_laps': traffic_counts.get('TRAFFIC_IN_CLASS', 0), 'traffic_out_of_class_laps': traffic_counts.get('TRAFFIC_OUT_OF_CLASS', 0)})
        return stint_results
    
    def _calculate_baseline_travel_time(self, car_df):
        green_laps = car_df[(car_df['FLAG_AT_FL'] == 'GF') & (car_df['LAP_TIME_SEC'].notna()) & (car_df['LAP_TIME_SEC'] > 0)]
        if green_laps.empty: return 90.0
        return green_laps['LAP_TIME_SEC'].quantile(0.1)

    def _analyze_pit_stops_original(self, car_df):
        baseline_travel_time_sec = self._calculate_baseline_travel_time(car_df); pit_analysis = {'total_pit_stops': 0, 'total_pit_time_sec': 0.0, 'total_pit_time_minus_travel_sec': 0.0}
        pit_sequences = car_df[car_df['is_pit_stop_lap']].groupby('stint_id_num'); pit_analysis['total_pit_stops'] = pit_sequences.ngroups
        for _, pit_seq_df in pit_sequences:
            pit_time_sec = pit_seq_df['LAP_TIME_SEC'].sum()
            if pd.notna(pit_time_sec) and pit_time_sec > 0:
                pit_analysis['total_pit_time_sec'] += pit_time_sec; pit_analysis['total_pit_time_minus_travel_sec'] += max(0, pit_time_sec - (baseline_travel_time_sec * len(pit_seq_df)))
        pit_analysis['total_pit_time_formatted'] = self._format_seconds_to_ms_str(pit_analysis['total_pit_time_sec']); pit_analysis['total_pit_time_minus_travel_formatted'] = self._format_seconds_to_ms_str(pit_analysis['total_pit_time_minus_travel_sec'])
        pit_analysis['average_pit_time_formatted'] = self._format_seconds_to_ms_str(pit_analysis['total_pit_time_sec'] / pit_analysis['total_pit_stops'] if pit_analysis['total_pit_stops'] > 0 else np.nan)
        return pit_analysis

    def _analyze_driver_changes_original(self, car_df):
        driver_analysis = {'total_driver_changes': 0, 'change_details': []}
        car_df_sorted = car_df.sort_values('LAP_NUMBER'); car_df_sorted['driver_changed'] = car_df_sorted['DRIVER_NUMBER'] != car_df_sorted['DRIVER_NUMBER'].shift(1)
        driver_changes = car_df_sorted[car_df_sorted['driver_changed']]
        if len(driver_changes) > 1:
            driver_analysis['total_driver_changes'] = len(driver_changes) - 1
            for _, change_row in driver_changes.iloc[1:].iterrows():
                prev_lap = car_df_sorted[car_df_sorted['LAP_NUMBER'] < change_row['LAP_NUMBER']]; prev_driver = prev_lap['DRIVER_NAME'].iloc[-1] if not prev_lap.empty else "Unknown"
                driver_analysis['change_details'].append({'lap_number': int(change_row['LAP_NUMBER']), 'from_driver': prev_driver, 'to_driver': change_row['DRIVER_NAME']})
        return driver_analysis

    def _categorize_laps_by_flag(self, laps_df):
        flag_stats = {'green_laps': 0, 'yellow_laps': 0, 'red_laps': 0, 'other_laps': 0}
        green_laps = laps_df[laps_df['FLAG_AT_FL'] == 'GF']; yellow_laps = laps_df[laps_df['FLAG_AT_FL'] == 'FCY']; red_laps = laps_df[laps_df['FLAG_AT_FL'] == 'RF']; other_laps = laps_df[~laps_df['FLAG_AT_FL'].isin(['GF', 'FCY', 'RF'])]
        flag_stats.update({'green_laps': len(green_laps), 'yellow_laps': len(yellow_laps), 'red_laps': len(red_laps), 'other_laps': len(other_laps)})
        if not green_laps.empty and green_laps['LAP_TIME_SEC'].notna().any(): flag_stats['avg_green_time_formatted'] = self._format_seconds_to_ms_str(green_laps['LAP_TIME_SEC'].mean()); flag_stats['best_green_time_formatted'] = self._format_seconds_to_ms_str(green_laps['LAP_TIME_SEC'].min())
        if not yellow_laps.empty and yellow_laps['LAP_TIME_SEC'].notna().any(): flag_stats['avg_yellow_time_formatted'] = self._format_seconds_to_ms_str(yellow_laps['LAP_TIME_SEC'].mean())
        return flag_stats

    def get_enhanced_strategy_analysis(self):
        results = [];
        for car_no, car_df in self.df.groupby('NUMBER'):
            if car_df.empty: continue
            clean_laps = car_df[(car_df['FLAG_AT_FL'] == 'GF') & (~car_df['is_pit_stop_lap']) & (car_df['LAP_CATEGORY'] == 'NORMAL')].dropna(subset=['LAP_TIME_FUEL_CORRECTED_SEC'])
            avg_green_pace_sec = clean_laps['LAP_TIME_FUEL_CORRECTED_SEC'].mean(); consistency = clean_laps['LAP_TIME_FUEL_CORRECTED_SEC'].std()
            pit_analysis = self._analyze_pit_stops(car_df) # This is a helper, not the main one
            degradation_model = self._calculate_advanced_degradation(car_df)
            results.append({"car_number": car_no, "team": car_df['TEAM'].mode()[0] if not car_df['TEAM'].mode().empty else "N/A", "manufacturer": car_df['MANUFACTURER'].mode()[0] if not car_df['MANUFACTURER'].mode().empty else "N/A", "avg_green_pace_fuel_corrected": self._format_seconds_to_ms_str(avg_green_pace_sec), "race_pace_consistency_stdev": round(consistency, 3) if pd.notna(consistency) else None, "avg_pit_stationary_time": self._format_seconds_to_ms_str(pit_analysis.get('average_stationary_time_sec')), "tire_degradation_model": degradation_model})
        return results

    def _calculate_advanced_degradation(self, car_df):
        stint_models, total_clean_laps_used = [], 0
        for _, stint_df in car_df.groupby('stint_id'):
            clean_laps = stint_df[(stint_df['FLAG_AT_FL'] == 'GF') & (~stint_df['is_pit_stop_lap']) & (stint_df['lap_in_stint'] > 1) & (stint_df['LAP_CATEGORY'] == 'NORMAL') & (stint_df['LAP_TIME_FUEL_CORRECTED_SEC'].notna())].copy()
            if len(clean_laps) >= self.config['min_laps_for_deg_model']:
                stint_models.append(np.polyfit(clean_laps['lap_in_stint'], clean_laps['LAP_TIME_FUEL_CORRECTED_SEC'], 2)); total_clean_laps_used += len(clean_laps)
        if not stint_models: return {"deg_coeff_a": None, "deg_coeff_b": None, "deg_coeff_c": None, "fastest_lap_of_stint_predicted_at": None, "model_quality": "INSUFFICIENT_DATA", "total_clean_laps_used": 0}
        avg_coeffs = np.mean(stint_models, axis=0); a, b, c = avg_coeffs[0], avg_coeffs[1], avg_coeffs[2]
        quality = "GOOD" if total_clean_laps_used > 20 else "FAIR" if total_clean_laps_used > 10 else "POOR"
        predicted_best_lap = -b / (2 * a) if a != 0 else np.nan
        return {"deg_coeff_a": round(a, 6), "deg_coeff_b": round(b, 6), "deg_coeff_c": round(c, 6), "fastest_lap_of_stint_predicted_at": round(predicted_best_lap, 1) if pd.notna(predicted_best_lap) else None, "model_quality": quality, "total_clean_laps_used": total_clean_laps_used}

    def _analyze_pit_stops(self, car_df):
        pit_stop_laps = car_df[car_df['is_pit_stop_lap']].copy()
        if pit_stop_laps.empty: return {'total_pit_stops': 0, 'average_stationary_time_sec': np.nan}
        num_stops = pit_stop_laps['stint_id_num'].nunique(); total_pit_time = pit_stop_laps['LAP_TIME_SEC'].sum()
        total_stationary = total_pit_time - (num_stops * self.config['pit_lane_delta_s'])
        return {'total_pit_stops': num_stops, 'average_stationary_time_sec': total_stationary / num_stops if num_stops > 0 else np.nan}

    def get_traffic_management_analysis(self):
        driver_results = []
        traffic_laps_df = self.df[self.df['LAP_CATEGORY'].isin(['TRAFFIC_IN_CLASS', 'TRAFFIC_OUT_OF_CLASS'])].copy()
        if traffic_laps_df.empty: return []
        traffic_laps_df['time_lost_sec'] = traffic_laps_df['LAP_TIME_SEC'] - traffic_laps_df['DRIVER_POTENTIAL_LAP_SEC']
        for driver_name, group in traffic_laps_df.groupby('DRIVER_NAME'):
            if group.empty or group['DRIVER_POTENTIAL_LAP_SEC'].isna().all(): continue
            in_class = group[group['LAP_CATEGORY'] == 'TRAFFIC_IN_CLASS']; out_of_class = group[group['LAP_CATEGORY'] == 'TRAFFIC_OUT_OF_CLASS']
            driver_results.append({'driver_name': driver_name, 'car_number': group['NUMBER'].iloc[0], 'team': group['TEAM'].iloc[0], 'avg_time_lost_total_sec': group['time_lost_sec'].mean(), 'avg_time_lost_in_class_sec': in_class['time_lost_sec'].mean() if not in_class.empty else np.nan, 'avg_time_lost_out_of_class_sec': out_of_class['time_lost_sec'].mean() if not out_of_class.empty else np.nan, 'total_traffic_laps': len(group), 'in_class_traffic_laps': len(in_class), 'out_of_class_traffic_laps': len(out_of_class)})
        driver_results.sort(key=lambda x: x.get('avg_time_lost_total_sec', float('inf')))
        return [{'rank': i + 1, **res} for i, res in enumerate(driver_results)]

    def get_full_pit_cycle_analysis(self, race_strategy_data, enhanced_strategy_data):
        car_results = []
        team_map = {car['car_number']: car.get('team', 'N/A') for car in enhanced_strategy_data}
        for car_data in race_strategy_data:
            car_number = car_data.get('car_number'); team_name = team_map.get(car_number, 'N/A'); all_losses = []
            for pit_stop in car_data.get('pit_stop_details', []):
                lap_entry = pit_stop.get('lap_number_entry')
                if lap_entry is None: continue
                stationary = max(0, self._parse_time_to_seconds(pit_stop.get('stationary_time', '0')))
                in_lap = self.df[(self.df['NUMBER'] == car_number) & (self.df['LAP_NUMBER'] == lap_entry - 1)]
                in_loss = in_lap.iloc[0]['LAP_TIME_SEC'] - in_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC'] if not in_lap.empty and pd.notna(in_lap.iloc[0]['LAP_TIME_SEC']) and pd.notna(in_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC']) else 0.0
                out_lap_num = next((int(s['lap_range'].split('-')[0]) for s in car_data.get('stints', []) if int(s['lap_range'].split('-')[0]) > lap_entry), -1)
                out_loss = 0.0
                if out_lap_num != -1:
                    out_lap = self.df[(self.df['NUMBER'] == car_number) & (self.df['LAP_NUMBER'] == out_lap_num)]
                    if not out_lap.empty and pd.notna(out_lap.iloc[0]['LAP_TIME_SEC']) and pd.notna(out_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC']):
                        out_loss = out_lap.iloc[0]['LAP_TIME_SEC'] - out_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC']
                all_losses.append(in_loss + stationary + out_loss)
            if all_losses: car_results.append({'car_number': car_number, 'team': team_name, 'average_cycle_loss_sec': np.mean(all_losses), 'average_cycle_loss': np.mean(all_losses), 'number_of_stops_analyzed': len(all_losses)})
        car_results.sort(key=lambda x: x.get('average_cycle_loss_sec', float('inf')))
        return [{'rank': i + 1, **res} for i, res in enumerate(car_results)]

    def add_degradation_cliff_analysis(self, enhanced_strategy_data):
        stint_len = 35
        for car_analysis in enhanced_strategy_data:
            model = car_analysis.get('tire_degradation_model', {})
            a, b, quality = model.get('deg_coeff_a'), model.get('deg_coeff_b'), model.get('model_quality')
            model.update({'end_of_stint_deg_rate_s_per_lap': None, 'predicted_final_5_laps_loss_s': None})
            if a is not None and b is not None and quality not in ["INSUFFICIENT_DATA", "POOR"]:
                deg_rate = (2 * a * stint_len) + b; model.update({'end_of_stint_deg_rate_s_per_lap': deg_rate, 'predicted_final_5_laps_loss_s': 5 * deg_rate})
        return enhanced_strategy_data

    def _get_social_media_highlights(self, fastest_by_car, race_strategy, enhanced_strategy):
        return {"metronome_award": self._get_metronome_award(), "metronome_award_longer": self._get_metronome_award_longer(), "perfect_lap_ranking": self._get_perfect_lap_ranking(fastest_by_car), "manufacturer_showdown": self._get_manufacturer_showdown(race_strategy, enhanced_strategy)}

    def _find_best_consistency_window(self, metric_sec, metric_str, window_size):
        # Find the most consistent window of laps for a given metric (lap time or sector time).
        green_laps = self.df[(self.df['FLAG_AT_FL'] == 'GF') & self.df[metric_sec].notna()].copy()
        if green_laps.empty:
            return {}
        
        best_std = float('inf')
        winner_data = {}
        
        for _, group in green_laps.groupby('stint_id'):
            if len(group) < window_size:
                continue
            
            # Calculate rolling standard deviation for the specified metric
            group['rolling_std'] = group[metric_sec].rolling(window=window_size).std()
            min_std_in_group = group['rolling_std'].min()

            if pd.notna(min_std_in_group) and min_std_in_group < best_std:
                best_std = min_std_in_group
                end_idx = group['rolling_std'].idxmin()
                # Ensure the start index is not negative
                start_idx = max(group.index.min(), end_idx - window_size + 1)
                window_df = group.loc[start_idx : end_idx]
                
                winner_data = {
                    'driver_name': window_df.iloc[-1]['DRIVER_NAME'],
                    'car_number': window_df.iloc[-1]['NUMBER'],
                    'team': window_df.iloc[-1]['TEAM'],
                    'consistency_stdev': best_std,
                    'start_lap': int(window_df.iloc[0]['LAP_NUMBER']),
                    'end_lap': int(window_df.iloc[-1]['LAP_NUMBER']),
                    'times': window_df[metric_str].tolist()
                }
        return winner_data

    def _get_metronome_award(self):
        window = self.config['min_laps_for_metronome']
        return {
            'lap_time': self._find_best_consistency_window('LAP_TIME_SEC', 'LAP_TIME', window),
            'sector_1': self._find_best_consistency_window('S1_SEC', 'S1', window),
            'sector_2': self._find_best_consistency_window('S2_SEC', 'S2', window),
            'sector_3': self._find_best_consistency_window('S3_SEC', 'S3', window)
        }

    def _get_metronome_award_longer(self):
        window = self.config['min_laps_for_metronome_longer']
        return {
            'lap_time': self._find_best_consistency_window('LAP_TIME_SEC', 'LAP_TIME', window),
            'sector_1': self._find_best_consistency_window('S1_SEC', 'S1', window),
            'sector_2': self._find_best_consistency_window('S2_SEC', 'S2', window),
            'sector_3': self._find_best_consistency_window('S3_SEC', 'S3', window)
        }

    def _get_perfect_lap_ranking(self, fastest_by_car):
        ranking = []
        for car in fastest_by_car:
            optimal = self._parse_time_to_seconds(car.get('optimal_lap_time')); fastest = self._parse_time_to_seconds(car.get('fastest_lap', {}).get('time'))
            if pd.notna(optimal) and pd.notna(fastest) and fastest > 0:
                pct = min(100.0, (optimal / fastest) * 100)
                ranking.append({'car_number': car['car_number'], 'driver_name': car.get('fastest_lap', {}).get('driver_name'), 'perfection_pct': pct, 'fastest_lap_time': car.get('fastest_lap', {}).get('time'), 'optimal_lap_time': car.get('optimal_lap_time')})
        ranking.sort(key=lambda x: x['perfection_pct'], reverse=True)
        return [{'rank': i + 1, **res} for i, res in enumerate(ranking)]

    def _get_manufacturer_showdown(self, race_strategy, enhanced_strategy):
        car_info = {c['car_number']: {'manufacturer': c.get('manufacturer'), 'team': c.get('team')} for c in enhanced_strategy}
        best_stints, min_laps = {}, self.config['min_laps_for_manu_showdown']
        for car in race_strategy:
            info = car_info.get(car['car_number'])
            if not info or not info.get('manufacturer'): continue
            for stint in car.get('stints', []):
                avg_time = self._parse_time_to_seconds(stint.get('avg_green_time_formatted')) 
                if pd.notna(avg_time) and stint.get('green_laps', 0) >= min_laps and (info['manufacturer'] not in best_stints or avg_time < best_stints[info['manufacturer']]['best_avg_stint_pace_sec']):
                    best_stints[info['manufacturer']] = {
                        'manufacturer': info['manufacturer'], 
                        'best_avg_stint_pace_sec': avg_time, 
                        'car_number': car['car_number'], 
                        'team': info['team'], 
                        'stint_details': {
                            'stint_number': stint.get('stint_number'),
                            'lap_count': stint.get('green_laps')
                        }
                    }
        showdown = sorted(list(best_stints.values()), key=lambda x: x['best_avg_stint_pace_sec'])
        return [{'rank': i + 1, **res} for i, res in enumerate(showdown)]

    def _remove_sec_keys_recursive(self, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.endswith('_sec'): continue
                if isinstance(v, float) and np.isnan(v): new_dict[k] = None
                else: new_dict[k] = self._remove_sec_keys_recursive(v)
            return new_dict
        elif isinstance(obj, list): return [self._remove_sec_keys_recursive(item) for item in obj]
        else:
            if isinstance(obj, (np.integer, np.floating)): return obj.item() if pd.notna(obj) else None
            if pd.isna(obj): return None
            if isinstance(obj, float): return f"{obj:.3f}"
            return obj

    def _clean_output_for_json(self, analysis_results_dict):
        return self._remove_sec_keys_recursive(analysis_results_dict)

    def run_all_analyses(self):
        print("\n--- Running All Analyses (Original and New) ---")
        print("Running original analysis suite...")
        final_results = {"fastest_by_car_number": self.get_fastest_by_car_number(), "fastest_by_manufacturer": self.get_fastest_by_manufacturer(), "longest_stints_by_manufacturer": self.get_longest_stints_by_manufacturer(), "driver_deltas_by_car": self.get_driver_deltas_by_car(), "manufacturer_driver_pace_gap": self.get_manufacturer_driver_pace_gap(), "race_strategy_by_car": self.get_race_strategy_by_car(), "enhanced_strategy_analysis": self.get_enhanced_strategy_analysis()}
        print("Running new internal analysis suite...")
        final_results["traffic_management_analysis"] = self.get_traffic_management_analysis()
        final_results["full_pit_cycle_analysis"] = self.get_full_pit_cycle_analysis(final_results['race_strategy_by_car'], final_results['enhanced_strategy_analysis'])
        final_results["enhanced_strategy_analysis"] = self.add_degradation_cliff_analysis(final_results['enhanced_strategy_analysis'])
        print("Running new social media highlights suite...")
        final_results["social_media_highlights"] = self._get_social_media_highlights(final_results['fastest_by_car_number'], final_results['race_strategy_by_car'], final_results['enhanced_strategy_analysis'])
        print("All analyses complete. Cleaning for JSON export...")
        return self._clean_output_for_json(final_results)

    def export_to_json_file(self, data, output_filepath):
        print(f"\n--- Exporting to {output_filepath} ---");
        try:
            with open(output_filepath, 'w') as f: json.dump(data, f, indent=4); print(f"Successfully exported data to {output_filepath}")
        except TypeError as e: print(f"ERROR: TypeError during JSON export: {e}.")
        except Exception as e: print(f"ERROR: An unexpected error occurred during JSON export: {e}")

# <<< MODIFIED Main Execution Block >>>
if __name__ == '__main__':
    csv_file = '2025_impc_mido.csv'
    pit_json_file = '2025_mido_race_pits.json'
    output_json_file = '2025_mido_race_results_FINAL_with_all_features.json'
    fuel_file = '2025_mido_fuel.json' # <<< NEW FILE
    try:
        # Pass the new fuel capacity file to the analyzer
        analyzer = IMSADataAnalyzer(
            csv_file, 
            pit_json_filepath=pit_json_file, 
            fuel_capacity_json_filepath=fuel_file
        )
        all_results = analyzer.run_all_analyses()
        analyzer.export_to_json_file(all_results, output_json_file)
    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"An unexpected error occurred at the top level: {e}"); import traceback; traceback.print_exc()