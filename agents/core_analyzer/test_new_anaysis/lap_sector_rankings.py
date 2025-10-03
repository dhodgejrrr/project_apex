import json
import pandas as pd
import numpy as np
import copy

class RaceReportGenerator:
    """
    Processes race timing data with a fully autonomous, data-driven handicap system,
    including a "Theoretical Best Lap" calculation and ranking.
    """
    STINT_LAP_PENALTY = 0.0001
    COMPETITIVE_PACE_PERCENTILE = 0.75
    PACE_LEADER_PERCENTILE = 0.25
    PACE_LEADER_WEIGHT = 0.7
    CONFIDENCE_DIVISOR = 2.0
    APEX_CLASS_DEFENSIVE_FACTOR = 0.999

    def __init__(self, data):
        self.raw_data = data
        self.driver_stats = []
        self.final_report = {}
        self.dynamic_license_factors = {}
        self.handicap_strength_used = 0.5
        self.apex_defensive_factor_used = self.APEX_CLASS_DEFENSIVE_FACTOR

    @staticmethod
    def _time_to_seconds(time_str):
        if not time_str: return None
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                return int(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except (ValueError, TypeError):
            return None

    def _process_data(self):
        if not self.raw_data: return pd.DataFrame()
        flat_laps = []
        for participant in self.raw_data.get('participants', []):
            driver_map = {str(d.get('number')): {'full_name': f"{d.get('firstname', '')} {d.get('surname', '')}".strip(), 'license': d.get('license')} for d in participant.get('drivers', [])}
            all_laps_for_car = sorted(participant.get('laps', []), key=lambda x: x.get('number', 0))
            driver_stint_counters = {num: 1 for num in driver_map.keys()}
            for lap in all_laps_for_car:
                driver_num = lap.get('driver_number')
                if not driver_num or driver_num not in driver_map: continue
                current_stint_lap = driver_stint_counters[driver_num]
                if lap.get('is_valid') and 'sector_times' in lap and len(lap['sector_times']) >= 3:
                    lap_time, s1_time, s2_time, s3_time = (self._time_to_seconds(lap.get('time')), self._time_to_seconds(lap['sector_times'][0].get('time')),
                                                          self._time_to_seconds(lap['sector_times'][1].get('time')), self._time_to_seconds(lap['sector_times'][2].get('time')))
                    if all(t is not None for t in [lap_time, s1_time, s2_time, s3_time]):
                        flat_laps.append({'driver_name': driver_map[driver_num]['full_name'], 'license': driver_map[driver_num]['license'], 'vehicle': participant.get('vehicle'),
                                          'class': participant.get('class'), 'lap_time': lap_time, 's1_time': s1_time, 's2_time': s2_time, 's3_time': s3_time,
                                          'stint_lap_number': current_stint_lap})
                if lap.get('crossing_pit_finish_lane', False): driver_stint_counters[driver_num] = 1
                else: driver_stint_counters[driver_num] += 1
        print(f"Processed {len(flat_laps)} valid laps.")
        return pd.DataFrame(flat_laps)

    @staticmethod
    def _calculate_stats(df_slice, time_column):
        df = df_slice[[time_column, 'stint_lap_number']].dropna().sort_values(by=time_column)
        if df.empty: return {}
        fastest_row, best_3_laps, best_5_laps = df.iloc[0], df.head(3), df.head(5)
        return {'fastest': fastest_row[time_column], 'fastest_stint_lap': int(fastest_row['stint_lap_number']),
                'best_3_avg': best_3_laps[time_column].mean(), 'avg_stint_lap_for_best_3': round(best_3_laps['stint_lap_number'].mean(), 2),
                'best_5_avg': best_5_laps[time_column].mean(), 'avg_stint_lap_for_best_5': round(best_5_laps['stint_lap_number'].mean(), 2),
                'deviation_3_lap': best_3_laps[time_column].mean() - fastest_row[time_column], 'deviation_5_lap': best_5_laps[time_column].mean() - fastest_row[time_column]}

    @staticmethod
    def _calculate_optimal_lap(df_slice):
        """
        NEW: Calculates a theoretical best lap by combining the best sectors.
        """
        if df_slice.empty: return {}
        
        s1_best_row = df_slice.loc[df_slice['s1_time'].idxmin()]
        s2_best_row = df_slice.loc[df_slice['s2_time'].idxmin()]
        s3_best_row = df_slice.loc[df_slice['s3_time'].idxmin()]
        
        optimal_time = s1_best_row['s1_time'] + s2_best_row['s2_time'] + s3_best_row['s3_time']
        avg_stint_lap = (s1_best_row['stint_lap_number'] + s2_best_row['stint_lap_number'] + s3_best_row['stint_lap_number']) / 3
        
        return {
            'optimal_lap_time': optimal_time,
            'avg_stint_lap_for_optimal': round(avg_stint_lap, 2)
        }

    def _calculate_all_driver_stats(self, laps_df):
        if laps_df.empty: return
        for group_keys, group_df in laps_df.groupby(['driver_name', 'license', 'vehicle', 'class']):
            driver_data = {'driver_name': group_keys[0], 'license': group_keys[1], 'vehicle': group_keys[2], 'class': group_keys[3],
                           'lap_times': self._calculate_stats(group_df, 'lap_time'), 'sector_1_times': self._calculate_stats(group_df, 's1_time'),
                           'sector_2_times': self._calculate_stats(group_df, 's2_time'), 'sector_3_times': self._calculate_stats(group_df, 's3_time')}
            
            # Calculate and merge in the optimal lap data
            optimal_data = self._calculate_optimal_lap(group_df)
            driver_data['lap_times'].update(optimal_data)
            
            self.driver_stats.append(driver_data)
        print(f"Calculated statistics for {len(self.driver_stats)} drivers.")

    def _calculate_dynamic_license_factors(self):
        df = pd.DataFrame(self.driver_stats)
        df['pace'] = df['lap_times'].apply(lambda x: x.get('best_5_avg'))
        df = df.dropna(subset=['pace', 'license'])
        valid_licenses = ['Bronze', 'Silver', 'Gold', 'Platinum']
        df = df[df['license'].isin(valid_licenses)]
        if df.empty: return

        competitive_paces, competitive_groups = {}, {}
        for license_class in valid_licenses:
            if license_class not in df['license'].unique(): continue
            class_df = df[df['license'] == license_class].copy()
            cutoff = class_df['pace'].quantile(self.COMPETITIVE_PACE_PERCENTILE)
            comp_group = class_df[class_df['pace'] <= cutoff]
            if comp_group.empty: continue
            competitive_groups[license_class] = comp_group
            if license_class == 'Bronze':
                leader_cutoff = class_df['pace'].quantile(self.PACE_LEADER_PERCENTILE)
                leaders, pack = class_df[class_df['pace'] <= leader_cutoff], class_df[(class_df['pace'] > leader_cutoff) & (class_df['pace'] <= cutoff)]
                if leaders.empty: competitive_paces['Bronze'] = pack['pace'].mean() if not pack.empty else None
                elif pack.empty: competitive_paces['Bronze'] = leaders['pace'].mean()
                else: competitive_paces['Bronze'] = (leaders['pace'].mean() * self.PACE_LEADER_WEIGHT) + (pack['pace'].mean() * (1 - self.PACE_LEADER_WEIGHT))
            else:
                competitive_paces[license_class] = comp_group['pace'].mean()
        if not competitive_paces: return

        baseline_class = min(competitive_paces, key=lambda k: competitive_paces.get(k, float('inf')))
        baseline_pace = competitive_paces[baseline_class]
        self.baseline_license_class_used = baseline_class
        print(f"\nIdentified '{baseline_class}' as the performance baseline (Pace: {baseline_pace:.3f}s).")
        
        if baseline_class in competitive_groups:
            baseline_group, baseline_spread = competitive_groups[baseline_class], competitive_groups[baseline_class]['pace'].std()
            if baseline_spread > 0:
                non_baseline_class = 'Bronze' if baseline_class != 'Bronze' else 'Silver'
                if non_baseline_class in competitive_paces:
                    raw_pace_gap = abs(competitive_paces[non_baseline_class] - baseline_pace)
                    signal_to_noise = raw_pace_gap / baseline_spread
                    self.handicap_strength_used = np.clip(signal_to_noise / self.CONFIDENCE_DIVISOR, 0.2, 0.9)
            print(f"Derived Handicap Strength: {self.handicap_strength_used*100:.0f}%")

        for license_class, avg_pace in competitive_paces.items():
            if avg_pace is None: continue
            deviation = (avg_pace - baseline_pace) / baseline_pace
            adjusted_deviation = deviation * self.handicap_strength_used
            self.dynamic_license_factors[license_class] = 1.0 - adjusted_deviation
        
        self.dynamic_license_factors[baseline_class] = self.APEX_CLASS_DEFENSIVE_FACTOR
        print(f"Generated final license adjustment factors.", json.dumps(self.dynamic_license_factors, indent=2))

    def _add_adjusted_scores(self):
        for driver in self.driver_stats:
            factor = self.dynamic_license_factors.get(driver['license'], 1.0)
            for time_key in ['lap_times', 'sector_1_times', 'sector_2_times', 'sector_3_times']:
                stats = driver.get(time_key, {})
                if not stats or 'fastest' not in stats: continue
                stint_fastest = stats['fastest'] + (stats.get('fastest_stint_lap', 0) * self.STINT_LAP_PENALTY)
                stint_3_avg = stats.get('best_3_avg', 0) + (stats.get('avg_stint_lap_for_best_3', 0) * self.STINT_LAP_PENALTY)
                stint_5_avg = stats.get('best_5_avg', 0) + (stats.get('avg_stint_lap_for_best_5', 0) * self.STINT_LAP_PENALTY)
                stats.update({'stint_adjusted_fastest': stint_fastest, 'stint_adjusted_best_3_avg': stint_3_avg, 'stint_adjusted_best_5_avg': stint_5_avg})
                stats.update({'license_adjusted_fastest': stats['fastest'] * factor, 'license_adjusted_best_3_avg': stats.get('best_3_avg', 0) * factor,
                              'license_adjusted_best_5_avg': stats.get('best_5_avg', 0) * factor})
                stats.update({'combo_adjusted_fastest': stint_fastest * factor, 'combo_adjusted_best_3_avg': stint_3_avg * factor,
                              'combo_adjusted_best_5_avg': stint_5_avg * factor})
            
            # Also apply adjustments to the optimal lap time
            lap_stats = driver.get('lap_times', {})
            if 'optimal_lap_time' in lap_stats:
                stint_optimal = lap_stats['optimal_lap_time'] + (lap_stats.get('avg_stint_lap_for_optimal', 0) * self.STINT_LAP_PENALTY)
                lap_stats['stint_adjusted_optimal_lap'] = stint_optimal
                lap_stats['license_adjusted_optimal_lap'] = lap_stats['optimal_lap_time'] * factor
                lap_stats['combo_adjusted_optimal_lap'] = stint_optimal * factor
        print("Generated all adjusted score sets.")

    def _generate_rankings(self):
        def create_ranking(stats, time_key, metric_key):
            return sorted([{'driver_name': d['driver_name'], 'license': d['license'], 'vehicle': d['vehicle'], 'value': d.get(time_key, {}).get(metric_key)}
                           for d in stats if d.get(time_key, {}).get(metric_key) is not None], key=lambda x: x['value'])
        def generate_rankings_for_group(stats):
            rankings = {}
            sector_metrics = ['fastest', 'best_3_avg', 'best_5_avg', 'stint_adjusted_fastest', 'stint_adjusted_best_3_avg', 'stint_adjusted_best_5_avg',
                              'license_adjusted_fastest', 'license_adjusted_best_3_avg', 'license_adjusted_best_5_avg', 'combo_adjusted_fastest',
                              'combo_adjusted_best_3_avg', 'combo_adjusted_best_5_avg']
            lap_only_metrics = ['deviation_3_lap', 'deviation_5_lap', 'fastest_stint_lap', 'avg_stint_lap_for_best_3', 'avg_stint_lap_for_best_5',
                                'optimal_lap_time', 'avg_stint_lap_for_optimal', 'stint_adjusted_optimal_lap', 'license_adjusted_optimal_lap',
                                'combo_adjusted_optimal_lap']
            
            for metric in sector_metrics:
                for name, k in {'lap': 'lap_times', 's1': 'sector_1_times', 's2': 'sector_2_times', 's3': 'sector_3_times'}.items():
                    rankings[f'by_{metric}_{name}'] = create_ranking(stats, k, metric)
            for metric in lap_only_metrics:
                rankings[f'by_{metric}_lap'] = create_ranking(stats, 'lap_times', metric)
            return rankings
        license_groups, vehicle_groups = pd.DataFrame(self.driver_stats).groupby('license'), pd.DataFrame(self.driver_stats).groupby('vehicle')
        return {'overall': generate_rankings_for_group(self.driver_stats),
                'by_license': {name: generate_rankings_for_group(g.to_dict('records')) for name, g in license_groups},
                'by_vehicle': {name: generate_rankings_for_group(g.to_dict('records')) for name, g in vehicle_groups}}

    def generate_report(self):
        laps_df = self._process_data()
        self._calculate_all_driver_stats(laps_df)
        self._calculate_dynamic_license_factors()
        self._add_adjusted_scores()
        self.final_report = {'driver_performance': self.driver_stats, 'rankings': self._generate_rankings(),
                             'dynamic_license_factors_used': self.dynamic_license_factors,
                             'baseline_license_class_used': getattr(self, 'baseline_license_class_used', 'N/A'),
                             'handicap_strength_used': self.handicap_strength_used,
                             'apex_defensive_factor_used': self.apex_defensive_factor_used}
        print("Final report generated.")
        return self.final_report

    def save_report(self, output_filepath):
        if not self.final_report: return
        with open(output_filepath, 'w') as f: json.dump(self.final_report, f, indent=2)
        print(f"Report saved to '{output_filepath}'")

if __name__ == "__main__":
    INPUT_FILE = "vp_vir_r1.json"
    try:
        print(f"Loading full dataset from '{INPUT_FILE}'...")
        with open(INPUT_FILE, 'r') as f: full_data = json.load(f)
        participants = full_data.get('participants', [])
        if not participants: raise ValueError("No participants found.")
        unique_classes = sorted({p.get('class') for p in participants if p.get('class')})
        print(f"Found unique classes: {unique_classes}\n")
        for race_class in unique_classes:
            print(f"--- Generating report for class: {race_class} ---")
            class_data = {'session': copy.deepcopy(full_data['session']), 'participants': [p for p in participants if p.get('class') == race_class]}
            if not class_data['participants']: continue
            report_generator = RaceReportGenerator(class_data)
            report_generator.generate_report()
            report_generator.save_report(f"race_report_vp_vir_r1_v2_{race_class}.json")
            print("-" * 50)
    except FileNotFoundError: print(f"FATAL ERROR: Input file '{INPUT_FILE}' not found.")
    except Exception as e: print(f"An unexpected error occurred: {e}")