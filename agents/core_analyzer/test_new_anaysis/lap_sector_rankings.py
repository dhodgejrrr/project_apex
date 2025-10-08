import json
import pandas as pd
import numpy as np
import copy

class RaceReportGenerator:
    """
    Processes race timing data with a fully autonomous, data-driven handicap system,
    and generates a detailed Pace Profile for each driver.
    """
    # STINT_LAP_PENALTY is now removed and replaced by a dynamic degradation model.
    COMPETITIVE_PACE_PERCENTILE = 0.75
    PACE_LEADER_PERCENTILE = 0.25
    PACE_LEADER_WEIGHT = 0.7
    CONFIDENCE_DIVISOR = 2.0
    APEX_CLASS_DEFENSIVE_FACTOR = 0.999
    # New: Threshold for flagging slow/erroneous laps (e.g., 1.07 = 7% slower)
    OUT_OF_RANGE_THRESHOLD = 1.05

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
                                          'stint_lap_number': current_stint_lap, 'lap_number': lap.get('number')})
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
        if df_slice.empty or not all(c in df_slice.columns for c in ['s1_time', 's2_time', 's3_time']): return {}
        s1_best_row, s2_best_row, s3_best_row = (df_slice.loc[df_slice['s1_time'].idxmin()], df_slice.loc[df_slice['s2_time'].idxmin()],
                                                 df_slice.loc[df_slice['s3_time'].idxmin()])
        optimal_time = s1_best_row['s1_time'] + s2_best_row['s2_time'] + s3_best_row['s3_time']
        avg_stint_lap = (s1_best_row['stint_lap_number'] + s2_best_row['stint_lap_number'] + s3_best_row['stint_lap_number']) / 3
        return {'optimal_lap_time': optimal_time, 'avg_stint_lap_for_optimal': round(avg_stint_lap, 2)}
    
    # NEW: Method to calculate stint degradation
    @staticmethod
    def _calculate_stint_degradation(driver_df):
        stint_starts = driver_df.index[driver_df['stint_lap_number'] == 1].tolist()
        if not stint_starts:
            stints = [driver_df]
        else:
            stints = np.split(driver_df, stint_starts[1:])
        
        degradation_rates = []
        consistency_scores = []
        
        for stint_df in stints:
            if len(stint_df) > 5: # Minimum stint length for meaningful regression
                try:
                    x = stint_df['stint_lap_number']
                    y = stint_df['lap_time']
                    
                    slope, _ = np.polyfit(x, y, 1)
                    
                    # Calculate R-squared for consistency
                    y_predicted = np.polyval(np.polyfit(x, y, 1), x)
                    ss_res = np.sum((y - y_predicted) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    
                    if ss_tot > 0:
                        r_squared = 1 - (ss_res / ss_tot)
                        degradation_rates.append(slope)
                        consistency_scores.append(r_squared)
                except (np.linalg.LinAlgError, ValueError):
                    continue # Skip if regression fails

        avg_degradation = np.mean(degradation_rates) if degradation_rates else None
        avg_consistency = np.mean(consistency_scores) if consistency_scores else None
        
        return {'average_degradation_rate': avg_degradation, 'stint_consistency_r2': avg_consistency}

    def _calculate_all_driver_stats(self, laps_df):
        if laps_df.empty: return
        for group_keys, group_df in laps_df.groupby(['driver_name', 'license', 'vehicle', 'class']):
            driver_data = {'driver_name': group_keys[0], 'license': group_keys[1], 'vehicle': group_keys[2], 'class': group_keys[3],
                           'lap_times': self._calculate_stats(group_df, 'lap_time'), 'sector_1_times': self._calculate_stats(group_df, 's1_time'),
                           'sector_2_times': self._calculate_stats(group_df, 's2_time'), 'sector_3_times': self._calculate_stats(group_df, 's3_time')}
            driver_data['lap_times'].update(self._calculate_optimal_lap(group_df))
            # NEW: Add stint degradation stats
            driver_data.update(self._calculate_stint_degradation(group_df))
            self.driver_stats.append(driver_data)
        print(f"Calculated statistics for {len(self.driver_stats)} drivers.")

    # MODIFIED: Expanded to include sector pace profiles
    def _calculate_pace_profiles_and_lap_counts(self, laps_df):
        if laps_df.empty: return
        print("Generating driver Pace Profiles...")
        driver_benchmarks = {d['driver_name']: d.get('lap_times', {}).get('best_5_avg') for d in self.driver_stats}
        laps_df['benchmark_pace'] = laps_df['driver_name'].map(driver_benchmarks)
        
        license_factors_s = laps_df['license'].map(self.dynamic_license_factors).fillna(1.0)
        
        # MODIFIED: Weighted time no longer includes stint penalty.
        laps_df['weighted_lap_time'] = laps_df['lap_time'] * license_factors_s
        # NEW: Add weighted sector times for profile calculation
        laps_df['weighted_s1_time'] = laps_df['s1_time'] * license_factors_s
        laps_df['weighted_s2_time'] = laps_df['s2_time'] * license_factors_s
        laps_df['weighted_s3_time'] = laps_df['s3_time'] * license_factors_s

        is_outlier = laps_df['lap_time'] > (laps_df['benchmark_pace'] * self.OUT_OF_RANGE_THRESHOLD)
        out_of_range_counts = laps_df[is_outlier].groupby('driver_name').size()
        total_lap_counts = laps_df.groupby('driver_name').size()

        # NEW: Expanded profile structure
        profile_keys = ['by_raw_pace', 'by_weighted_pace', 'by_raw_s1_pace', 'by_weighted_s1_pace',
                        'by_raw_s2_pace', 'by_weighted_s2_pace', 'by_raw_s3_pace', 'by_weighted_s3_pace']
        empty_profile = {key: {'p1_laps': 0, 'top_5_laps': 0, 'top_10_laps': 0, 'top_15_laps': 0} for key in profile_keys}
        profiles = {d['driver_name']: copy.deepcopy(empty_profile) for d in self.driver_stats}

        rank_configs = [
            {'sort_col': 'lap_time', 'profile_key': 'by_raw_pace'},
            {'sort_col': 'weighted_lap_time', 'profile_key': 'by_weighted_pace'},
            {'sort_col': 's1_time', 'profile_key': 'by_raw_s1_pace'},
            {'sort_col': 'weighted_s1_time', 'profile_key': 'by_weighted_s1_pace'},
            {'sort_col': 's2_time', 'profile_key': 'by_raw_s2_pace'},
            {'sort_col': 'weighted_s2_time', 'profile_key': 'by_weighted_s2_pace'},
            {'sort_col': 's3_time', 'profile_key': 'by_raw_s3_pace'},
            {'sort_col': 'weighted_s3_time', 'profile_key': 'by_weighted_s3_pace'},
        ]

        for lap_num, group in laps_df.groupby('lap_number'):
            for config in rank_configs:
                sorted_group = group.sort_values(config['sort_col']).reset_index()
                for rank, row in sorted_group.head(15).iterrows():
                    driver_name = row['driver_name']
                    if driver_name not in profiles: continue
                    profile = profiles[driver_name][config['profile_key']]
                    if rank == 0: profile['p1_laps'] += 1
                    if rank < 5: profile['top_5_laps'] += 1
                    if rank < 10: profile['top_10_laps'] += 1
                    profile['top_15_laps'] += 1

        for driver_stat in self.driver_stats:
            driver_name = driver_stat['driver_name']
            driver_stat['total_valid_laps'] = int(total_lap_counts.get(driver_name, 0))
            profile = profiles.get(driver_name)
            if profile:
                profile['total_laps_in_profile'] = int(total_lap_counts.get(driver_name, 0))
                profile['laps_out_of_range'] = int(out_of_range_counts.get(driver_name, 0))
                driver_stat['pace_profile'] = profile

    # MODIFIED: Apex class factor logic updated to be a fallback.
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
        
        if baseline_class in competitive_groups:
            baseline_spread = competitive_groups[baseline_class]['pace'].std()
            if baseline_spread > 0:
                non_baseline_class = 'Bronze' if baseline_class != 'Bronze' else 'Silver'
                if non_baseline_class in competitive_paces:
                    raw_pace_gap = abs(competitive_paces[non_baseline_class] - baseline_pace)
                    self.handicap_strength_used = np.clip((raw_pace_gap / baseline_spread) / self.CONFIDENCE_DIVISOR, 0.2, 0.9)
        
        for license_class, avg_pace in competitive_paces.items():
            if avg_pace is None: continue
            deviation = (avg_pace - baseline_pace) / baseline_pace
            self.dynamic_license_factors[license_class] = 1.0 - (deviation * self.handicap_strength_used)
        
        # MODIFIED: Set baseline to 1.0 by default. Apply defensive factor only if strength is extreme.
        self.dynamic_license_factors[baseline_class] = 1.0
        if self.handicap_strength_used > 0.85:
            self.dynamic_license_factors[baseline_class] = self.APEX_CLASS_DEFENSIVE_FACTOR
            self.apex_defensive_factor_used = self.APEX_CLASS_DEFENSIVE_FACTOR # Log that it was used
        else:
            self.apex_defensive_factor_used = 1.0 # Log that it was NOT used

    # MODIFIED: Stint adjustments removed, combo adjustments simplified.
    def _add_adjusted_scores(self):
        for driver in self.driver_stats:
            factor = self.dynamic_license_factors.get(driver['license'], 1.0)
            for time_key in ['lap_times', 'sector_1_times', 'sector_2_times', 'sector_3_times']:
                stats = driver.get(time_key, {})
                if not stats or 'fastest' not in stats: continue
                # REMOVED: Stint-adjusted calculations are gone.
                # MODIFIED: Combo-adjusted now equals license-adjusted.
                stats.update({'license_adjusted_fastest': stats['fastest'] * factor, 'license_adjusted_best_3_avg': stats.get('best_3_avg', 0) * factor,
                              'license_adjusted_best_5_avg': stats.get('best_5_avg', 0) * factor})
                stats.update({'combo_adjusted_fastest': stats['fastest'] * factor, 'combo_adjusted_best_3_avg': stats.get('best_3_avg', 0) * factor,
                              'combo_adjusted_best_5_avg': stats.get('best_5_avg', 0) * factor})

            lap_stats = driver.get('lap_times', {})
            if 'optimal_lap_time' in lap_stats:
                lap_stats.update({'license_adjusted_optimal_lap': lap_stats['optimal_lap_time'] * factor,
                                  'combo_adjusted_optimal_lap': lap_stats['optimal_lap_time'] * factor})
        print("Generated all adjusted score sets.")

    # MODIFIED: Added degradation rate to Alpha Score.
    def _calculate_alpha_score(self):
        pillars = {
            'z_fastest': {'path': ('lap_times', 'fastest'), 'weight': 1.0, 'invert': True},
            'z_optimal': {'path': ('lap_times', 'optimal_lap_time'), 'weight': 1.15, 'invert': True},
            'z_combo_fastest': {'path': ('lap_times', 'combo_adjusted_fastest'), 'weight': 1.35, 'invert': True},
            'z_combo_optimal': {'path': ('lap_times', 'combo_adjusted_optimal_lap'), 'weight': 1.75, 'invert': True},
            'z_best_3_avg': {'path': ('lap_times', 'best_3_avg'), 'weight': 1.0, 'invert': True},
            'z_consistency': {'path': ('lap_times', 'deviation_5_lap'), 'weight': 1.15, 'invert': True},
            'z_s1_combo_fastest': {'path': ('sector_1_times', 'combo_adjusted_fastest'), 'weight': 1.25, 'invert': True, 'sector': 's1'},
            'z_s1_best_3_avg': {'path': ('sector_1_times', 'best_3_avg'), 'weight': 1.0, 'invert': True, 'sector': 's1'},
            'z_s2_combo_fastest': {'path': ('sector_2_times', 'combo_adjusted_fastest'), 'weight': 1.25, 'invert': True, 'sector': 's2'},
            'z_s2_best_3_avg': {'path': ('sector_2_times', 'best_3_avg'), 'weight': 1.0, 'invert': True, 'sector': 's2'},
            'z_s3_combo_fastest': {'path': ('sector_3_times', 'combo_adjusted_fastest'), 'weight': 1.25, 'invert': True, 'sector': 's3'},
            'z_s3_best_3_avg': {'path': ('sector_3_times', 'best_3_avg'), 'weight': 1.0, 'invert': True, 'sector': 's3'},
            # NEW: Degradation rate as a pillar for Alpha Score
            'z_degradation': {'path': ('average_degradation_rate',), 'weight': 1.5, 'invert': True},
        }
        # MODIFIED: Handle paths of different lengths
        data_for_z = []
        for d in self.driver_stats:
            row_data = {'driver_name': d['driver_name']}
            for name, p in pillars.items():
                if len(p['path']) == 2:
                    row_data[name] = d.get(p['path'][0], {}).get(p['path'][1])
                else:
                    row_data[name] = d.get(p['path'][0])
            data_for_z.append(row_data)
        
        df_z = pd.DataFrame(data_for_z)

        avg_s1, avg_s2, avg_s3 = df_z['z_s1_best_3_avg'].mean(), df_z['z_s2_best_3_avg'].mean(), df_z['z_s3_best_3_avg'].mean()
        total_time = avg_s1 + avg_s2 + avg_s3
        if total_time == 0: return
        sector_weights = {'s1': avg_s1 / total_time, 's2': avg_s2 / total_time, 's3': avg_s3 / total_time}

        for driver in self.driver_stats:
            driver['z_scores'] = {}
            total_score, total_weight = 0, 0
            row = df_z.loc[df_z['driver_name'] == driver['driver_name']]
            if row.empty: continue
            
            for name, p in pillars.items():
                value = row[name].iloc[0]
                if pd.isna(value): continue
                mean, std = df_z[name].mean(), df_z[name].std()
                if std == 0: continue
                
                z_score = (value - mean) / std
                if p['invert']: z_score *= -1
                driver['z_scores'][name] = z_score
                
                final_weight = p['weight'] * sector_weights.get(p.get('sector'), 1)
                total_score += z_score * final_weight
                total_weight += final_weight
            
            driver['alpha_score'] = total_score / total_weight if total_weight > 0 else 0
        print("Generated final weighted 'Alpha Scores'.")

    def _generate_rankings(self):
        def create_ranking(stats, metric_key, time_key=None, descending=False):
            ranked_list = []
            for d in stats:
                value = None
                if time_key:
                    value = d.get(time_key, {}).get(metric_key)
                else: 
                    value = d.get(metric_key)
                
                if value is not None:
                    ranked_list.append({
                        'driver_name': d.get('driver_name'), 'license': d.get('license'),
                        'vehicle': d.get('vehicle'), 'value': value
                    })
            return sorted(ranked_list, key=lambda x: x['value'], reverse=descending)

        def generate_rankings_for_group(stats):
            rankings = {}
            time_metrics = {'lap': 'lap_times', 's1': 'sector_1_times', 's2': 'sector_2_times', 's3': 'sector_3_times'}
            
            # MODIFIED: Removed stint_adjusted rankings.
            metrics_to_rank = [
                'fastest', 'best_3_avg', 'best_5_avg', 'license_adjusted_fastest', 'license_adjusted_best_3_avg', 'license_adjusted_best_5_avg',
                'combo_adjusted_fastest', 'combo_adjusted_best_3_avg', 'combo_adjusted_best_5_avg'
            ]
            
            for metric in metrics_to_rank:
                for name, key in time_metrics.items():
                    rankings[f'by_{metric}_{name}'] = create_ranking(stats, metric, time_key=key)

            lap_only_metrics = ['deviation_5_lap', 'optimal_lap_time', 'combo_adjusted_optimal_lap']
            for metric in lap_only_metrics:
                 rankings[f'by_{metric}_lap'] = create_ranking(stats, metric, time_key='lap_times')
            
            # Add new degradation rankings
            rankings['by_degradation_rate'] = create_ranking(stats, 'average_degradation_rate')
            rankings['by_stint_consistency'] = create_ranking(stats, 'stint_consistency_r2', descending=True)

            rankings['by_alpha_score'] = create_ranking(stats, 'alpha_score', descending=True)

            return rankings
        
        license_groups = pd.DataFrame(self.driver_stats).groupby('license')
        vehicle_groups = pd.DataFrame(self.driver_stats).groupby('vehicle')
        return {'overall': generate_rankings_for_group(self.driver_stats),
                'by_license': {name: generate_rankings_for_group(g.to_dict('records')) for name, g in license_groups},
                'by_vehicle': {name: generate_rankings_for_group(g.to_dict('records')) for name, g in vehicle_groups}}

    def generate_report(self):
        laps_df = self._process_data()
        self._calculate_all_driver_stats(laps_df)
        self._calculate_dynamic_license_factors()
        self._calculate_pace_profiles_and_lap_counts(laps_df) 
        self._add_adjusted_scores()
        self._calculate_alpha_score()
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

class RaceTimelineGenerator:
    """
    Generates a chronological, lap-by-lap report of race performance, including
    both raw and weighted (combo-adjusted) rankings.
    """
    # REMOVED: STINT_LAP_PENALTY is no longer used.
    
    def __init__(self, data, license_factors):
        self.raw_data = data
        self.final_report = {}
        self.dynamic_license_factors = license_factors

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

    # MODIFIED: Removed stint penalty from calculations
    def _process_all_laps(self):
        all_laps_flat = []
        if not self.raw_data: return pd.DataFrame()

        for participant in self.raw_data.get('participants', []):
            driver_map = {str(d.get('number')): {'full_name': d.get('firstname', '') + ' ' + d.get('surname', ''), 'license': d.get('license')}
                          for d in participant.get('drivers', [])}
            
            all_laps_for_car = sorted(participant.get('laps', []), key=lambda x: x.get('number', 0))
            
            for lap in all_laps_for_car:
                driver_num = lap.get('driver_number')
                if not driver_num or driver_num not in driver_map: continue
                
                if lap.get('is_valid') and 'sector_times' in lap and len(lap['sector_times']) >= 3:
                    lap_data = {
                        'lap_number': lap.get('number'),
                        'driver_name': driver_map[driver_num]['full_name'],
                        'license': driver_map[driver_num]['license'],
                        'vehicle': participant.get('vehicle')
                    }
                    
                    times = {
                        'lap_time': self._time_to_seconds(lap.get('time')),
                        's1_time': self._time_to_seconds(lap['sector_times'][0].get('time')),
                        's2_time': self._time_to_seconds(lap['sector_times'][1].get('time')),
                        's3_time': self._time_to_seconds(lap['sector_times'][2].get('time')),
                    }

                    if not all(times.values()): continue 

                    license_factor = self.dynamic_license_factors.get(lap_data['license'], 1.0)
                    
                    for key, raw_time in times.items():
                        combo_adjusted = raw_time * license_factor
                        lap_data[f"raw_{key}"] = raw_time
                        lap_data[f"weighted_{key}"] = combo_adjusted
                    
                    all_laps_flat.append(lap_data)
        
        print(f"Timeline: Processed {len(all_laps_flat)} laps for chronological analysis.")
        return pd.DataFrame(all_laps_flat)

    def _generate_timeline(self, all_laps_df):
        if all_laps_df.empty: return {}, 0
        
        timeline = {}
        grouped_by_lap = all_laps_df.groupby('lap_number')
        max_lap = all_laps_df['lap_number'].max()

        metrics_to_rank = [
            {'key': 'raw_lap_time', 'name': 'by_raw_lap_time', 'time_col': 'raw_lap_time'},
            {'key': 'weighted_lap_time', 'name': 'by_weighted_lap_time', 'time_col': 'raw_lap_time'},
            {'key': 'raw_s1_time', 'name': 'by_raw_s1_time', 'time_col': 'raw_s1_time'},
            {'key': 'weighted_s1_time', 'name': 'by_weighted_s1_time', 'time_col': 'raw_s1_time'},
            {'key': 'raw_s2_time', 'name': 'by_raw_s2_time', 'time_col': 'raw_s2_time'},
            {'key': 'weighted_s2_time', 'name': 'by_weighted_s2_time', 'time_col': 'raw_s2_time'},
            {'key': 'raw_s3_time', 'name': 'by_raw_s3_time', 'time_col': 'raw_s3_time'},
            {'key': 'weighted_s3_time', 'name': 'by_weighted_s3_time', 'time_col': 'raw_s3_time'},
        ]

        for lap_num, group in grouped_by_lap:
            lap_key = str(lap_num)
            timeline[lap_key] = {}
            for config in metrics_to_rank:
                sorted_group = group.sort_values(by=config['key']).head(15)
                ranking_data = []
                for i, row in enumerate(sorted_group.iterrows(), 1):
                    entry = {
                        'rank': i,
                        'driver_name': row[1]['driver_name'],
                        'license': row[1]['license'],
                        'vehicle': row[1]['vehicle']
                    }
                    if 'weighted' in config['key']:
                        entry['value'] = row[1][config['key']]
                        entry['raw_time'] = row[1][config['time_col']]
                    else: 
                        entry[config['key'].replace('raw_', '')] = row[1][config['key']]
                    ranking_data.append(entry)
                timeline[lap_key][config['name']] = ranking_data
        
        return timeline, max_lap

    def generate_report(self):
        all_laps_df = self._process_all_laps()
        timeline_data, total_laps = self._generate_timeline(all_laps_df)
        self.final_report = {
            'session_details': {'session_name': self.raw_data['session'].get('session_name'), 
                                'event_name': self.raw_data['session'].get('event_name'),
                                'total_laps': int(total_laps) if pd.notna(total_laps) else 0},
            'timeline': timeline_data
        }
        print("Timeline: Final report generated.")
        return self.final_report

    # REMOVED: Duplicate method is gone.
    def save_report(self, output_filepath):
        if not self.final_report: return
        with open(output_filepath, 'w') as f: json.dump(self.final_report, f, indent=2)
        print(f"Timeline: Report successfully saved to '{output_filepath}'")

# --- MAIN ORCHESTRATOR SCRIPT ---
if __name__ == "__main__":
    INPUT_FILE = "impc_mido.json"
    
    try:
        print(f"Loading full dataset from '{INPUT_FILE}'...")
        with open(INPUT_FILE, 'r') as f: full_data = json.load(f)
        
        participants = full_data.get('participants', [])
        if not participants: raise ValueError("No participants found.")
        
        unique_classes = sorted({p.get('class') for p in participants if p.get('class')})
        print(f"Found unique classes: {unique_classes}\n")

        for race_class in unique_classes:
            print(f"--- Generating reports for class: {race_class} ---")
            
            class_data = {'session': copy.deepcopy(full_data['session']), 'participants': [p for p in participants if p.get('class') == race_class]}
            if not class_data['participants']: continue

            print("Generating main driver performance report...")
            report_generator = RaceReportGenerator(class_data)
            report_generator.generate_report()
            report_generator.save_report(f"race_report_mido_v10_{race_class}.json")
            
            license_factors = report_generator.dynamic_license_factors
            if not license_factors:
                print("Skipping timeline generation as no license factors were created.")
                print("-" * 50)
                continue

            print("\nGenerating chronological race timeline report...")
            timeline_generator = RaceTimelineGenerator(class_data, license_factors)
            timeline_generator.generate_report()
            timeline_generator.save_report(f"race_timeline_mido_v10_{race_class}.json")
            
            print("-" * 50)
            
    except FileNotFoundError: print(f"FATAL ERROR: Input file '{INPUT_FILE}' not found.")
    except Exception as e: print(f"An unexpected error occurred: {e}")