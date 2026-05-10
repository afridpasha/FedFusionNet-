"""
Survival Analysis Module for FedFusionNet++
Generates Kaplan-Meier curves and survival predictions
"""

import numpy as np
from datetime import datetime, timedelta
import json

class SurvivalAnalyzer:
    """
    Analyzes survival probabilities for oral cancer patients
    """
    
    def __init__(self):
        """
        Initialize survival analyzer with baseline survival data
        """
        # Baseline survival rates by stage (from literature)
        # Source: AJCC Cancer Staging Manual, 8th Edition
        self.baseline_survival = {
            0: {  # Stage 0 (Carcinoma in situ)
                '1_year': 0.98,
                '3_year': 0.95,
                '5_year': 0.92,
                '10_year': 0.88
            },
            1: {  # Stage I
                '1_year': 0.95,
                '3_year': 0.88,
                '5_year': 0.83,
                '10_year': 0.75
            },
            2: {  # Stage II
                '1_year': 0.90,
                '3_year': 0.78,
                '5_year': 0.68,
                '10_year': 0.55
            },
            3: {  # Stage III
                '1_year': 0.82,
                '3_year': 0.62,
                '5_year': 0.48,
                '10_year': 0.35
            },
            4: {  # Stage IV
                '1_year': 0.65,
                '3_year': 0.42,
                '5_year': 0.28,
                '10_year': 0.15
            }
        }
        
        # Risk factor multipliers (hazard ratios from literature)
        self.risk_multipliers = {
            'Tobacco Use': 1.8,
            'Alcohol Consumption': 1.5,
            'Betel Quid Use': 2.2,
            'HPV Infection': 0.7,  # Protective for survival
            'Poor Oral Hygiene': 1.3,
            'Compromised Immune System': 1.6,
            'Age_over_60': 1.4,
            'Male': 1.2
        }
    
    def calculate_survival_curve(self, cancer_stage, patient_data, confidence=0.95):
        """
        Calculate personalized survival curve for a patient
        
        Args:
            cancer_stage: Cancer stage (0-4)
            patient_data: Dictionary of patient features
            confidence: Confidence interval (default 0.95)
            
        Returns:
            Dictionary containing survival curve data
        """
        # Get baseline survival for this stage
        if cancer_stage not in self.baseline_survival:
            cancer_stage = 0  # Default to stage 0 if invalid
        
        baseline = self.baseline_survival[cancer_stage]
        
        # Calculate risk adjustment based on patient factors
        risk_adjustment = self._calculate_risk_adjustment(patient_data)
        
        # Generate time points (months)
        time_points = list(range(0, 121, 3))  # 0 to 120 months (10 years), every 3 months
        
        # Calculate survival probabilities
        survival_probs = []
        lower_ci = []
        upper_ci = []
        
        for month in time_points:
            # Interpolate survival probability
            prob = self._interpolate_survival(baseline, month, risk_adjustment)
            
            # Calculate confidence intervals
            se = self._calculate_standard_error(prob, month)
            ci_range = 1.96 * se  # 95% CI
            
            survival_probs.append(prob)
            lower_ci.append(max(0, prob - ci_range))
            upper_ci.append(min(1, prob + ci_range))
        
        # Calculate key milestones
        milestones = {
            '1_year': self._get_survival_at_time(12, time_points, survival_probs),
            '3_year': self._get_survival_at_time(36, time_points, survival_probs),
            '5_year': self._get_survival_at_time(60, time_points, survival_probs),
            '10_year': self._get_survival_at_time(120, time_points, survival_probs)
        }
        
        # Calculate median survival time
        median_survival = self._calculate_median_survival(time_points, survival_probs)
        
        return {
            'time_points': time_points,
            'survival_probabilities': survival_probs,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'milestones': milestones,
            'median_survival_months': median_survival,
            'risk_adjustment_factor': risk_adjustment,
            'baseline_stage': cancer_stage
        }
    
    def _calculate_risk_adjustment(self, patient_data):
        """
        Calculate cumulative risk adjustment based on patient factors
        
        Args:
            patient_data: Dictionary of patient features
            
        Returns:
            Risk adjustment factor (1.0 = baseline, >1.0 = worse prognosis)
        """
        risk_factor = 1.0
        
        # Apply risk multipliers
        if patient_data.get('Tobacco Use') == 'Yes':
            risk_factor *= self.risk_multipliers['Tobacco Use']
        
        if patient_data.get('Alcohol Consumption') == 'Yes':
            risk_factor *= self.risk_multipliers['Alcohol Consumption']
        
        if patient_data.get('Betel Quid Use') == 'Yes':
            risk_factor *= self.risk_multipliers['Betel Quid Use']
        
        if patient_data.get('HPV Infection') == 'Yes':
            risk_factor *= self.risk_multipliers['HPV Infection']
        
        if patient_data.get('Poor Oral Hygiene') == 'Yes':
            risk_factor *= self.risk_multipliers['Poor Oral Hygiene']
        
        if patient_data.get('Compromised Immune System') == 'Yes':
            risk_factor *= self.risk_multipliers['Compromised Immune System']
        
        # Age factor
        age = patient_data.get('Age', 50)
        if age > 60:
            risk_factor *= self.risk_multipliers['Age_over_60']
        
        # Gender factor
        if patient_data.get('Gender') == 'Male':
            risk_factor *= self.risk_multipliers['Male']
        
        return risk_factor
    
    def _interpolate_survival(self, baseline, month, risk_adjustment):
        """
        Interpolate survival probability at a given time point
        
        Args:
            baseline: Baseline survival rates dictionary
            month: Time point in months
            risk_adjustment: Risk adjustment factor
            
        Returns:
            Survival probability (0-1)
        """
        # Convert month to years
        years = month / 12.0
        
        # Interpolate between known time points
        if years <= 1:
            base_prob = baseline['1_year']
        elif years <= 3:
            # Linear interpolation between 1 and 3 years
            t = (years - 1) / 2
            base_prob = baseline['1_year'] * (1 - t) + baseline['3_year'] * t
        elif years <= 5:
            # Linear interpolation between 3 and 5 years
            t = (years - 3) / 2
            base_prob = baseline['3_year'] * (1 - t) + baseline['5_year'] * t
        elif years <= 10:
            # Linear interpolation between 5 and 10 years
            t = (years - 5) / 5
            base_prob = baseline['5_year'] * (1 - t) + baseline['10_year'] * t
        else:
            base_prob = baseline['10_year']
        
        # Apply risk adjustment using exponential model
        # S(t) = S_baseline(t) ^ risk_adjustment
        adjusted_prob = base_prob ** risk_adjustment
        
        return max(0, min(1, adjusted_prob))
    
    def _calculate_standard_error(self, prob, month):
        """
        Calculate standard error for confidence interval
        
        Args:
            prob: Survival probability
            month: Time point in months
            
        Returns:
            Standard error
        """
        # Greenwood's formula approximation
        if prob <= 0 or prob >= 1:
            return 0
        
        # Simplified SE calculation
        n_at_risk = max(100 - month, 10)  # Simulated cohort size
        se = np.sqrt(prob * (1 - prob) / n_at_risk)
        
        return se
    
    def _get_survival_at_time(self, target_month, time_points, survival_probs):
        """
        Get survival probability at a specific time point
        
        Args:
            target_month: Target time in months
            time_points: List of time points
            survival_probs: List of survival probabilities
            
        Returns:
            Survival probability at target time
        """
        # Find closest time point
        idx = min(range(len(time_points)), key=lambda i: abs(time_points[i] - target_month))
        return survival_probs[idx]
    
    def _calculate_median_survival(self, time_points, survival_probs):
        """
        Calculate median survival time (time when survival = 50%)
        
        Args:
            time_points: List of time points
            survival_probs: List of survival probabilities
            
        Returns:
            Median survival time in months (or None if >10 years)
        """
        for i, prob in enumerate(survival_probs):
            if prob < 0.5:
                return time_points[i]
        
        return None  # Median survival > 10 years
    
    def compare_with_population(self, patient_curve, cancer_stage):
        """
        Compare patient survival curve with population average
        
        Args:
            patient_curve: Patient's survival curve data
            cancer_stage: Cancer stage
            
        Returns:
            Comparison data
        """
        baseline = self.baseline_survival.get(cancer_stage, self.baseline_survival[0])
        
        # Calculate differences at key time points
        comparison = {
            '1_year': {
                'patient': patient_curve['milestones']['1_year'],
                'population': baseline['1_year'],
                'difference': patient_curve['milestones']['1_year'] - baseline['1_year']
            },
            '3_year': {
                'patient': patient_curve['milestones']['3_year'],
                'population': baseline['3_year'],
                'difference': patient_curve['milestones']['3_year'] - baseline['3_year']
            },
            '5_year': {
                'patient': patient_curve['milestones']['5_year'],
                'population': baseline['5_year'],
                'difference': patient_curve['milestones']['5_year'] - baseline['5_year']
            },
            '10_year': {
                'patient': patient_curve['milestones']['10_year'],
                'population': baseline['10_year'],
                'difference': patient_curve['milestones']['10_year'] - baseline['10_year']
            }
        }
        
        # Overall assessment
        avg_difference = np.mean([
            comparison['1_year']['difference'],
            comparison['3_year']['difference'],
            comparison['5_year']['difference']
        ])
        
        if avg_difference > 0.05:
            assessment = 'Better than average'
        elif avg_difference < -0.05:
            assessment = 'Below average'
        else:
            assessment = 'Similar to average'
        
        return {
            'comparison': comparison,
            'assessment': assessment,
            'avg_difference_percent': round(avg_difference * 100, 1)
        }
    
    def generate_survival_report(self, cancer_stage, patient_data):
        """
        Generate complete survival analysis report
        
        Args:
            cancer_stage: Cancer stage (0-4)
            patient_data: Dictionary of patient features
            
        Returns:
            Complete survival report dictionary
        """
        # Calculate survival curve
        curve = self.calculate_survival_curve(cancer_stage, patient_data)
        
        # Compare with population
        comparison = self.compare_with_population(curve, cancer_stage)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            cancer_stage, 
            patient_data, 
            curve['risk_adjustment_factor']
        )
        
        return {
            'survival_curve': curve,
            'population_comparison': comparison,
            'assessment': comparison['assessment'],  # Add assessment at top level
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, cancer_stage, patient_data, risk_factor):
        """
        Generate personalized recommendations based on survival analysis
        
        Args:
            cancer_stage: Cancer stage
            patient_data: Patient features
            risk_factor: Calculated risk adjustment factor
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Stage-specific recommendations
        if cancer_stage >= 3:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Treatment',
                'recommendation': 'Immediate aggressive treatment recommended (surgery + radiation/chemotherapy)',
                'impact': 'Critical for survival'
            })
        elif cancer_stage >= 1:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Treatment',
                'recommendation': 'Prompt treatment initiation recommended',
                'impact': 'Significant impact on outcomes'
            })
        
        # Lifestyle recommendations
        if patient_data.get('Tobacco Use') == 'Yes':
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Lifestyle',
                'recommendation': 'Immediate tobacco cessation - can improve survival by 20-30%',
                'impact': 'High impact on survival'
            })
        
        if patient_data.get('Betel Quid Use') == 'Yes':
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Lifestyle',
                'recommendation': 'Stop betel quid use immediately - major risk factor',
                'impact': 'High impact on survival'
            })
        
        if patient_data.get('Alcohol Consumption') == 'Yes':
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Lifestyle',
                'recommendation': 'Reduce or eliminate alcohol consumption',
                'impact': 'Moderate impact on survival'
            })
        
        # Follow-up recommendations
        if cancer_stage > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Follow-up',
                'recommendation': 'Regular follow-up every 3 months for first 2 years',
                'impact': 'Early detection of recurrence'
            })
        
        return recommendations


def create_survival_analyzer():
    """
    Factory function to create survival analyzer
    
    Returns:
        SurvivalAnalyzer instance
    """
    return SurvivalAnalyzer()
