"""
Temporal Comparison Module for NeuroPlex
Compares current scan with previous scans to track progression
"""

import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import cv2

class TemporalComparator:
    """
    Compares medical images over time to track disease progression
    """
    
    def __init__(self):
        """
        Initialize temporal comparator
        """
        pass
    
    def compare_with_history(self, current_test, previous_tests):
        """
        Compare current test with previous tests
        
        Args:
            current_test: Current test data dictionary
            previous_tests: List of previous test dictionaries
            
        Returns:
            Comparison analysis dictionary
        """
        if not previous_tests or len(previous_tests) == 0:
            return {
                'has_history': False,
                'message': 'No previous tests available for comparison'
            }
        
        # Sort previous tests by timestamp (most recent first)
        sorted_tests = sorted(
            previous_tests, 
            key=lambda x: x.get('timestamp', datetime.min),
            reverse=True
        )
        
        # Compare with most recent test
        most_recent = sorted_tests[0]
        
        # Calculate time difference
        time_diff = self._calculate_time_difference(
            current_test.get('timestamp'),
            most_recent.get('timestamp')
        )
        
        # Compare predictions
        prediction_change = self._compare_predictions(current_test, most_recent)
        
        # Compare confidence levels
        confidence_change = self._compare_confidence(current_test, most_recent)
        
        # Compare cancer stages (if available)
        stage_change = self._compare_stages(current_test, most_recent)
        
        # Calculate progression velocity
        progression_velocity = self._calculate_progression_velocity(
            current_test, 
            most_recent, 
            time_diff
        )
        
        # Generate trend analysis
        trend_analysis = self._analyze_trend(current_test, sorted_tests)
        
        # Compare images (if available)
        image_comparison = self._compare_images(current_test, most_recent)
        
        return {
            'has_history': True,
            'comparison_count': len(previous_tests),
            'most_recent_test': {
                'timestamp': most_recent.get('timestamp'),
                'days_ago': time_diff['days']
            },
            'prediction_change': prediction_change,
            'confidence_change': confidence_change,
            'stage_change': stage_change,
            'progression_velocity': progression_velocity,
            'trend_analysis': trend_analysis,
            'image_comparison': image_comparison,
            'time_difference': time_diff
        }
    
    def _calculate_time_difference(self, current_time, previous_time):
        """
        Calculate time difference between two timestamps
        
        Args:
            current_time: Current timestamp
            previous_time: Previous timestamp
            
        Returns:
            Dictionary with time difference in various units
        """
        try:
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            if isinstance(previous_time, str):
                previous_time = datetime.fromisoformat(previous_time.replace('Z', '+00:00'))
            
            diff = current_time - previous_time
            days = diff.days
            months = days / 30.44  # Average days per month
            years = days / 365.25
            
            return {
                'days': days,
                'months': round(months, 1),
                'years': round(years, 2),
                'formatted': self._format_time_difference(days)
            }
        except Exception as e:
            print(f"[TEMPORAL] Error calculating time difference: {e}")
            return {
                'days': 0,
                'months': 0,
                'years': 0,
                'formatted': 'Unknown'
            }
    
    def _format_time_difference(self, days):
        """
        Format time difference in human-readable format
        
        Args:
            days: Number of days
            
        Returns:
            Formatted string
        """
        if days < 1:
            return 'Today'
        elif days == 1:
            return '1 day ago'
        elif days < 30:
            return f'{days} days ago'
        elif days < 365:
            months = round(days / 30.44)
            return f'{months} month{"s" if months > 1 else ""} ago'
        else:
            years = round(days / 365.25, 1)
            return f'{years} year{"s" if years > 1 else ""} ago'
    
    def _compare_predictions(self, current, previous):
        """
        Compare prediction results
        
        Args:
            current: Current test data
            previous: Previous test data
            
        Returns:
            Prediction comparison dictionary
        """
        current_pred = current.get('final_prediction', 'Unknown')
        previous_pred = previous.get('final_prediction', 'Unknown')
        
        if current_pred == previous_pred:
            status = 'Stable'
            description = f'Prediction remains {current_pred}'
            severity = 'info'
        elif previous_pred == 'Normal' and current_pred == 'OSCC':
            status = 'Worsened'
            description = 'New OSCC detection (was Normal)'
            severity = 'critical'
        elif previous_pred == 'OSCC' and current_pred == 'Normal':
            status = 'Improved'
            description = 'No longer detecting OSCC (was OSCC)'
            severity = 'positive'
        else:
            status = 'Changed'
            description = f'Changed from {previous_pred} to {current_pred}'
            severity = 'warning'
        
        return {
            'current': current_pred,
            'previous': previous_pred,
            'status': status,
            'description': description,
            'severity': severity
        }
    
    def _compare_confidence(self, current, previous):
        """
        Compare confidence levels
        
        Args:
            current: Current test data
            previous: Previous test data
            
        Returns:
            Confidence comparison dictionary
        """
        current_conf = current.get('final_confidence', 0)
        previous_conf = previous.get('final_confidence', 0)
        
        diff = current_conf - previous_conf
        percent_change = (diff / previous_conf * 100) if previous_conf > 0 else 0
        
        if abs(diff) < 5:
            status = 'Stable'
            description = 'Confidence level unchanged'
        elif diff > 0:
            status = 'Increased'
            description = f'Confidence increased by {abs(diff):.1f}%'
        else:
            status = 'Decreased'
            description = f'Confidence decreased by {abs(diff):.1f}%'
        
        return {
            'current': current_conf,
            'previous': previous_conf,
            'difference': round(diff, 1),
            'percent_change': round(percent_change, 1),
            'status': status,
            'description': description
        }
    
    def _compare_stages(self, current, previous):
        """
        Compare cancer stages
        
        Args:
            current: Current test data
            previous: Previous test data
            
        Returns:
            Stage comparison dictionary
        """
        current_stage = None
        previous_stage = None
        
        if current.get('stage2_tabular'):
            current_stage = current['stage2_tabular'].get('cancer_stage')
        
        if previous.get('stage2_tabular'):
            previous_stage = previous['stage2_tabular'].get('cancer_stage')
        
        if current_stage is None or previous_stage is None:
            return {
                'available': False,
                'message': 'Stage information not available for comparison'
            }
        
        diff = current_stage - previous_stage
        
        if diff == 0:
            status = 'Stable'
            description = f'Stage remains at {current_stage}'
            severity = 'info'
        elif diff > 0:
            status = 'Progressed'
            description = f'Stage increased from {previous_stage} to {current_stage}'
            severity = 'critical'
        else:
            status = 'Regressed'
            description = f'Stage decreased from {previous_stage} to {current_stage}'
            severity = 'positive'
        
        return {
            'available': True,
            'current': current_stage,
            'previous': previous_stage,
            'difference': diff,
            'status': status,
            'description': description,
            'severity': severity
        }
    
    def _calculate_progression_velocity(self, current, previous, time_diff):
        """
        Calculate rate of disease progression
        
        Args:
            current: Current test data
            previous: Previous test data
            time_diff: Time difference dictionary
            
        Returns:
            Progression velocity dictionary
        """
        if time_diff['days'] == 0:
            return {
                'available': False,
                'message': 'Cannot calculate velocity (same day)'
            }
        
        # Calculate confidence change rate
        current_conf = current.get('final_confidence', 0)
        previous_conf = previous.get('final_confidence', 0)
        conf_change_per_month = (current_conf - previous_conf) / max(time_diff['months'], 0.1)
        
        # Calculate stage change rate (if available)
        stage_velocity = None
        if current.get('stage2_tabular') and previous.get('stage2_tabular'):
            current_stage = current['stage2_tabular'].get('cancer_stage', 0)
            previous_stage = previous['stage2_tabular'].get('cancer_stage', 0)
            stage_change = current_stage - previous_stage
            stage_velocity = stage_change / max(time_diff['months'], 0.1)
        
        # Interpret velocity
        if abs(conf_change_per_month) < 2:
            interpretation = 'Slow/Stable progression'
            risk = 'Low'
        elif abs(conf_change_per_month) < 5:
            interpretation = 'Moderate progression rate'
            risk = 'Medium'
        else:
            interpretation = 'Rapid progression'
            risk = 'High'
        
        return {
            'available': True,
            'confidence_change_per_month': round(conf_change_per_month, 2),
            'stage_velocity': round(stage_velocity, 3) if stage_velocity else None,
            'interpretation': interpretation,
            'risk_level': risk
        }
    
    def _analyze_trend(self, current, all_tests):
        """
        Analyze overall trend across multiple tests
        
        Args:
            current: Current test data
            all_tests: List of all previous tests (sorted by time)
            
        Returns:
            Trend analysis dictionary
        """
        if len(all_tests) < 2:
            return {
                'available': False,
                'message': 'Need at least 2 previous tests for trend analysis'
            }
        
        # Include current test in analysis
        all_tests_with_current = [current] + all_tests
        
        # Extract confidence values over time
        confidences = [t.get('final_confidence', 0) for t in all_tests_with_current]
        
        # Calculate trend direction
        if len(confidences) >= 3:
            # Linear regression to find trend
            x = np.arange(len(confidences))
            y = np.array(confidences)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 2:
                trend = 'Worsening'
                description = 'Confidence levels increasing over time'
                severity = 'warning'
            elif slope < -2:
                trend = 'Improving'
                description = 'Confidence levels decreasing over time'
                severity = 'positive'
            else:
                trend = 'Stable'
                description = 'Confidence levels relatively stable'
                severity = 'info'
        else:
            trend = 'Insufficient data'
            description = 'Need more tests for reliable trend'
            severity = 'info'
            slope = 0
        
        # Count OSCC detections
        oscc_count = sum(1 for t in all_tests_with_current if t.get('final_prediction') == 'OSCC')
        normal_count = len(all_tests_with_current) - oscc_count
        
        return {
            'available': True,
            'trend': trend,
            'description': description,
            'severity': severity,
            'slope': round(slope, 2),
            'total_tests': len(all_tests_with_current),
            'oscc_detections': oscc_count,
            'normal_detections': normal_count,
            'confidence_values': confidences
        }
    
    def _compare_images(self, current, previous):
        """
        Compare histopathology images
        
        Args:
            current: Current test data
            previous: Previous test data
            
        Returns:
            Image comparison dictionary
        """
        current_img_b64 = current.get('original_image')
        previous_img_b64 = previous.get('original_image')
        
        if not current_img_b64 or not previous_img_b64:
            return {
                'available': False,
                'message': 'Images not available for comparison'
            }
        
        try:
            # Decode base64 images
            current_img = self._decode_base64_image(current_img_b64)
            previous_img = self._decode_base64_image(previous_img_b64)
            
            # Resize to same size for comparison
            target_size = (240, 240)
            current_img = cv2.resize(current_img, target_size)
            previous_img = cv2.resize(previous_img, target_size)
            
            # Calculate structural similarity
            similarity = self._calculate_image_similarity(current_img, previous_img)
            
            # Generate difference map
            diff_map = self._generate_difference_map(current_img, previous_img)
            
            # Encode difference map to base64
            diff_map_b64 = self._encode_image_to_base64(diff_map)
            
            return {
                'available': True,
                'similarity_score': round(similarity, 3),
                'difference_map': diff_map_b64,
                'interpretation': self._interpret_similarity(similarity)
            }
            
        except Exception as e:
            print(f"[TEMPORAL] Error comparing images: {e}")
            return {
                'available': False,
                'message': f'Error comparing images: {str(e)}'
            }
    
    def _decode_base64_image(self, base64_str):
        """
        Decode base64 string to numpy array
        
        Args:
            base64_str: Base64 encoded image string
            
        Returns:
            Numpy array (BGR format)
        """
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    
    def _calculate_image_similarity(self, img1, img2):
        """
        Calculate similarity between two images using SSIM
        
        Args:
            img1: First image (numpy array)
            img2: Second image (numpy array)
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean squared error (simple similarity metric)
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        # Convert MSE to similarity score (0-1)
        max_mse = 255 ** 2  # Maximum possible MSE
        similarity = 1 - (mse / max_mse)
        
        return similarity
    
    def _generate_difference_map(self, img1, img2):
        """
        Generate visual difference map between two images
        
        Args:
            img1: Current image
            img2: Previous image
            
        Returns:
            Difference map (numpy array)
        """
        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Convert to grayscale
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply colormap for visualization
        diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        
        # Create side-by-side comparison
        h, w = img1.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = img2  # Previous
        comparison[:, w:2*w] = img1  # Current
        comparison[:, 2*w:] = diff_colored  # Difference
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Previous', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, 'Current', (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, 'Difference', (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        return comparison
    
    def _encode_image_to_base64(self, img_array):
        """
        Encode numpy array to base64 string
        
        Args:
            img_array: Numpy array (BGR format)
            
        Returns:
            Base64 encoded string with data URI prefix
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)
        
        # Encode to base64
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _interpret_similarity(self, similarity):
        """
        Interpret similarity score
        
        Args:
            similarity: Similarity score (0-1)
            
        Returns:
            Interpretation string
        """
        if similarity > 0.9:
            return 'Very similar - minimal changes detected'
        elif similarity > 0.75:
            return 'Similar - minor changes detected'
        elif similarity > 0.5:
            return 'Moderate changes detected'
        else:
            return 'Significant changes detected'


def create_temporal_comparator():
    """
    Factory function to create temporal comparator
    
    Returns:
        TemporalComparator instance
    """
    return TemporalComparator()
