"""
Orchestrator - Antigravity-style agent coordination
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.vision_agent import VisionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.reasoning_agent import ReasoningAgent
from agents.report_agent import ReportAgent


class Orchestrator:
    """
    Orchestrates all agents in an Antigravity-style architecture
    Coordinates: Vision → Explainability → Reasoning → Report
    """
    
    def __init__(self, gemini_api_key=None):
        print("🚀 Initializing AI Tumor Classification System...")
        print("="*60)
        
        # Initialize agents
        self.vision_agent = VisionAgent()
        self.reasoning_agent = ReasoningAgent(api_key=gemini_api_key)
        self.report_agent = ReportAgent()
        self.explainability_agent = None  # Created after model loads
        
        print("✅ Orchestrator initialized")
    
    def load_model(self):
        """Load the model through vision agent"""
        self.vision_agent.load_model()
        
        # Initialize explainability agent with loaded model
        model = self.vision_agent.get_model_for_gradcam()
        self.explainability_agent = ExplainabilityAgent(model)
        
        print("✅ All agents ready")
    
    def process_image(self, image_path, generate_gradcam=True, generate_llm=True):
        """
        Main orchestration flow
        
        Args:
            image_path: Path to MRI image
            generate_gradcam: Whether to generate Grad-CAM
            generate_llm: Whether to use LLM for explanation
            
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}\n")
        
        # Step 1: Vision Agent - Classification
        print("🔍 Step 1: Running classification...")
        prediction_results = self.vision_agent.predict(image_path)
        print(f"   ✅ Predicted: {prediction_results['predicted_class']} ({prediction_results['confidence']:.1f}%)")
        
        # Step 2: Explainability Agent - Grad-CAM
        gradcam_results = None
        gradcam_error = None
        if generate_gradcam and self.explainability_agent:
            print("\n🎨 Step 2: Generating Grad-CAM visualization...")
            try:
                gradcam_results = self.explainability_agent.explain(
                    prediction_results['preprocessed_tensor'],
                    prediction_results['original_image'],
                    prediction_results['predicted_idx']
                )
                print("   ✅ Grad-CAM generated")
            except Exception as e:
                print(f"   ⚠️  Grad-CAM failed: {e}")
                gradcam_error = str(e)
        
        # Step 3: Reasoning Agent - Explanation
        print("\n🧠 Step 3: Generating explanation...")
        explanation = self.reasoning_agent.generate_explanation(
            prediction_results['predicted_class'],
            prediction_results['confidence'],
            prediction_results['probabilities'],
            prediction_results['class_names'],
            image_path=image_path  # Pass the actual image for Gemini Vision
        )
        print("   ✅ Explanation generated")
        
        # Step 4: Report Agent - Structured Report
        print("\n📄 Step 4: Creating report...")
        report = self.report_agent.generate_report(prediction_results, explanation)
        summary = self.report_agent.generate_summary(prediction_results)
        print("   ✅ Report created")
        
        print(f"\n{'='*60}")
        print("✅ Processing complete!")
        print(f"{'='*60}\n")
        
        return {
            'prediction': prediction_results,
            'gradcam': gradcam_results,
            'gradcam_error': gradcam_error,
            'explanation': explanation,
            'report': report,
            'summary': summary
        }
    
    def generate_batch_report(self, results_list, mode='batch', patient_id=None):
        """Delegate batch report generation to report agent"""
        return self.report_agent.generate_batch_report(results_list, mode=mode, patient_id=patient_id)

    def synthesize_patient_report(self, results_list):
        """
        Synthesize a holistic patient report from multiple view results.
        If minimal conflict, calls Reasoning Agent.
        If significant conflict, SKIPS Reasoning Agent as per strict user rule.
        """
        if not results_list:
            return "No scans to analyze."

        # 1. Check for Diagnostic Conflict
        class_counts = {}
        for res in results_list:
            pred = res['prediction']['predicted_class']
            class_counts[pred] = class_counts.get(pred, 0) + 1
            
        unique_diagnoses = len(class_counts)
        
        # STRICT RULE: If conflict exists (more than 1 diagnosis type), DO NOT call Reasoning Agent.
        if unique_diagnoses > 1:
            return {
                "conflict_detected": True,
                "summary": "Diagnostic Conflict Detected. AI Interpretation Halted.",
                "explanation": None, 
                "class_counts": class_counts
            }

        # 2. If no conflict, proceed with synthesis
        try:
            return self.reasoning_agent.synthesize_case(results_list)
        except Exception as e:
            return f"Error synthesizing case: {str(e)}"


# Test function
def test_orchestrator():
    """Test the orchestrator with a sample image"""
    print("Testing Orchestrator...")
    print("="*60)
    
    orchestrator = Orchestrator()
    orchestrator.load_model()
    
    print("\n✅ Orchestrator is ready!")
    print("   All agents loaded and coordinated")
    print("\n   To process an image, call:")
    print("   results = orchestrator.process_image('path/to/image.jpg')")


if __name__ == "__main__":
    test_orchestrator()
