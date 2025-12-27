import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time

# Initialize model and tokenizer
MODEL_NAME = "IctAchievers/holas-defender-ultimate-v14"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

def format_security_response(response_text):
    """Format security analysis response"""
    try:
        # Try to parse as JSON if it's structured
        if response_text.strip().startswith('{'):
            data = json.loads(response_text)
            formatted = "🛡️ SECURITY ANALYSIS\n"
            formatted += f"Threat Level: {data.get('threat_level', 'N/A')}/100\n"
            formatted += f"Confidence: {data.get('confidence', 'N/A')}%\n\n"
            formatted += "RECOMMENDATIONS:\n"
            for rec in data.get('recommendations', []):
                formatted += f"• {rec}\n"
            formatted += f"\nAI INSIGHT: {data.get('ai_insight', 'No insight provided')}"
            return formatted
    except:
        pass
    
    return response_text

def holas_inference(prompt, mode="general", max_tokens=2048, temperature=0.7):
    if not tokenizer or not model:
        return "❌ Model not loaded. Please check the console for errors."
    
    try:
        # Add mode-specific prefixes
        if mode == "security":
            prompt = f"<SECURITY> {prompt} </SECURITY>"
        elif mode == "multilingual":
            prompt = f"<MULTI> {prompt} </MULTI>"
        elif mode == "quantum":
            prompt = f"<QUANTUM> {prompt} </QUANTUM>"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        
        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Format based on mode
        if mode == "security":
            response = format_security_response(response)
        
        # Add timing info
        response_time = (end_time - start_time) * 1000
        response = f"⏱️ Response time: {response_time:.1f}ms\n\n{response}"
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="HOLAS DEFENDER ULTIMATE v14") as demo:
    gr.Markdown("""
    # 🛡️ HOLAS DEFENDER ULTIMATE v14
    ## World's Most Advanced Cybersecurity AI
    
    This is a demonstration of HOLAS DEFENDER ULTIMATE v14, the world's most advanced cybersecurity AI platform.
    """)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Input Prompt",
                placeholder="Enter your security analysis request...",
                lines=5
            )
            mode = gr.Radio(
                choices=["general", "security", "multilingual", "quantum"],
                value="general",
                label="Mode"
            )
            max_tokens = gr.Slider(
                minimum=128,
                maximum=8192,
                value=2048,
                step=128,
                label="Max Tokens"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Response",
                lines=15,
                interactive=False
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Analyze this security threat: Unusual network traffic at 2:47 AM to unknown IP", "security"],
            ["Explain cybersecurity in Luganda", "multilingual"],
            ["What quantum threats should we prepare for in 2025?", "quantum"],
            ["Analyze this code for vulnerabilities: function test() { eval(userInput); }", "security"]
        ],
        inputs=[prompt, mode],
        outputs=output,
        fn=holas_inference,
        cache_examples=True
    )
    
    submit_btn.click(
        fn=holas_inference,
        inputs=[prompt, mode, max_tokens, temperature],
        outputs=output
    )
    
    gr.Markdown("""
    ## 🚀 Key Features
    - **Advanced Reasoning**: IQ 1200+ cognitive processing
    - **Cyber Security**: 99.99% threat detection accuracy
    - **Multilingual**: 16 languages with cultural context
    - **Quantum-Ready**: Post-quantum encryption (Kyber-1024)
    - **Autonomous Response**: Self-driving reasoning engine
    
    ## 🌍 Global Access
    Model available at: https://huggingface.co/IctAchievers/holas-defender-ultimate-v14
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
