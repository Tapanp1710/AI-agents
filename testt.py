import cv2
import numpy as np
import random

# -------------------------
# Agent 1: Face Detection & Matching
# -------------------------
def face_detection_agent(image):
    # Using OpenCV Haar Cascade as a placeholder
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"deepfake_suspect": False, "reason": "No manipulated face detected"}
    else:
        return {"deepfake_suspect": True, "reason": "Face irregularities detected"}


# -------------------------
# Agent 2: Pixel Artifact Detection
# -------------------------
def pixel_artifact_agent(image):
    # Simplified: Use random probability (replace with CNN model)
    prob = random.uniform(0, 1)
    if prob > 0.6:
        return {"deepfake_suspect": True, "reason": "Pixel-level artifacts detected"}
    else:
        return {"deepfake_suspect": False, "reason": "No pixel anomalies"}


# -------------------------
# Agent 3: Frequency Domain Analysis
# -------------------------
def frequency_domain_agent(image):
    # Apply FFT and check high-frequency anomalies
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    anomaly_score = np.mean(magnitude_spectrum)
    if anomaly_score > 200:  # Threshold (placeholder)
        return {"deepfake_suspect": True, "reason": "Frequency anomalies found"}
    else:
        return {"deepfake_suspect": False, "reason": "No frequency anomalies"}


# -------------------------
# Agent 4: Noise & Metadata Agent
# -------------------------
def noise_metadata_agent(image, metadata=None):
    # Placeholder: Random check for noise/metadata mismatch
    prob = random.uniform(0, 1)
    if prob > 0.5:
        return {"authentic": False, "reason": "Metadata/Noise mismatch"}
    else:
        return {"authentic": True, "reason": "Metadata consistent"}


# -------------------------
# Decision-Orchestrator Agent
# -------------------------
def decision_orchestrator(results):
    deepfake_flags = [r.get("deepfake_suspect", False) for r in results]
    auth_flags = [r.get("authentic", True) for r in results]

    final_decision = {}
    if any(deepfake_flags):
        final_decision["Deepfake Detection"] = "Fake"
    else:
        final_decision["Deepfake Detection"] = "Real"

    if all(auth_flags):
        final_decision["Media Authentication"] = "Authentic"
    else:
        final_decision["Media Authentication"] = "Manipulated"

    final_decision["Explanation"] = [r["reason"] for r in results]
    return final_decision


# -------------------------
# Main Function
# -------------------------
if __name__ == "__main__":
    # Load test image
    image_path = "sample5.png"  # Replace with your image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        exit()

    # Run agents
    results = []
    results.append(face_detection_agent(image))
    results.append(pixel_artifact_agent(image))
    results.append(frequency_domain_agent(image))
    results.append(noise_metadata_agent(image))

    # Final decision
    final_output = decision_orchestrator(results)

    print("\n--- Final Output ---")
    for key, value in final_output.items():
        print(f"{key}: {value}")
