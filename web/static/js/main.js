/**
 * Main JavaScript for Credit Risk Prediction Web App
 */

// Check API health on page load
document.addEventListener('DOMContentLoaded', function() {
    checkHealth();
});

/**
 * Check if the backend API is healthy
 */
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API health check failed:', error);
    }
}

/**
 * Make a prediction (to be implemented in Milestone 2)
 */
async function makePrediction(formData) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Prediction failed:', error);
        throw error;
    }
}

/**
 * Display prediction results with explanation
 */
function displayResults(prediction, explanation) {
    // To be implemented in Milestone 2
    console.log('Prediction:', prediction);
    console.log('Explanation:', explanation);
}

/**
 * Display error message to user
 */
function showError(message) {
    console.error(message);
    // TODO: Add user-friendly error display
}
