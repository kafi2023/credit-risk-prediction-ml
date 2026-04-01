/**
 * Main JavaScript for Credit Risk Prediction Web App
 */

let shapChartInstance = null;

// Initialize app on page load
document.addEventListener('DOMContentLoaded', async function() {
    await checkHealth();
    await loadModels();
    await loadSchema();

    document.getElementById('prediction-form').addEventListener('submit', handleFormSubmit);
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
 * Load available models from the backend
 */
async function loadModels() {
    try {
        const response = await fetch('/models');
        const data = await response.json();
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model.replace('_', ' ').toUpperCase();
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

/**
 * Load form schema from the backend and generate inputs
 */
async function loadSchema() {
    try {
        const response = await fetch('/schema');
        const data = await response.json();
        
        const container = document.getElementById('dynamic-form-fields');
        container.innerHTML = '';

        data.fields.forEach(field => {
            const group = document.createElement('div');
            group.className = 'form-group';
            
            const label = document.createElement('label');
            label.htmlFor = field.name;
            label.textContent = field.label;
            group.appendChild(label);

            if (field.type === 'select') {
                const select = document.createElement('select');
                select.id = field.name;
                select.name = field.name;
                select.className = 'form-control';
                select.required = true;
                
                // Add default empty option
                const defaultOpt = document.createElement('option');
                defaultOpt.value = '';
                defaultOpt.textContent = 'Select...';
                defaultOpt.disabled = true;
                defaultOpt.selected = true;
                select.appendChild(defaultOpt);

                field.options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    select.appendChild(option);
                });
                group.appendChild(select);
            } else if (field.type === 'number') {
                const input = document.createElement('input');
                input.type = 'number';
                input.id = field.name;
                input.name = field.name;
                input.className = 'form-control';
                input.required = true;
                if (field.min !== null) input.min = field.min;
                if (field.max !== null) input.max = field.max;
                group.appendChild(input);
            }

            container.appendChild(group);
        });
    } catch (error) {
        console.error('Failed to load schema:', error);
        document.getElementById('dynamic-form-fields').innerHTML = '<p style="color:red">Failed to load form fields from server.</p>';
    }
}

/**
 * Handle form submit
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert numbers if present
        if (!isNaN(value) && value !== '') {
            data[key] = Number(value);
        } else {
            data[key] = value;
        }
    });

    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.textContent = 'Predicting...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayResults(result);
        } else {
            showError(result.errors.join('<br>'));
        }
    } catch (error) {
        showError('Request failed: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Credit Risk';
    }
}

/**
 * Display prediction results with explanation
 */
function displayResults(data) {
    const resultsCard = document.getElementById('results-card');
    const predictionAlert = document.getElementById('prediction-result');
    
    // Reset classes
    predictionAlert.className = 'prediction-alert';
    
    // Format probability
    const probPct = (data.probability * 100).toFixed(1) + '%';
    
    if (data.prediction_label === 'Good') {
        predictionAlert.classList.add('prediction-good');
        predictionAlert.innerHTML = `LOW RISK (Good Credit)<br><small>Confidence: ${probPct}</small>`;
    } else {
        predictionAlert.classList.add('prediction-bad');
        predictionAlert.innerHTML = `HIGH RISK (Bad Credit)<br><small>Confidence: ${probPct}</small>`;
    }
    
    resultsCard.style.display = 'block';
    
    // Draw SHAP chart
    drawShapChart(data.explanation.feature_contributions);
}

/**
 * Draw SHAP feature contributions using Chart.js
 */
function drawShapChart(contributions) {
    const ctx = document.getElementById('shapChart').getContext('2d');
    
    // If we already have a chart, destroy it to redraw
    if (shapChartInstance) {
        shapChartInstance.destroy();
    }
    
    // Extract labels and data
    // SHAP values are usually returned sorted by magnitude
    // We want to show top 10 for clarity
    const topContributors = contributions.slice(0, 10);
    
    // Reverse so the largest is at the top of a horizontal bar chart
    topContributors.reverse();
    
    const labels = topContributors.map(c => c.feature);
    const data = topContributors.map(c => c.contribution);
    
    // Colors based on positive or negative contribution
    const backgroundColors = data.map(val => 
        val >= 0 ? 'rgba(220, 53, 69, 0.7)' : 'rgba(40, 167, 69, 0.7)' // Red increases risk, Green decreases
    );
    
    const borderColors = data.map(val => 
        val >= 0 ? 'rgb(220, 53, 69)' : 'rgb(40, 167, 69)'
    );

    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on Risk)',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.parsed.x > 0 ? 'Increases Risk' : 'Decreases Risk';
                            return `${label}: ${context.parsed.x.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Display error message to user
 */
function showError(message) {
    alert("Error: " + message);
}
