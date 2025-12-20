# System Diagrams

This folder contains all the system architecture and design diagrams for the Credit Risk Prediction ML project.

## Diagrams Overview

### 1. System Architecture Diagram
**File:** `System Architecture Diagram.png`

Shows the complete system architecture with all layers:
- User Interface Layer
- Application Layer (Flask)
- ML Pipeline Layer
- Explainability Layer
- Data Layer

### 2. User Flow Diagram
**File:** `User Flow Diagram.png`

Illustrates the complete user journey:
- Landing on the website
- Entering applicant information
- Receiving predictions and explanations
- Viewing visualizations
- Decision points and actions

### 3. Class Diagram (UML)
**File:** `Class Daigram.png`

Object-oriented design showing:
- DataPreprocessor
- ModelTrainer
- Predictor
- Explainer
- FlaskApp
- CreditRiskModel

### 4. Sequence Diagram - Prediction Flow
**File:** `Sequence Diagram - Prediction Flow.png`

Step-by-step interaction between components during prediction:
- User input
- Backend processing
- Model prediction
- Explanation generation
- Result display

### 5. Data Flow Diagram
**File:** `Data Flow.png`

Shows how data flows through the system:
- Data Collection
- Data Processing
- Model Training
- Model Evaluation
- Deployment

### 6. Component Diagram
**File:** `component diagram.png`

Displays all system components and their relationships:
- Frontend Components
- Backend Components
- Data Components
- ML Components

### 7. State Diagram - Prediction Lifecycle
**File:** `state diagram.png`

Shows different states during prediction:
- Idle
- Input Received
- Validating
- Processing
- Predicting
- Explaining
- Displaying Results
- Error States

## Usage

These diagrams are used for:
- **Thesis Documentation**: Visual representation of system design
- **Development Reference**: Understanding component interactions
- **Stakeholder Communication**: Explaining system architecture
- **Code Implementation**: Guiding development based on design

## Mermaid Source

All diagrams were created using Mermaid diagram syntax. The source code is documented in the project for future modifications.

---

**Created:** December 20, 2025  
**Project:** Credit Risk Prediction with Explainable AI  
**Student:** Kafi MD Abdullah Hel (N06WMD)
