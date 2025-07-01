/**
 * ========================================
 * PNEUMONIA AI - ADVANCED ANALYZER
 * Professional Medical Analysis Interface
 * Version: 2.0.0
 * ========================================
 */

'use strict';

/**
 * Advanced Pneumonia Analysis System
 * Extends MedicalCore with specialized pneumonia detection capabilities
 */
class PneumoniaAnalyzer extends MedicalCore.UIComponent {
    
    constructor(element, options = {}) {
        super(element, options);
        
        this.analysisQueue = [];
        this.currentAnalysis = null;
        this.analysisHistory = [];
        this.sessionStats = {
            totalAnalyses: 0,
            successfulAnalyses: 0,
            averageConfidence: 0,
            averageProcessingTime: 0,
            pneumoniaDetected: 0,
            normalResults: 0
        };
        
        this.initializeComponents();
        this.loadAnalysisHistory();
    }

    get defaultOptions() {
        return {
            maxFileSize: MedicalCore.config.get('upload.maxFileSize'),
            allowedTypes: MedicalCore.config.get('upload.allowedTypes'),
            autoAnalyze: true,
            showProgress: true,
            enableHistory: true,
            confidenceThreshold: MedicalCore.config.get('analysis.confidenceThreshold')
        };
    }

    initializeComponents() {
        this.initializeUploadZone();
        this.initializeProgressSystem();
        this.initializeResultsDisplay();
        this.initializeHistoryPanel();
        this.initializeStatsPanel();
        this.setupValidation();
    }

    initializeUploadZone() {
        this.uploadZone = this.element;
        this.fileInput = document.getElementById('medicalFileInput');
        
        if (!this.uploadZone || !this.fileInput) {
            throw new Error('Upload zone or file input not found');
        }

        // Enhanced drag and drop
        this.uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadZone.addEventListener('drop', this.handleDrop.bind(this));
        this.uploadZone.addEventListener('click', (e) => {
            // Only trigger file input if clicking on the zone itself, not on buttons
            if (e.target === this.uploadZone || e.target.closest('#uploadContent')) {
                this.fileInput.click();
            }
        });
        
        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
    }

    initializeProgressSystem() {
        this.progressContainer = document.querySelector('.progress-container');
        this.progressBar = document.getElementById('medicalProgressBar');
        this.progressText = document.getElementById('progressText');
        this.analysisStatus = document.getElementById('analysisStatus');
    }

    initializeResultsDisplay() {
        this.resultsContainer = document.getElementById('resultsContainer');
        this.resultCard = document.querySelector('.result-card');
        this.confidenceDisplay = document.querySelector('.confidence-display');
        this.predictionResult = document.querySelector('.prediction-result');
        this.analysisDetails = document.querySelector('.analysis-details');
    }

    initializeHistoryPanel() {
        this.historyContainer = document.querySelector('.history-container');
        this.historyList = document.querySelector('.history-list');
        this.clearHistoryBtn = document.querySelector('.clear-history-btn');
        
        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', this.clearHistory.bind(this));
        }
    }

    initializeStatsPanel() {
        this.statsContainer = document.querySelector('.stats-container');
        this.updateStatsDisplay();
    }

    setupValidation() {
        // File validation rules
        MedicalCore.validation.addRule('file', 
            MedicalCore.ValidationManager.validators.required, 
            'Please select a file to analyze'
        );
        
        MedicalCore.validation.addRule('file', 
            MedicalCore.ValidationManager.validators.isFile, 
            'Invalid file selection'
        );
        
        MedicalCore.validation.addRule('file', 
            MedicalCore.ValidationManager.validators.fileSize(this.options.maxFileSize), 
            `File size must be less than ${MedicalCore.utils.formatFileSize(this.options.maxFileSize)}`
        );
        
        MedicalCore.validation.addRule('file', 
            MedicalCore.ValidationManager.validators.fileType(this.options.allowedTypes), 
            'File must be a valid medical image (JPEG, PNG, BMP, or TIFF)'
        );
    }

    // File Handling Methods
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.add('dragover');
        this.uploadZone.style.borderColor = 'var(--medical-success)';
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
        this.uploadZone.style.borderColor = '';
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
        this.uploadZone.style.borderColor = '';
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        try {
            // Validate file
            const validation = MedicalCore.validation.validate({ file });
            if (!validation.isValid) {
                const errorMessage = Object.values(validation.errors).flat().join(', ');
                MedicalCore.notifications.error(errorMessage, { title: 'File Validation Error' });
                return;
            }

            // Create analysis task
            const analysisTask = {
                id: MedicalCore.utils.generateId(),
                file,
                fileName: file.name,
                fileSize: file.size,
                timestamp: Date.now(),
                status: 'queued'
            };

            // Add to queue
            this.analysisQueue.push(analysisTask);
            
            // Process if auto-analyze is enabled
            if (this.options.autoAnalyze && !this.currentAnalysis) {
                await this.processAnalysisQueue();
            }

            this.emit('fileAdded', { task: analysisTask });
            
        } catch (error) {
            MedicalCore.errors.handle(error, { operation: 'file-processing' });
        }
    }

    async processAnalysisQueue() {
        if (this.analysisQueue.length === 0 || this.currentAnalysis) {
            return;
        }

        const task = this.analysisQueue.shift();
        this.currentAnalysis = task;
        
        try {
            await this.analyzeImage(task);
        } catch (error) {
            MedicalCore.errors.handle(error, { operation: 'image-analysis', taskId: task.id });
        } finally {
            this.currentAnalysis = null;
            // Process next in queue
            setTimeout(() => this.processAnalysisQueue(), 100);
        }
    }

    async analyzeImage(task) {
        const startTime = Date.now();
        
        try {
            // Update UI state
            this.setState({ analyzing: true, currentTask: task });
            this.showProgress(true);
            this.updateAnalysisStatus('Preparing image for analysis...');
            
            // Create FormData
            const formData = new FormData();
            formData.append('image', task.file);
            
            // Add model selection if available
            const modelSelect = document.getElementById('medicalModelSelect');
            if (modelSelect) {
                formData.append('model_name', modelSelect.value);
            }
            
            // Add enhancement option if available
            const enhancementCheck = document.getElementById('medicalEnhancement');
            if (enhancementCheck) {
                formData.append('enhance_image', enhancementCheck.checked);
            }
            
            // Add ensemble mode if available
            const ensembleCheck = document.getElementById('ensembleMode');
            if (ensembleCheck) {
                formData.append('use_ensemble', ensembleCheck.checked);
            }
            
            // Add visualization request if available
            const visualizationCheck = document.getElementById('generateVisualization');
            if (visualizationCheck) {
                formData.append('generate_visualization', visualizationCheck.checked);
            }
            
            // Show upload progress
            await this.simulateProgress(0, 30, 'Uploading image...');
            
            // Make API call
            const response = await MedicalCore.api.analyzeImage(formData);
            
            if (!response.success) {
                throw new Error(response.error || 'Analysis failed');
            }

            // Process analysis - handle Django response format
            await this.simulateProgress(30, 80, 'Analyzing medical image...');
            
            const analysis = response.data.analysis; // Django returns nested analysis object
            const endTime = Date.now();
            const processingTime = analysis.processing_time ? analysis.processing_time * 1000 : endTime - startTime;
            
            // Create analysis record matching Django format
            const analysisRecord = {
                ...task,
                result: analysis,
                processingTime,
                completedAt: endTime,
                status: 'completed'
            };
            
            // Complete progress
            await this.simulateProgress(80, 100, 'Analysis complete!');
            
            // Update history and stats
            this.addToHistory(analysisRecord);
            this.updateSessionStats(analysisRecord);
            
            // Display results
            await this.displayResults(analysisRecord);
            
            // Show success notification
            const message = response.data.message || 'Analysis completed successfully';
            MedicalCore.notifications.success(message, { title: 'Analysis Complete' });
            
            this.emit('analysisComplete', { record: analysisRecord });
            
        } catch (error) {
            task.status = 'failed';
            task.error = error.message;
            this.showProgress(false);
            MedicalCore.errors.handle(error, { operation: 'image-analysis', taskId: task.id });
        } finally {
            this.setState({ analyzing: false, currentTask: null });
        }
    }

    async simulateProgress(start, end, statusText) {
        return new Promise((resolve) => {
            let current = start;
            const increment = (end - start) / 20;
            const interval = 50;
            
            // Update processing steps based on progress
            this.updateProcessingSteps(start, end);
            
            const timer = setInterval(() => {
                current += increment;
                this.updateProgress(Math.min(current, end), statusText);
                this.updateProcessingSteps(current, end);
                
                if (current >= end) {
                    clearInterval(timer);
                    resolve();
                }
            }, interval);
        });
    }

    updateProcessingSteps(current, end) {
        const steps = document.querySelectorAll('.processing-step');
        if (steps.length === 0) return;
        
        steps.forEach((step, index) => {
            step.classList.remove('active', 'completed');
        });
        
        if (current <= 30) {
            // Upload phase
            steps[0]?.classList.add('active');
        } else if (current <= 60) {
            // Preprocessing phase
            steps[0]?.classList.add('completed');
            steps[1]?.classList.add('active');
        } else if (current <= 90) {
            // AI Analysis phase
            steps[0]?.classList.add('completed');
            steps[1]?.classList.add('completed');
            steps[2]?.classList.add('active');
        } else {
            // Results phase
            steps[0]?.classList.add('completed');
            steps[1]?.classList.add('completed');
            steps[2]?.classList.add('completed');
            steps[3]?.classList.add('active');
        }
    }

    updateProgress(percentage, status) {
        if (this.progressBar) {
            this.progressBar.style.width = `${percentage}%`;
        }
        
        if (this.progressText) {
            this.progressText.textContent = `${Math.round(percentage)}%`;
        }
        
        if (this.analysisStatus && status) {
            this.analysisStatus.textContent = status;
        }
    }

    updateAnalysisStatus(status) {
        if (this.analysisStatus) {
            this.analysisStatus.textContent = status;
        }
    }

    showProgress(show) {
        const uploadContent = document.getElementById('uploadContent');
        const processingContent = document.getElementById('processingContent');
        
        if (uploadContent && processingContent) {
            if (show) {
                uploadContent.style.display = 'none';
                processingContent.style.display = 'block';
            } else {
                uploadContent.style.display = 'block';
                processingContent.style.display = 'none';
            }
        }
    }

    async displayResults(analysisRecord) {
        if (!this.resultsContainer) return;
        
        const { result, processingTime, fileName } = analysisRecord;
        
        // Animate results in
        MedicalCore.animations.fadeIn(this.resultsContainer);
        
        // Create result card HTML
        const resultHtml = this.generateResultHTML(result, processingTime, fileName);
        
        if (this.resultCard) {
            this.resultCard.innerHTML = resultHtml;
            this.resultCard.classList.add('medical-animate-scale-in');
        }
        
        // Update individual result components
        this.updateConfidenceDisplay(result);
        this.updatePredictionResult(result);
        this.updateAnalysisDetails(result, processingTime);
        
        // Handle visualization if available
        this.updateVisualizationDisplay(result, fileName);
        
        // Handle pathology analysis if available
        this.updatePathologyDisplay(result);
        
        // Show results container
        this.resultsContainer.style.display = 'block';
        
        // Show success notification
        const diagnosis = result.prediction === 'PNEUMONIA' ? 'Pneumonia detected' : 'Normal chest X-ray';
        MedicalCore.notifications.success(
            `Analysis complete: ${diagnosis} (${(result.confidence * 100).toFixed(1)}% confidence)`,
            { title: 'Analysis Results', duration: 8000 }
        );
    }

    generateResultHTML(result, processingTime, fileName) {
        // Handle Django API response format
        const isPneumonia = result.prediction_class === 'pneumonia' || result.is_pneumonia_detected;
        const confidence = result.confidence_percentage || (result.confidence_score * 100).toFixed(1);
        const cardClass = isPneumonia ? 'medical-card-danger' : 'medical-card-success';
        const iconClass = isPneumonia ? 'text-danger' : 'text-success';
        const statusIcon = isPneumonia ? '⚠️' : '✅';
        const diagnosisText = isPneumonia ? 'PNEUMONIA DETECTED' : 'NORMAL';
        
        return `
            <div class="medical-card ${cardClass}">
                <div class="medical-card-header">
                    <div class="d-flex align-items-center justify-content-between">
                        <div>
                            <h4 class="medical-heading-3 mb-1">${statusIcon} ${diagnosisText}</h4>
                            <p class="medical-caption mb-0">Analysis of ${fileName}</p>
                        </div>
                        <div class="text-end">
                            <div class="medical-stat-value ${iconClass}">${confidence}%</div>
                            <div class="medical-stat-label">Confidence</div>
                        </div>
                    </div>
                </div>
                
                <div class="medical-card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="medical-label">Diagnosis</label>
                                <div class="medical-diagnosis-text ${iconClass}">
                                    ${isPneumonia ? 'Pneumonia Detected' : 'Normal Chest X-ray'}
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label class="medical-label">Confidence Level</label>
                                <div class="medical-progress mb-2">
                                    <div class="medical-progress-bar" style="width: ${confidence}%;"></div>
                                </div>
                                <small class="medical-caption">${this.getConfidenceDescription(result.confidence_score || confidence / 100)}</small>
                            </div>
                            
                            ${result.raw_probabilities ? `
                            <div class="mb-3">
                                <label class="medical-label">Probability Breakdown</label>
                                <div class="probability-bars">
                                    <div class="d-flex justify-content-between mb-1">
                                        <span>Normal:</span>
                                        <span>${(result.raw_probabilities.normal * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="medical-progress mb-2">
                                        <div class="medical-progress-bar" style="width: ${result.raw_probabilities.normal * 100}%; background: var(--medical-success);"></div>
                                    </div>
                                    
                                    <div class="d-flex justify-content-between mb-1">
                                        <span>Pneumonia:</span>
                                        <span>${(result.raw_probabilities.pneumonia * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="medical-progress">
                                        <div class="medical-progress-bar" style="width: ${result.raw_probabilities.pneumonia * 100}%; background: var(--medical-danger);"></div>
                                    </div>
                                </div>
                            </div>
                            ` : ''}
                        </div>
                        
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="medical-label">Processing Time</label>
                                <div class="medical-stat-value">${MedicalCore.utils.formatDuration(processingTime)}</div>
                            </div>
                            
                            <div class="mb-3">
                                <label class="medical-label">Model Information</label>
                                <div class="model-info">
                                    <div><strong>Version:</strong> ${result.model_version || 'Unknown'}</div>
                                    <div><strong>Image Size:</strong> ${result.image_width || 'N/A'}×${result.image_height || 'N/A'}</div>
                                    <div><strong>File Size:</strong> ${MedicalCore.utils.formatFileSize(result.file_size || 0)}</div>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label class="medical-label">Medical Recommendation</label>
                                <div class="medical-recommendation">
                                    ${this.getRecommendation(result)}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="medical-card-footer">
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="medical-caption">
                            Analysis completed at ${new Date(result.created_at || Date.now()).toLocaleString()}
                        </small>
                        <div class="btn-group">
                            <button class="medical-btn medical-btn-sm medical-btn-outline" onclick="window.medicalAnalyzer.downloadReport('${result.id}')">
                                📄 Download Report
                            </button>
                            <button class="medical-btn medical-btn-sm medical-btn-outline" onclick="window.medicalAnalyzer.shareResults('${result.id}')">
                                📤 Share Results
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getConfidenceDescription(confidence) {
        if (confidence >= 0.9) return 'Very High Confidence';
        if (confidence >= 0.8) return 'High Confidence';
        if (confidence >= 0.7) return 'Moderate Confidence';
        if (confidence >= 0.6) return 'Fair Confidence';
        return 'Low Confidence - Consider Additional Testing';
    }

    getRecommendation(result) {
        const isPneumonia = result.prediction_class === 'pneumonia' || result.is_pneumonia_detected;
        const confidence = result.confidence_score || (result.confidence_percentage / 100);
        
        if (isPneumonia) {
            if (confidence >= 0.8) {
                return '🏥 <strong>High Confidence Detection:</strong> Consult with a pulmonologist or emergency physician immediately for proper treatment and confirmation.';
            } else {
                return '👨‍⚕️ <strong>Moderate Confidence:</strong> Seek medical attention for further evaluation and additional imaging for confirmation.';
            }
        } else {
            if (confidence >= 0.8) {
                return '✅ <strong>Normal Result:</strong> Chest X-ray appears normal. Continue routine healthcare monitoring as recommended by your physician.';
            } else {
                return '🔍 <strong>Uncertain Result:</strong> Consider additional imaging or consultation if symptoms persist. Further evaluation may be needed.';
            }
        }
    }

    updateConfidenceDisplay(result) {
        if (!this.confidenceDisplay) return;
        
        const confidence = (result.confidence * 100).toFixed(1);
        this.confidenceDisplay.innerHTML = `
            <div class="confidence-meter">
                <div class="confidence-bar" style="width: ${confidence}%;"></div>
                <span class="confidence-text">${confidence}%</span>
            </div>
        `;
    }

    updatePredictionResult(result) {
        if (!this.predictionResult) return;
        
        const isPneumonia = result.prediction === 'PNEUMONIA';
        const statusClass = isPneumonia ? 'medical-text-danger' : 'medical-text-success';
        
        this.predictionResult.innerHTML = `
            <div class="prediction-status ${statusClass}">
                ${isPneumonia ? '⚠️ PNEUMONIA DETECTED' : '✅ NORMAL'}
            </div>
        `;
    }

    updateAnalysisDetails(result, processingTime) {
        if (!this.analysisDetails) return;
        
        this.analysisDetails.innerHTML = `
            <div class="detail-item">
                <span class="detail-label">Processing Time:</span>
                <span class="detail-value">${MedicalCore.utils.formatDuration(processingTime)}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Model Version:</span>
                <span class="detail-value">${result.model_version || 'v1.0'}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Analysis ID:</span>
                <span class="detail-value">${result.analysis_id || 'N/A'}</span>
            </div>
        `;
    }

    addToHistory(analysisRecord) {
        this.analysisHistory.unshift(analysisRecord);
        
        // Limit history size
        if (this.analysisHistory.length > 50) {
            this.analysisHistory = this.analysisHistory.slice(0, 50);
        }
        
        this.updateHistoryDisplay();
        this.saveAnalysisHistory();
    }

    updateHistoryDisplay() {
        if (!this.historyList) return;
        
        if (this.analysisHistory.length === 0) {
            this.historyList.innerHTML = `
                <div class="text-center py-4">
                    <p class="medical-caption">No analysis history yet</p>
                </div>
            `;
            return;
        }
        
        const historyHtml = this.analysisHistory.map(record => `
            <div class="history-item medical-card mb-3" data-id="${record.id}">
                <div class="medical-card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${record.fileName}</h6>
                            <p class="medical-caption mb-2">
                                ${new Date(record.completedAt).toLocaleString()}
                            </p>
                            <span class="badge ${record.result.prediction === 'PNEUMONIA' ? 'bg-danger' : 'bg-success'}">
                                ${record.result.prediction}
                            </span>
                        </div>
                        <div class="text-end">
                            <div class="medical-stat-value">${(record.result.confidence * 100).toFixed(1)}%</div>
                            <div class="medical-stat-label">Confidence</div>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
        
        this.historyList.innerHTML = historyHtml;
    }

    updateSessionStats(analysisRecord) {
        this.sessionStats.totalAnalyses++;
        
        if (analysisRecord.status === 'completed') {
            this.sessionStats.successfulAnalyses++;
            
            // Update averages - handle Django format
            const result = analysisRecord.result;
            const newConfidence = result.confidence_score || (result.confidence_percentage / 100);
            this.sessionStats.averageConfidence = 
                (this.sessionStats.averageConfidence * (this.sessionStats.successfulAnalyses - 1) + newConfidence) / 
                this.sessionStats.successfulAnalyses;
            
            this.sessionStats.averageProcessingTime = 
                (this.sessionStats.averageProcessingTime * (this.sessionStats.successfulAnalyses - 1) + analysisRecord.processingTime) / 
                this.sessionStats.successfulAnalyses;
            
            // Update diagnosis counts - handle Django format
            const isPneumonia = result.prediction_class === 'pneumonia' || result.is_pneumonia_detected;
            if (isPneumonia) {
                this.sessionStats.pneumoniaDetected++;
            } else {
                this.sessionStats.normalResults++;
            }
        }
        
        this.updateStatsDisplay();
    }

    updateStatsDisplay() {
        if (!this.statsContainer) return;
        
        const stats = this.sessionStats;
        const successRate = stats.totalAnalyses > 0 ? 
            (stats.successfulAnalyses / stats.totalAnalyses * 100).toFixed(1) : 0;
        
        this.statsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="medical-stat-card">
                        <div class="medical-stat-value">${stats.totalAnalyses}</div>
                        <div class="medical-stat-label">Total Analyses</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="medical-stat-card">
                        <div class="medical-stat-value">${stats.pneumoniaDetected}</div>
                        <div class="medical-stat-label">Pneumonia Cases</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="medical-stat-card">
                        <div class="medical-stat-value">${(stats.averageConfidence * 100).toFixed(1)}%</div>
                        <div class="medical-stat-label">Avg Confidence</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="medical-stat-card">
                        <div class="medical-stat-value">${MedicalCore.utils.formatDuration(stats.averageProcessingTime)}</div>
                        <div class="medical-stat-label">Avg Processing</div>
                    </div>
                </div>
            </div>
        `;
    }

    async clearHistory() {
        try {
            const response = await MedicalCore.api.clearAnalysisHistory();
            
            if (response.success) {
                this.analysisHistory = [];
                this.sessionStats = {
                    totalAnalyses: 0,
                    successfulAnalyses: 0,
                    averageConfidence: 0,
                    averageProcessingTime: 0,
                    pneumoniaDetected: 0,
                    normalResults: 0
                };
                
                this.updateHistoryDisplay();
                this.updateStatsDisplay();
                
                MedicalCore.notifications.success('Analysis history cleared successfully');
            }
        } catch (error) {
            MedicalCore.errors.handle(error, { operation: 'clear-history' });
        }
    }

    saveAnalysisHistory() {
        try {
            localStorage.setItem('pneumonia_analysis_history', JSON.stringify(this.analysisHistory));
            localStorage.setItem('pneumonia_session_stats', JSON.stringify(this.sessionStats));
        } catch (error) {
            console.warn('Could not save analysis history to localStorage:', error);
        }
    }

    loadAnalysisHistory() {
        try {
            const history = localStorage.getItem('pneumonia_analysis_history');
            const stats = localStorage.getItem('pneumonia_session_stats');
            
            if (history) {
                this.analysisHistory = JSON.parse(history);
                this.updateHistoryDisplay();
            }
            
            if (stats) {
                this.sessionStats = { ...this.sessionStats, ...JSON.parse(stats) };
                this.updateStatsDisplay();
            }
        } catch (error) {
            console.warn('Could not load analysis history from localStorage:', error);
        }
    }

    // Public API Methods
    async startAnalysis(file) {
        if (file) {
            await this.processFile(file);
        }
    }

    getAnalysisHistory() {
        return [...this.analysisHistory];
    }

    getSessionStats() {
        return { ...this.sessionStats };
    }

    // Report and sharing functionality
    downloadReport(analysisId) {
        MedicalCore.notifications.info('Report download feature coming soon!', { title: 'Feature Preview' });
        console.log('Download report for analysis:', analysisId);
    }

    shareResults(analysisId) {
        MedicalCore.notifications.info('Results sharing feature coming soon!', { title: 'Feature Preview' });
        console.log('Share results for analysis:', analysisId);
    }

    updateVisualizationDisplay(result, fileName) {
        const visualizationSection = document.getElementById('visualizationSection');
        if (!visualizationSection) return;

        // Check if visualization data is available
        if (result.visualization && !result.visualization.error) {
            const originalImage = document.getElementById('originalImage');
            const heatmapOverlay = document.getElementById('heatmapOverlay');
            const aiExplanation = document.getElementById('aiExplanation');

            // Display original image (use uploaded file as fallback)
            if (originalImage) {
                originalImage.src = result.original_image_url || '#';
                originalImage.alt = `Original X-ray: ${fileName}`;
            }

            // Display heatmap if available
            if (heatmapOverlay && result.visualization.visualization) {
                // Convert PIL Image to base64 data URL if needed
                if (typeof result.visualization.visualization === 'string') {
                    heatmapOverlay.src = result.visualization.visualization;
                } else {
                    heatmapOverlay.src = '#'; // Placeholder for now
                }
                heatmapOverlay.alt = `AI Heatmap: ${fileName}`;
            }

            // Display AI explanation
            if (aiExplanation && result.visualization.explanation) {
                aiExplanation.textContent = result.visualization.explanation;
            }

            // Show visualization section
            visualizationSection.style.display = 'block';
            MedicalCore.animations.fadeIn(visualizationSection);
        } else {
            // Hide visualization section if no data
            visualizationSection.style.display = 'none';
        }
    }

    updatePathologyDisplay(result) {
        const pathologySection = document.getElementById('pathologySection');
        const pathologyFindings = document.getElementById('pathologyFindings');
        
        if (!pathologySection || !pathologyFindings) return;

        // Check if pathology data is available and user wants to see it
        const showPathologyDetails = document.getElementById('showPathologyDetails');
        const shouldShow = showPathologyDetails && showPathologyDetails.checked;

        if (shouldShow && result.pathology_predictions) {
            // Sort pathology findings by probability
            const findings = Object.entries(result.pathology_predictions)
                .sort(([,a], [,b]) => b - a)
                .filter(([,prob]) => prob > 0.1); // Only show significant findings

            if (findings.length > 0) {
                const findingsHtml = findings.map(([pathology, probability]) => {
                    const percentage = (probability * 100).toFixed(1);
                    const significance = probability > 0.7 ? 'high' : probability > 0.5 ? 'moderate' : 'low';
                    const badgeClass = significance === 'high' ? 'bg-danger' : 
                                     significance === 'moderate' ? 'bg-warning' : 'bg-secondary';
                    
                    return `
                        <div class="row align-items-center mb-3">
                            <div class="col-6">
                                <span class="medical-subheading">${pathology.replace(/_/g, ' ')}</span>
                            </div>
                            <div class="col-3">
                                <div class="medical-progress">
                                    <div class="medical-progress-bar" style="width: ${percentage}%"></div>
                                </div>
                            </div>
                            <div class="col-3 text-end">
                                <span class="badge ${badgeClass}">${percentage}%</span>
                            </div>
                        </div>
                    `;
                }).join('');

                pathologyFindings.innerHTML = `
                    <div class="alert alert-info mb-3">
                        <i class="bi bi-info-circle me-2"></i>
                        <strong>Pathology Analysis:</strong> This detailed breakdown shows the AI's assessment 
                        of various lung conditions. Higher percentages indicate stronger detection confidence.
                    </div>
                    ${findingsHtml}
                `;

                pathologySection.style.display = 'block';
                MedicalCore.animations.fadeIn(pathologySection);
            } else {
                pathologySection.style.display = 'none';
            }
        } else {
            pathologySection.style.display = 'none';
        }
    }

    destroy() {
        this.saveAnalysisHistory();
        super.destroy();
    }
}

// Make available globally
window.PneumoniaAnalyzer = PneumoniaAnalyzer;