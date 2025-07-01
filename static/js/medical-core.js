/**
 * ========================================
 * PNEUMONIA AI - MEDICAL CORE JAVASCRIPT
 * Professional Medical Interface System
 * Version: 2.0.0
 * ========================================
 */

'use strict';

/**
 * Medical Core Namespace
 * Provides professional-grade utilities and base classes
 */
const MedicalCore = (() => {
    
    /**
     * Configuration Management System
     */
    class ConfigManager {
        constructor() {
            this.config = {
                api: {
                    baseUrl: '/api',
                    timeout: 30000,
                    retryAttempts: 3,
                    retryDelay: 1000
                },
                ui: {
                    animationDuration: 300,
                    toastDuration: 5000,
                    progressUpdateInterval: 100
                },
                upload: {
                    maxFileSize: 10 * 1024 * 1024, // 10MB
                    allowedTypes: ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'],
                    chunkSize: 1024 * 1024 // 1MB chunks
                },
                analysis: {
                    confidenceThreshold: 0.7,
                    maxAnalysisTime: 60000,
                    pollingInterval: 500
                }
            };
        }

        get(path) {
            return path.split('.').reduce((obj, key) => obj?.[key], this.config);
        }

        set(path, value) {
            const keys = path.split('.');
            const lastKey = keys.pop();
            const target = keys.reduce((obj, key) => obj[key] = obj[key] || {}, this.config);
            target[lastKey] = value;
        }
    }

    /**
     * Enhanced Event System
     */
    class EventManager {
        constructor() {
            this.events = new Map();
            this.eventHistory = [];
            this.maxHistorySize = 100;
        }

        on(eventName, callback, options = {}) {
            if (!this.events.has(eventName)) {
                this.events.set(eventName, []);
            }
            
            const listener = {
                callback,
                once: options.once || false,
                priority: options.priority || 0,
                id: this.generateId()
            };
            
            this.events.get(eventName).push(listener);
            this.events.get(eventName).sort((a, b) => b.priority - a.priority);
            
            return listener.id;
        }

        off(eventName, listenerId) {
            if (!this.events.has(eventName)) return false;
            
            const listeners = this.events.get(eventName);
            const index = listeners.findIndex(l => l.id === listenerId);
            
            if (index !== -1) {
                listeners.splice(index, 1);
                return true;
            }
            return false;
        }

        emit(eventName, data = {}) {
            const eventData = {
                name: eventName,
                data,
                timestamp: Date.now(),
                id: this.generateId()
            };
            
            this.addToHistory(eventData);
            
            if (!this.events.has(eventName)) return [];
            
            const listeners = this.events.get(eventName);
            const results = [];
            
            for (let i = listeners.length - 1; i >= 0; i--) {
                const listener = listeners[i];
                
                try {
                    const result = listener.callback(eventData);
                    results.push(result);
                    
                    if (listener.once) {
                        listeners.splice(i, 1);
                    }
                } catch (error) {
                    console.error(`Event listener error for ${eventName}:`, error);
                }
            }
            
            return results;
        }

        addToHistory(eventData) {
            this.eventHistory.push(eventData);
            if (this.eventHistory.length > this.maxHistorySize) {
                this.eventHistory.shift();
            }
        }

        generateId() {
            return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        }
    }

    /**
     * Professional API Client with advanced features
     */
    class APIClient {
        constructor(config) {
            this.config = config;
            this.requestInterceptors = [];
            this.responseInterceptors = [];
            this.retryConfig = {
                attempts: config.get('api.retryAttempts'),
                delay: config.get('api.retryDelay')
            };
        }

        addRequestInterceptor(interceptor) {
            this.requestInterceptors.push(interceptor);
        }

        addResponseInterceptor(interceptor) {
            this.responseInterceptors.push(interceptor);
        }

        async request(endpoint, options = {}) {
            const config = {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                timeout: this.config.get('api.timeout'),
                ...options
            };

            // Apply request interceptors
            for (const interceptor of this.requestInterceptors) {
                await interceptor(config);
            }

            return this.executeWithRetry(endpoint, config);
        }

        async executeWithRetry(endpoint, config, attempt = 1) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.timeout);

                const response = await fetch(`${this.config.get('api.baseUrl')}${endpoint}`, {
                    ...config,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                // Apply response interceptors
                for (const interceptor of this.responseInterceptors) {
                    await interceptor(response);
                }

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                return { success: true, data, response };

            } catch (error) {
                if (attempt < this.retryConfig.attempts && this.shouldRetry(error)) {
                    await this.delay(this.retryConfig.delay * attempt);
                    return this.executeWithRetry(endpoint, config, attempt + 1);
                }
                
                return { success: false, error, attempt };
            }
        }

        shouldRetry(error) {
            return error.name === 'AbortError' || 
                   error.message.includes('Failed to fetch') ||
                   error.message.includes('5');
        }

        async delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        getCSRFToken() {
            return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
        }

        // Specialized medical API methods
        async analyzeImage(formData, progressCallback) {
            const config = {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': this.getCSRFToken()
                }
            };

            // Remove Content-Type header for FormData
            delete config.headers['Content-Type'];

            return this.request('/analyze/', config);
        }

        async getModelInfo() {
            return this.request('/model-info/');
        }

        async getSystemStats() {
            return this.request('/stats/');
        }

        async getAnalysisHistory() {
            return this.request('/analyses/');
        }

        async clearAnalysisHistory() {
            return this.request('/analyses/clear_session/', { method: 'DELETE' });
        }
    }

    /**
     * Professional UI Component Base Class
     */
    class UIComponent {
        constructor(element, options = {}) {
            this.element = typeof element === 'string' ? document.querySelector(element) : element;
            this.options = { ...this.defaultOptions, ...options };
            this.eventManager = new EventManager();
            this.state = {};
            this.initialized = false;
            
            if (this.element) {
                this.init();
            }
        }

        get defaultOptions() {
            return {};
        }

        init() {
            this.bindEvents();
            this.render();
            this.initialized = true;
            this.emit('initialized');
        }

        bindEvents() {
            // Override in subclasses
        }

        render() {
            // Override in subclasses
        }

        setState(newState) {
            const oldState = { ...this.state };
            this.state = { ...this.state, ...newState };
            this.onStateChange(oldState, this.state);
            this.emit('stateChange', { oldState, newState: this.state });
        }

        onStateChange(oldState, newState) {
            // Override in subclasses
        }

        emit(eventName, data) {
            return this.eventManager.emit(eventName, data);
        }

        on(eventName, callback, options) {
            return this.eventManager.on(eventName, callback, options);
        }

        off(eventName, listenerId) {
            return this.eventManager.off(eventName, listenerId);
        }

        destroy() {
            this.emit('destroy');
            this.eventManager.events.clear();
            this.element = null;
        }
    }

    /**
     * Enhanced Animation System
     */
    class AnimationManager {
        constructor() {
            this.animations = new Map();
            this.timeline = [];
        }

        animate(element, keyframes, options = {}) {
            const animationId = this.generateId();
            
            const defaultOptions = {
                duration: 300,
                easing: 'ease-out',
                fill: 'both'
            };

            const animation = element.animate(keyframes, { ...defaultOptions, ...options });
            
            this.animations.set(animationId, animation);
            
            animation.addEventListener('finish', () => {
                this.animations.delete(animationId);
            });

            return { animation, id: animationId };
        }

        fadeIn(element, duration = 300) {
            return this.animate(element, [
                { opacity: 0, transform: 'translateY(20px)' },
                { opacity: 1, transform: 'translateY(0)' }
            ], { duration });
        }

        fadeOut(element, duration = 300) {
            return this.animate(element, [
                { opacity: 1 },
                { opacity: 0 }
            ], { duration });
        }

        slideUp(element, duration = 400) {
            return this.animate(element, [
                { opacity: 0, transform: 'translateY(100%)' },
                { opacity: 1, transform: 'translateY(0)' }
            ], { duration, easing: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)' });
        }

        scaleIn(element, duration = 300) {
            return this.animate(element, [
                { opacity: 0, transform: 'scale(0.8)' },
                { opacity: 1, transform: 'scale(1)' }
            ], { duration });
        }

        pulse(element, duration = 1000, iterations = Infinity) {
            return this.animate(element, [
                { opacity: 1 },
                { opacity: 0.7 },
                { opacity: 1 }
            ], { duration, iterations });
        }

        generateId() {
            return `anim-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        }
    }

    /**
     * Professional Notification System
     */
    class NotificationManager {
        constructor() {
            this.container = this.createContainer();
            this.notifications = [];
            this.maxNotifications = 5;
        }

        createContainer() {
            let container = document.querySelector('.medical-notification-container');
            
            if (!container) {
                container = document.createElement('div');
                container.className = 'medical-notification-container';
                container.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 9999;
                    max-width: 400px;
                `;
                document.body.appendChild(container);
            }
            
            return container;
        }

        show(message, type = 'info', options = {}) {
            const notification = this.createNotification(message, type, options);
            
            if (this.notifications.length >= this.maxNotifications) {
                this.remove(this.notifications[0]);
            }
            
            this.container.appendChild(notification.element);
            this.notifications.push(notification);
            
            // Animate in
            requestAnimationFrame(() => {
                notification.element.style.transform = 'translateX(0)';
                notification.element.style.opacity = '1';
            });
            
            // Auto-remove
            if (options.duration !== 0) {
                setTimeout(() => this.remove(notification), options.duration || 5000);
            }
            
            return notification;
        }

        createNotification(message, type, options) {
            const element = document.createElement('div');
            element.className = `medical-toast medical-toast-${type}`;
            element.style.cssText = `
                transform: translateX(100%);
                opacity: 0;
                transition: all 0.3s ease-out;
                margin-bottom: 10px;
            `;
            
            const iconMap = {
                success: '✓',
                warning: '⚠',
                error: '✕',
                info: 'ℹ'
            };
            
            element.innerHTML = `
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 18px; font-weight: bold;">${iconMap[type] || iconMap.info}</span>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; margin-bottom: 4px;">${options.title || type.toUpperCase()}</div>
                        <div style="font-size: 14px; opacity: 0.9;">${message}</div>
                    </div>
                    ${options.closable !== false ? '<button style="background: none; border: none; font-size: 18px; cursor: pointer; opacity: 0.7;">×</button>' : ''}
                </div>
            `;
            
            const closeBtn = element.querySelector('button');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => this.remove({ element }));
            }
            
            return { element, type, message, options };
        }

        remove(notification) {
            if (!notification.element.parentNode) return;
            
            notification.element.style.transform = 'translateX(100%)';
            notification.element.style.opacity = '0';
            
            setTimeout(() => {
                if (notification.element.parentNode) {
                    notification.element.remove();
                }
                const index = this.notifications.indexOf(notification);
                if (index > -1) {
                    this.notifications.splice(index, 1);
                }
            }, 300);
        }

        success(message, options = {}) {
            return this.show(message, 'success', options);
        }

        warning(message, options = {}) {
            return this.show(message, 'warning', options);
        }

        error(message, options = {}) {
            return this.show(message, 'error', options);
        }

        info(message, options = {}) {
            return this.show(message, 'info', options);
        }
    }

    /**
     * Data Validation System
     */
    class ValidationManager {
        constructor() {
            this.rules = new Map();
            this.customValidators = new Map();
        }

        addRule(field, validator, message) {
            if (!this.rules.has(field)) {
                this.rules.set(field, []);
            }
            this.rules.get(field).push({ validator, message });
        }

        addCustomValidator(name, validator) {
            this.customValidators.set(name, validator);
        }

        validate(data) {
            const errors = {};
            
            for (const [field, validators] of this.rules) {
                const value = data[field];
                
                for (const { validator, message } of validators) {
                    if (!validator(value, data)) {
                        if (!errors[field]) errors[field] = [];
                        errors[field].push(message);
                    }
                }
            }
            
            return {
                isValid: Object.keys(errors).length === 0,
                errors
            };
        }

        // Built-in validators
        static validators = {
            required: (value) => value !== null && value !== undefined && value !== '',
            email: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
            minLength: (min) => (value) => value && value.length >= min,
            maxLength: (max) => (value) => value && value.length <= max,
            isFile: (value) => value instanceof File,
            fileSize: (maxSize) => (value) => value instanceof File && value.size <= maxSize,
            fileType: (types) => (value) => value instanceof File && types.includes(value.type)
        };
    }

    /**
     * Professional Error Handler
     */
    class ErrorHandler {
        constructor(notificationManager) {
            this.notificationManager = notificationManager;
            this.errorLog = [];
            this.maxLogSize = 100;
        }

        handle(error, context = {}) {
            const errorData = {
                message: error.message || 'Unknown error',
                stack: error.stack,
                context,
                timestamp: new Date().toISOString(),
                id: this.generateId()
            };
            
            this.log(errorData);
            this.displayError(errorData);
            
            return errorData;
        }

        log(errorData) {
            this.errorLog.push(errorData);
            if (this.errorLog.length > this.maxLogSize) {
                this.errorLog.shift();
            }
            
            console.error('Medical App Error:', errorData);
        }

        displayError(errorData) {
            const userMessage = this.getUserFriendlyMessage(errorData);
            this.notificationManager.error(userMessage, {
                title: 'System Error',
                duration: 8000
            });
        }

        getUserFriendlyMessage(errorData) {
            const { message, context } = errorData;
            
            if (message.includes('network') || message.includes('fetch')) {
                return 'Network connection issue. Please check your internet connection.';
            }
            
            if (message.includes('timeout')) {
                return 'The operation took too long. Please try again.';
            }
            
            if (context.operation === 'file-upload') {
                return 'File upload failed. Please check the file format and size.';
            }
            
            return 'An unexpected error occurred. Please try again or contact support.';
        }

        generateId() {
            return `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        }
    }

    // Initialize core system
    const config = new ConfigManager();
    const eventManager = new EventManager();
    const apiClient = new APIClient(config);
    const animationManager = new AnimationManager();
    const notificationManager = new NotificationManager();
    const validationManager = new ValidationManager();
    const errorHandler = new ErrorHandler(notificationManager);

    // Public API
    return {
        ConfigManager,
        EventManager,
        APIClient,
        UIComponent,
        AnimationManager,
        NotificationManager,
        ValidationManager,
        ErrorHandler,
        
        // Instances
        config,
        events: eventManager,
        api: apiClient,
        animations: animationManager,
        notifications: notificationManager,
        validation: validationManager,
        errors: errorHandler,
        
        // Utilities
        utils: {
            debounce: (func, delay) => {
                let timeoutId;
                return (...args) => {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => func.apply(null, args), delay);
                };
            },
            
            throttle: (func, limit) => {
                let inThrottle;
                return (...args) => {
                    if (!inThrottle) {
                        func.apply(null, args);
                        inThrottle = true;
                        setTimeout(() => inThrottle = false, limit);
                    }
                };
            },
            
            formatFileSize: (bytes) => {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            },
            
            formatDuration: (ms) => {
                const seconds = Math.floor(ms / 1000);
                const minutes = Math.floor(seconds / 60);
                return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
            },
            
            generateId: () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        }
    };
})();

// Make available globally
window.MedicalCore = MedicalCore;