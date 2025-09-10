// MeowTrix-AI Application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    initializeNavigation();
    initializeExampleTabs();
    initializeProgressBars();
    initializeMetricCards();
    addSmoothScrolling();
}

// Navigation Tab Functionality
function initializeNavigation() {
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    navTabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            const targetTab = this.dataset.tab;
            
            // Remove active class from all tabs and contents
            navTabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            this.classList.add('active');
            const targetContent = document.getElementById(targetTab);
            
            if (targetContent) {
                targetContent.classList.add('active');
                
                // Animate content appearance
                targetContent.style.opacity = '0';
                targetContent.style.transform = 'translateY(20px)';
                
                setTimeout(() => {
                    targetContent.style.transition = 'all 0.3s ease';
                    targetContent.style.opacity = '1';
                    targetContent.style.transform = 'translateY(0)';
                }, 50);
                
                // Scroll to top of content
                const mainContent = document.querySelector('.main-content');
                if (mainContent) {
                    mainContent.scrollIntoView({ 
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
                
                // Trigger animations for the newly visible tab
                if (targetTab === 'performance') {
                    setTimeout(() => {
                        triggerPerformanceAnimations();
                    }, 300);
                }
            } else {
                console.warn(`Target content not found for tab: ${targetTab}`);
            }
        });
    });
}

// Example Tabs in Getting Started Section
function initializeExampleTabs() {
    const exampleTabs = document.querySelectorAll('.example-tab');
    const exampleContents = document.querySelectorAll('.example-content');

    exampleTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetExample = this.dataset.example;
            
            // Remove active class from all example tabs and contents
            exampleTabs.forEach(t => t.classList.remove('active'));
            exampleContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            this.classList.add('active');
            const targetContent = document.getElementById(targetExample + '-example');
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
}

// Trigger performance tab animations
function triggerPerformanceAnimations() {
    animateProgressBars();
    animateMetricCards();
    animateMatrix();
}

// Animate Progress Bars
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    
    // Function to animate progress bars when they come into view
    const animateProgressBars = () => {
        progressBars.forEach(bar => {
            const rect = bar.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
            
            if (isVisible && !bar.classList.contains('animated')) {
                const targetWidth = bar.style.width;
                bar.style.width = '0%';
                bar.classList.add('animated');
                
                setTimeout(() => {
                    bar.style.width = targetWidth;
                }, 100);
            }
        });
    };

    // Check on scroll and initial load
    window.addEventListener('scroll', animateProgressBars);
}

// Separate function to trigger progress bar animation
function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    
    progressBars.forEach((bar, index) => {
        if (!bar.classList.contains('animated')) {
            const targetWidth = bar.style.width;
            bar.style.width = '0%';
            bar.classList.add('animated');
            
            setTimeout(() => {
                bar.style.transition = 'width 1s ease';
                bar.style.width = targetWidth;
            }, index * 100);
        }
    });
}

// Animate Metric Cards
function initializeMetricCards() {
    const metricCards = document.querySelectorAll('.metric-card');
    
    const animateMetrics = () => {
        metricCards.forEach((card, index) => {
            const rect = card.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
            
            if (isVisible && !card.classList.contains('animated')) {
                animateMetricCard(card, index);
            }
        });
    };

    window.addEventListener('scroll', animateMetrics);
}

// Separate function to animate metric cards
function animateMetricCards() {
    const metricCards = document.querySelectorAll('.metric-card');
    
    metricCards.forEach((card, index) => {
        if (!card.classList.contains('animated')) {
            animateMetricCard(card, index);
        }
    });
}

function animateMetricCard(card, index) {
    card.classList.add('animated');
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    
    setTimeout(() => {
        card.style.transition = 'all 0.5s ease';
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
        
        // Animate the metric value
        const metricValue = card.querySelector('.metric-value');
        if (metricValue) {
            setTimeout(() => {
                animateValue(metricValue);
            }, 200);
        }
    }, index * 100);
}

// Animate numeric values
function animateValue(element) {
    const text = element.textContent;
    const isPercentage = text.includes('%');
    const numericValue = parseInt(text.replace(/[^\d]/g, ''));
    
    if (isNaN(numericValue)) return;
    
    let currentValue = 0;
    const increment = Math.max(1, numericValue / 30); // Animation duration
    const timer = setInterval(() => {
        currentValue += increment;
        if (currentValue >= numericValue) {
            currentValue = numericValue;
            clearInterval(timer);
        }
        
        element.textContent = isPercentage 
            ? Math.floor(currentValue) + '%'
            : Math.floor(currentValue) + (text.includes('GB') ? 'GB' : '');
    }, 50);
}

// Add smooth scrolling for anchor links
function addSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Add interactive hover effects
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to feature items
    const featureItems = document.querySelectorAll('.feature-item');
    featureItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px) scale(1.02)';
            this.style.transition = 'all 0.3s ease';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Add hover effects to tech cards
    const techCards = document.querySelectorAll('.tech-card');
    techCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'all 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Add click effects to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('click', function() {
            // Add a pulse effect
            this.style.transform = 'scale(0.95)';
            this.style.transition = 'transform 0.15s ease';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);
        });
    });

    // Add interactive effects to component cards
    const componentCards = document.querySelectorAll('.component-card');
    componentCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            const techTags = this.querySelectorAll('.tech-tag');
            techTags.forEach((tag, index) => {
                setTimeout(() => {
                    tag.style.transform = 'scale(1.1)';
                    tag.style.transition = 'transform 0.2s ease';
                }, index * 50);
            });
        });
        
        card.addEventListener('mouseleave', function() {
            const techTags = this.querySelectorAll('.tech-tag');
            techTags.forEach(tag => {
                tag.style.transform = 'scale(1)';
            });
        });
    });

    // Add typing effect to code blocks
    const codeBlocks = document.querySelectorAll('.code-block code');
    const observerOptions = {
        threshold: 0.3,
        rootMargin: '0px 0px -50px 0px'
    };

    const codeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('typed')) {
                typeCode(entry.target);
            }
        });
    }, observerOptions);

    codeBlocks.forEach(block => {
        codeObserver.observe(block);
    });
});

// Typing effect for code blocks
function typeCode(element) {
    element.classList.add('typed');
    const text = element.textContent;
    element.textContent = '';
    element.style.borderRight = '2px solid var(--color-primary)';
    
    let index = 0;
    const typeInterval = setInterval(() => {
        if (index < text.length) {
            element.textContent += text.charAt(index);
            index++;
        } else {
            clearInterval(typeInterval);
            element.style.borderRight = 'none';
        }
    }, 20);
}

// Add matrix animation for confusion matrix
function animateMatrix() {
    const matrixCells = document.querySelectorAll('.matrix-cell');
    matrixCells.forEach((cell, index) => {
        if (!cell.classList.contains('matrix-animated')) {
            cell.classList.add('matrix-animated');
            cell.style.opacity = '0';
            cell.style.transform = 'scale(0.8)';
            
            setTimeout(() => {
                cell.style.transition = 'all 0.4s ease';
                cell.style.opacity = '1';
                cell.style.transform = 'scale(1)';
                
                // Animate the value
                const valueElement = cell.querySelector('.cell-value');
                if (valueElement) {
                    setTimeout(() => {
                        animateValue(valueElement);
                    }, 200);
                }
            }, index * 150);
        }
    });
}

// Add scroll-triggered animations
window.addEventListener('scroll', () => {
    // Only animate visible elements to improve performance
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab && activeTab.id === 'performance') {
        const rect = activeTab.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
            animateMatrix();
        }
    }
});

// Add keyboard navigation support
document.addEventListener('keydown', function(e) {
    // Only handle keyboard navigation if focus is on navigation area
    const focusedElement = document.activeElement;
    const isNavFocused = focusedElement.classList.contains('nav-tab');
    
    if (isNavFocused && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
        e.preventDefault();
        const navTabs = document.querySelectorAll('.nav-tab');
        const currentIndex = Array.from(navTabs).indexOf(focusedElement);
        let nextIndex;
        
        if (e.key === 'ArrowRight') {
            nextIndex = (currentIndex + 1) % navTabs.length;
        } else {
            nextIndex = (currentIndex - 1 + navTabs.length) % navTabs.length;
        }
        
        navTabs[nextIndex].click();
        navTabs[nextIndex].focus();
    }
});

// Add performance monitoring
function addPerformanceMonitoring() {
    let lastScrollTime = Date.now();
    let ticking = false;
    
    function updateAnimations() {
        // Batch DOM updates here if needed
        ticking = false;
    }
    
    window.addEventListener('scroll', () => {
        const now = Date.now();
        if (now - lastScrollTime > 16) { // ~60fps
            lastScrollTime = now;
            
            if (!ticking) {
                requestAnimationFrame(updateAnimations);
                ticking = true;
            }
        }
    });
}

// Initialize performance monitoring
addPerformanceMonitoring();

// Add error handling
window.addEventListener('error', function(e) {
    console.error('Application error:', e.error);
    // Could add user-friendly error display here
});

// Add loading states for images
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
        if (!img.complete) {
            img.style.opacity = '0';
            img.addEventListener('load', function() {
                this.style.transition = 'opacity 0.3s ease';
                this.style.opacity = '1';
            });
            
            img.addEventListener('error', function() {
                console.warn('Failed to load image:', this.src);
                this.style.opacity = '0.3';
                this.alt = 'Image unavailable';
            });
        }
    });
});

// Export functions for potential testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeApp,
        initializeNavigation,
        initializeExampleTabs,
        animateValue
    };
}