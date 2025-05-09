// static/js/main.js
document.addEventListener('DOMContentLoaded', function () {
    console.log("Boutique AI Website JS Initialized");

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            try {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            } catch (error) {
                console.warn('Smooth scroll target not found or invalid selector:', targetId);
            }
        });
    });

    // Mobile menu toggle (if not already in navbar.html script tag)
    const menuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    const menuIconOpen = document.getElementById('menu-icon-open');
    const menuIconClose = document.getElementById('menu-icon-close');

    if (menuButton && mobileMenu && menuIconOpen && menuIconClose) {
        menuButton.addEventListener('click', () => {
            const expanded = menuButton.getAttribute('aria-expanded') === 'true' || false;
            menuButton.setAttribute('aria-expanded', String(!expanded));
            mobileMenu.classList.toggle('hidden');
            menuIconOpen.classList.toggle('hidden');
            menuIconClose.classList.toggle('hidden');
        });
    }

    // Example: Simple fade-in for elements with class 'fade-in-on-load'
    const fadeElements = document.querySelectorAll('.fade-in-on-load');
    fadeElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transition = `opacity 0.5s ease-in-out ${index * 0.1}s`;
        setTimeout(() => {
            el.style.opacity = '1';
        }, 100); // Slight delay to ensure transition applies
    });

});